import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Optional


class AttentionalGCNLayer(nn.Module):
    """
    Single graph convolutional layer with attention over edges.

    The standard operation is:
        S @ x @ W

    Here we replace S with an attention-reweighted version:
        S_tilde(x) = S ⊙ A(x)

    where A(x)[i, j] is an attention coefficient computed from node features,
    and ⊙ denotes element-wise multiplication. The final operation is:

        out = S_tilde(x) @ x @ W

    Parameters
    ----------
    in_features : int
        Number of input features per node.
    out_features : int
        Number of output features per node.
    activation : callable, optional
        Optional activation function to apply after the linear transformation.
        It must accept and return a `torch.Tensor`. If None, no activation is
        applied inside the layer (default: None).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        # Standard GCN weight matrix (as in your original GCNLayer).
        self.weights = nn.Parameter(torch.empty(self.in_features, self.out_features))

        # Learnable attention matrix U used in a bilinear form:
        #   e_ij = h_i^T U h_j
        # where h = x @ W are transformed node features.
        self.att_matrix = nn.Parameter(
            torch.empty(self.out_features, self.out_features)
        )

        self.init_parameters()

    def init_parameters(self) -> None:
        """
        Initialize layer parameters with Xavier uniform initialization.
        """
        nn.init.xavier_uniform_(self.weights)
        nn.init.xavier_uniform_(self.att_matrix)

    def forward(self, x: Tensor, S: Tensor) -> Tensor:
        """
        Apply the attentional graph convolution.

        Steps:
        1. Add self-loops to S.
        2. Linearly transform features: h = x @ W.
        3. Compute attention scores:
               e_ij = h_i^T U h_j
           using the learnable matrix U = att_matrix.
        4. Mask scores with S (only neighbors attend), then row-wise softmax.
        5. Build attention-weighted shift:
               S_tilde = S ⊙ alpha
        6. Convolve:
               out = S_tilde @ h
        """
        device = x.device

        # 1. Add self-loops.
        S = S + torch.eye(x.size(0), device=device)  # (N, N)

        # 2. Linear transform: h = x W
        h = x @ self.weights  # (N, out_features)

        # 3. Bilinear attention scores: e_ij = h_i^T U h_j
        #    scores shape: (N, N)
        #    (h @ U) -> (N, out_features), then (h @ U) @ h^T -> (N, N)
        scores = (h @ self.att_matrix) @ h.T  # (N, N)

        # 4. Mask non-edges and normalize with softmax.
        #    Use S > 0 to keep only existing edges (including self-loops).
        mask = (S > 0).float()  # (N, N)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        att = torch.softmax(scores, dim=-1)  # (N, N)
        att = att * mask  # ensure non-edges remain exactly zero

        # 5. Attention-weighted shift operator.
        S_tilde = S * att  # (N, N)

        # 6. Graph convolution with attention-weighted shift.
        out = S_tilde @ h  # (N, out_features)

        if self.activation is not None:
            out = self.activation(out)

        return out


class AttentionalGCNModel(nn.Module):
    """
    Two-layer Graph Convolutional Network (GCN) with attention in each layer.

    This model consists of:
        - One hidden attentional graph convolutional layer with ReLU activation.
        - One output attentional graph convolutional layer without activation.

    Parameters
    ----------
    in_channels : int
        Number of input features per node.
    hidden_channels : int
        Number of hidden units in the first GCN layer.
    out_channels : int
        Number of output classes (for node classification) or features.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.conv1 = AttentionalGCNLayer(
            in_features=in_channels,
            out_features=hidden_channels,
            activation=torch.relu,
        )
        self.conv2 = AttentionalGCNLayer(
            in_features=hidden_channels,
            out_features=out_channels,
            activation=None,
        )

    def forward(self, x: Tensor, S: Tensor) -> Tensor:
        """
        Forward pass of the attentional GCN model.

        Parameters
        ----------
        x : torch.Tensor
            Input node feature matrix of shape ``(N, in_channels)``.
        S : torch.Tensor
            Graph shift operator of shape ``(N, N)``.

        Returns
        -------
        torch.Tensor
            Output logits (unnormalized scores) of shape
            ``(N, out_channels)``.
        """
        x = self.conv1(x, S)  # (N, hidden_channels) with attention
        x = self.conv2(x, S)  # (N, out_channels) with attention
        return x
