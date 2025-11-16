# Solutions of Exercise 1 : write a 2-layer Graph Convolutional Neural Network (GCN) for node classification by using:
# - 1-st order Graph Filters at each GCN layer
# - The Adjacency matrix as Graph Shift Operator inside the GConv layers
# - the Relu activation function at the fist GCN layer


import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Optional


class GCNLayer(nn.Module):
    """
    Single graph convolutional layer.

    This layer applies a linear transformation to the node features and then
    propagates them through the (normalized) graph shift operator S.

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

        self.weights = nn.Parameter(torch.empty(self.in_features, self.out_features))
        self.init_parameters()

    def init_parameters(self) -> None:
        """
        Initialize layer parameters with Xavier uniform initialization.

        This is the method described in:
        *Glorot, X., & Bengio, Y. (2010). "Understanding the difficulty of
        training deep feedforward neural networks".*

        Returns
        -------
        None
        """
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x: Tensor, S: Tensor) -> Tensor:
        """
        Apply the graph convolution.

        The operation is:
            S @ x @ W
        where:
            - S is the (normalized) graph shift / adjacency-like matrix,
            - x are the node features,
            - W are the learnable weights.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape ``(N, in_features)``, where
            ``N`` is the number of nodes.
        S : torch.Tensor
            Graph shift operator of shape ``(N, N)``. Typically a (normalized)
            adjacency or Laplacian-derived matrix.

        Returns
        -------
        torch.Tensor
            Output node features of shape ``(N, out_features)``.
        """
        device = x.device
        S = S + torch.eye(x.size(0), device=device)
        out = S @ x @ self.weights
        if self.activation is not None:
            out = self.activation(out)
        return out


class GCNModel(nn.Module):
    """
    Two-layer Graph Convolutional Network (GCN).

    This model consists of:
        - One hidden graph convolutional layer with ReLU activation.
        - One output graph convolutional layer without activation.

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
        self.conv1 = GCNLayer(
            in_features=in_channels,
            out_features=hidden_channels,
            activation=torch.relu,
        )
        self.conv2 = GCNLayer(
            in_features=hidden_channels,
            out_features=out_channels,
            activation=None,
        )

    def forward(self, x: Tensor, S: Tensor) -> Tensor:
        """
        Forward pass of the GCN model.

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
        x = self.conv1(x, S)  # (N, hidden_channels)
        x = self.conv2(x, S)  # (N, out_channels)
        return x
