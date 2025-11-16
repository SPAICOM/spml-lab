# Solutions for Exercise 2: Implement a multi-layer GCN for graph classification extending
# the model from ex2.py with (non-global) pooling layers after each intermediate GCN layer.

import torch
from torch import Tensor
from torch_geometric.nn import (
    GCNConv,
    MLP,
    TopKPooling,
    global_add_pool,
)
from typing import List


class GCN(torch.nn.Module):
    """
    Multi-layer Graph Convolutional Network (GCN) with TopKPooling
    after each layer, followed by an MLP for graph-level classification.

    This model applies:
    - A stack of GCNConv layers with ReLU activation.
    - After each GCNConv, a TopKPooling layer to reduce the number of nodes.
    - Global add pooling to obtain graph embeddings.
    - A final MLP for classification.

    Parameters
    ----------
    in_channels : int
        Number of input features per node.
    hidden_channels : int
        Number of hidden units in each GCN layer.
    out_channels : int
        Number of output classes or features.
    num_layers : int
        Number of GCN layers to apply.
    pool_ratio : float, optional
        Fraction of nodes to keep in each TopKPooling step (0 < ratio <= 1).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        pool_ratio: float = 0.5,
    ) -> None:
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()

        for _ in range(num_layers):
            conv = GCNConv(in_channels, hidden_channels)
            pool = TopKPooling(hidden_channels, ratio=pool_ratio)

            self.convs.append(conv)
            self.pools.append(pool)

            in_channels = hidden_channels

        # MLP maps graph embeddings -> output classes
        self.mlp = MLP([hidden_channels, out_channels], norm=None)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tensor:
        """
        Forward pass of the hierarchical GCN model with TopKPooling.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``(N, in_channels)``.
        edge_index : torch.Tensor
            Graph connectivity in COO format of shape ``(2, E)``.
        batch : torch.Tensor
            Batch vector assigning each node to a graph, of shape ``(N,)``.

        Returns
        -------
        torch.Tensor
            Graph-level outputs of shape ``(B, out_channels)``,
            where ``B`` is the number of graphs in the batch.
        """
        # Hierarchical message passing + pooling
        for conv, pool in zip(self.convs, self.pools):
            x = conv(x, edge_index).relu()
            x, edge_index, _, batch, _, _ = pool(x, edge_index, batch=batch)

        # Graph-level pooling
        x = global_add_pool(x, batch)

        # Final MLP classifier
        return self.mlp(x)
