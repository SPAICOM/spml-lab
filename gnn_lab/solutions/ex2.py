# Solutions for Exercise 2: Implement a multi-layer GCN for graph classification with the following requirements:
# - Each GCN layer must use **filter order = 1** (you might use `GCNConv` from `torch_geometric.nn`) and be followed by a **ReLU** activation.
# - Stack multiple GCN layers of your choice (hidden size and number of layers are up to you).
# - After the last GCN layer, apply a **global readout** (`global_add_pool`, `mean`, or `max`) to obtain a graph-level embedding.
# - Feed the graph embedding into a final **MLP classifier** that outputs the class logits.

import torch
from torch import Tensor
from torch_geometric.nn import GCNConv, MLP, global_add_pool
from typing import List


class GCN(torch.nn.Module):
    """
    Multi-layer Graph Convolutional Network (GCN) followed by an MLP for
    graph-level classification.

    This model applies:
    - A stack of GCNConv layers with ReLU activation.
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
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(in_channels, hidden_channels))
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
        Forward pass of the GCN model.

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
        # Node-level feature propagation
        for conv in self.convs:
            x = conv(x, edge_index).relu()

        # Graph-level pooling
        x = global_add_pool(x, batch)

        # Final MLP classifier
        return self.mlp(x)
