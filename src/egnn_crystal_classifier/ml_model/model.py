import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter


class ConvLayer(MessagePassing):
    def __init__(
        self,
        num_buckets: int,
        input_size: int,
        hidden_and_output: int,
        dropout_prob: float,
    ) -> None:
        super().__init__(aggr="mean")

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_size + input_size + 1 + num_buckets, hidden_and_output),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_and_output, 1),
            nn.Identity(),
        )

        self.gate = nn.Linear(hidden_and_output, 1)

        self.node_mlp = nn.Sequential(
            nn.Linear(input_size + hidden_and_output, hidden_and_output),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_and_output, hidden_and_output),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
        )

    def forward(
        self,
        h: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_angle_hist: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # Prepare
        row, col = edge_index
        deg = scatter(torch.ones_like(row), row, dim=0, dim_size=x.size(0))
        deg = deg.to(x.dtype).clamp_min(1.0)

        edge_vec = x[col] - x[row]

        # Extract distance and angle histograms across every edge and combine to form desired vector for each edge
        edge_dists = edge_vec.norm(p=2, dim=1, keepdim=True)

        edge_info = torch.cat([h[row], h[col], edge_dists, edge_angle_hist], dim=1)
        edge_out = self.edge_mlp(edge_info)

        # Weight each edge
        gate = torch.sigmoid(self.gate(edge_out))
        edge_out = edge_out * gate

        # Propagate (for edge i,j it'll set look at m[i,j])
        node_accum = self.propagate(edge_index, m=edge_out)

        # Run MLP on all nodes at once
        node_out = self.node_mlp(torch.cat([h, node_accum], dim=1))

        # Update positional embeddings
        weight = self.coord_mlp(edge_out)
        pos_message = edge_vec * weight

        delta = scatter(pos_message, col, dim=0, dim_size=x.size(0))
        x_new = x + delta / deg[:, None]

        return node_out, x_new

    def message(self, m: Tensor) -> Tensor:
        return m


class EGNN(nn.Module):
    def __init__(
        self,
        num_buckets: int,
        hidden: int,
        num_reg_layers: int,
        num_classes: int,
        dropout_prob: float,
    ) -> None:
        super().__init__()
        self.dropout_prob = dropout_prob

        self.firstLayer = ConvLayer(num_buckets, 0, hidden, self.dropout_prob)

        self.otherLayers = nn.ModuleList(
            ConvLayer(num_buckets, hidden, hidden, self.dropout_prob)
            for _ in range(num_reg_layers)
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(num_reg_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden)

        # Don't add any activation at the end since we use CrossEntropyLoss
        self.classifier_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, data: Data) -> tuple[Tensor, Tensor]:
        pos_norm, edge_index, edge_angle_hist, center_mask = (
            data.pos_norm,
            data.edge_index,
            data.edge_angle_hist,
            data.center_mask,
        )

        # Note that no initial node features will be provided
        h = torch.empty(
            (pos_norm.shape[0], 0), dtype=pos_norm.dtype, device=pos_norm.device
        )
        x = pos_norm

        h, x = self.firstLayer(h, x, edge_index, edge_angle_hist)

        # Run rest of layers
        for layer, norm in zip(self.otherLayers, self.norms):
            h_res = h

            # Run layer and prenormalize
            h, x = layer(norm(h), x, edge_index, edge_angle_hist)

            # Residual connection
            h = h + h_res

        # Run it through final MLP to get classification (assuming center_mask contains one center per graph)
        embeddings = self.final_norm(h[center_mask])
        return self.classifier_mlp(embeddings), embeddings
