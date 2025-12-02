"""
The core of the interaction block described in the NequIP paper.
Includes everything up to the non-linearity.
"""

from typing import cast

import torch
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct, Irreps
from e3nn.o3._linear import Linear
from torch import nn
from torch_geometric.utils import scatter


class InteractionCore(nn.Module):
    """
    Core NequIP-style equivariant message-passing portion.
    Performs everything in the interaction block excludign the
    non-linearity.

    Consists of the following along the main branch:
    1) Linear mixing.
    2) Message passing involving the spherical harmonics of edge
       vectors as well as the length of each edge. Done using a
       custom tensor product with spherical harmonics using weights
       determined via a MLP on the embedded edge lengths.
    3) A second linear mixing.

    Along the self-connection branch, it consists of:
    1) Tensor product with frozen invariant embeddings.

    The results of both branches are added.
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        irreps_node_frozen: Irreps,
        irreps_edge_sph: Irreps,
        edge_dist_n: int,
        radial_hidden: int,
    ) -> None:
        """
        Constructs the InteractionCore module.

        Args:
            irreps_in: Node embeddings before the layer.
            irreps_out: Node embeddings after the module and right
                        before the gate (so make sure to allocate the
                        necessary extra scalars to be used for gating).
            irreps_node_frozen: Irreps of frozen node embeddings used for
                                the skip connection tensor product.
            edge_dist_n: Dimensionality of radial distance embedding used as input
                         to the edge MLP.
            radial_hidden: Hidden-layer width for the edge MLP.
        """
        super().__init__()

        # Convolution for a single edge.
        # Each (mul, l_in) x (singular l_filter Y) -> (mul, l_out).
        # There will be mul paths with in the instruction so it will contain mul different
        # weights, each of which will be from the MLP.

        irreps_mid_list = []
        instructions = []
        weights_count = 0

        for i, (mul, ir_in) in enumerate(irreps_in):
            for j, (_, ir_edge) in enumerate(irreps_edge_sph):
                for ir_out in ir_in * ir_edge:
                    if ir_out in irreps_out:
                        irreps_mid_list.append((mul, ir_out))
                        instructions.append(
                            (i, j, len(irreps_mid_list) - 1, "uvu", True)
                        )
                        weights_count += mul

        irreps_mid = o3.Irreps(irreps_mid_list)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [
            (i, j, p[k], mode, train) for i, j, k, mode, train in instructions
        ]

        self.conv = o3.TensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_edge_sph,
            irreps_out=irreps_mid,
            instructions=instructions,
            internal_weights=False,
            shared_weights=False,
        )
        assert weights_count == self.conv.weight_numel

        # Linear layer before the convolution (channel mixing)
        self.lin1 = Linear(
            irreps_in=irreps_in,
            irreps_out=irreps_in,
        )

        # Linear layer after the convolution (projects to output size)
        self.lin2 = Linear(
            irreps_in=irreps_mid.simplify(),  # Better normalization
            irreps_out=irreps_out,
        )

        # Edge MLP
        self.edge_mlp = FullyConnectedNet(
            [edge_dist_n, radial_hidden, weights_count], nn.SiLU()
        )

        # Self connection
        self.sc = FullyConnectedTensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_node_frozen,
            irreps_out=irreps_out,
        )

    def forward(
        self,
        node_emb_in: torch.Tensor,
        node_emb_frozen: torch.Tensor,
        edge_index: torch.Tensor,
        edge_sph: torch.Tensor,
        edge_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies everything in the interaction layer prior
        to the non-linearity.

        Args:
            node_emb_in (N, irreps_in.dim): Incoming node embeddings.
            node_emb_frozen (N, irreps_node_frozen.dim): Frozen node embeddings.
            edge_index (2, E): Edge index in PyGeomtric format.
            edge_sph (E, irreps_edge_sph.dim): Spherical harmonics for each edge.
            edge_emb (E, edge_dist_n): Distance embedding for each edge.

        Returns:
            Intermediate embeddings to be fed into the gate of the shape (N, irreps_out.dim).
        """
        # Get basic info. Manually taking degree is necessary because of dropout.
        nodes = node_emb_in.shape[0]

        row, col = edge_index
        skip = self.sc(node_emb_in, node_emb_frozen)
        deg = scatter(
            src=torch.ones_like(row), index=col, dim=0, dim_size=nodes, reduce="sum"
        )

        # Self interaction, conv, self interaction
        node_vals = self.lin1(node_emb_in)
        edge_messages = self.conv(
            node_vals[row], edge_sph, weight=self.edge_mlp(edge_emb)
        )

        # Deviation from NequIP: mean rather than sqrt
        message_agg = scatter(
            src=edge_messages, index=col, dim=0, dim_size=nodes, reduce="sum"
        ) / deg.unsqueeze(-1).clamp(min=1.0)

        node_result = self.lin2(message_agg)

        return cast(torch.Tensor, skip + node_result)
