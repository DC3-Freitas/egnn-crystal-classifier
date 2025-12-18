"""
The entire NequIP-style model for structural classification and
local structure descriptors.
"""

import torch
import torch.nn as nn
from e3nn.nn import Gate
from e3nn.o3 import FullyConnectedTensorProduct, Irrep, Irreps, spherical_harmonics
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from egnn_crystal_classifier.config import Config
from egnn_crystal_classifier.ml_model.edge_embed import bessel
from egnn_crystal_classifier.ml_model.interaction import InteractionCore


class NequIP(nn.Module):
    """
    Equivariant GNN for local structural classification and
    invariant representation based on the NequIP architecture.

    Follows the structure:
    1) Embedding matrix lookup for initial and frozen embeddings.
    2) Many equivariant interaction layers.
    3) Combining embeddings to form graph-level descriptors (weighted tensor
       product between center node embeddings and mean-pooled graph
       embeddings).
    4) L2-normalization of the invariant descriptor for each graph. This
       serves as our graph-level invariant embedding.
    5) A final linear classification head.

    The implementaiton assumes all irreps have even parity.
    """

    def __init__(
        self,
        config: Config,
    ) -> None:
        """
        Initializes all parts of the model.

        Args:
            config: Includes all information necessary to construct the model.
                    See Config class for more details.
        """
        super().__init__()

        # Requires even irreps
        for _, ir in Irreps(config.irreps_hidden) + Irreps(config.irreps_edge_sph):
            assert ir.p == 1, "we assume that all irreps are even"

        self.config = config

        # Embedding
        self.embedding_frozen = nn.Embedding(
            config.num_species, config.node_embedding_frozen
        )
        self.embedding_init = nn.Embedding(
            config.num_species, config.node_embedding_init
        )

        # Initialize layers
        self.layers_core = nn.ModuleList()
        self.layers_gate = nn.ModuleList()

        for i in range(config.num_convs):
            # The following is simplified from the original NequIP and denoiser implementations
            # and is based on the assumption that all Irreps are even.
            scalars = 0
            gate_count = 0
            gated: list[tuple[int, Irrep]] = []

            for mul, ir in Irreps(config.irreps_hidden):
                if ir.l == 0:
                    scalars += mul
                else:
                    gate_count += mul
                    gated.append((mul, ir))

            gate = Gate(
                irreps_scalars=f"{scalars}x0e",
                act_scalars=[nn.SiLU()],
                irreps_gates=f"{gate_count}x0e",
                act_gates=[nn.Sigmoid()],
                irreps_gated=Irreps(gated),
            )
            assert gate.irreps_out == Irreps(
                config.irreps_hidden
            ), "output of gate should be hidden"

            self.layers_core.append(
                InteractionCore(
                    irreps_in=(
                        Irreps(f"{config.node_embedding_init}x0e")
                        if i == 0
                        else Irreps(config.irreps_hidden)
                    ),
                    irreps_out=gate.irreps_in,
                    irreps_node_frozen=Irreps(f"{config.node_embedding_frozen}x0e"),
                    irreps_edge_sph=Irreps(config.irreps_edge_sph),
                    edge_dist_n=config.edge_dist_n,
                    radial_hidden=config.radial_hidden,
                )
            )
            self.layers_gate.append(gate)

        assert len(self.layers_core) == len(self.layers_gate) == config.num_convs

        self.to_inv_embedding = FullyConnectedTensorProduct(
            irreps_in1=Irreps(config.irreps_hidden),
            irreps_in2=Irreps(config.irreps_hidden),
            irreps_out=f"{config.invariant_embedding_size}x0e",
        )
        self.pre_head_act = nn.SiLU()
        self.head = nn.Linear(
            in_features=config.invariant_embedding_size,
            out_features=config.output_classes,
        )

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the entire model pipeline for structural classification.

        Args:
            data: PyGeometric data object containing batched graph information.
                  Must be batched and consist of the following fields:
                  x, pos, edge_index, center_mask, batch, y
                  See construct_batched_graph for more details.

        Returns:
            logits (B, output_classes): Logits for each graph for classificaiton.
            inv_embeddings (B, invariant_embedding_size): Normalized (L2) invariant embedding
                                                          for each graph.
        """
        # Node and edge dropout
        if self.training:
            non_center = ~data.center_mask
            drop_node_mask = (
                torch.rand(non_center.shape[-1], device=data.x.device)
                < self.config.node_dropout
            ) & non_center
            data = data.subgraph(~drop_node_mask)

            drop_edge_mask = (
                torch.rand(data.edge_index.shape[-1], device=data.x.device)
                < self.config.edge_dropout
            )
            data.edge_index = data.edge_index[:, ~drop_edge_mask]

        # We let data.x just be the species
        node_emb = self.embedding_init(data.x)
        frozen_emb = self.embedding_frozen(data.x)

        # Prepare edge info
        row, col = data.edge_index

        # In message passing, we pass along j --> i yet vectors should be relative
        # to central atom i so we need to use vector corresponding to i --> j which
        # is why we invert by subtracting the target from the source.
        edge_vecs = data.pos[row] - data.pos[col]
        edge_dists = edge_vecs.norm(dim=-1, keepdim=True)

        edge_sph = spherical_harmonics(
            l=Irreps(self.config.irreps_edge_sph),
            x=edge_vecs,
            normalize=True,
            normalization="component",
        )
        edge_emb = bessel(edge_dists, self.config.edge_dist_n, self.config.r_c)

        # Go through layers
        for core, gate in zip(self.layers_core, self.layers_gate):
            node_emb = core(node_emb, frozen_emb, data.edge_index, edge_sph, edge_emb)
            node_emb = gate(node_emb)

        # Obtain final invariant embeddings
        final_eq_embeddings = node_emb[data.center_mask]
        average_eq_embeddings_per_graph = scatter(
            node_emb, data.batch, dim=0, reduce="mean"
        )

        inv_embeddings = self.to_inv_embedding(
            final_eq_embeddings, average_eq_embeddings_per_graph
        )

        # Normalize invariant embeddings so that it lives on hypersphere
        inv_embeddings = inv_embeddings / (
            inv_embeddings.norm(dim=-1, keepdim=True) + 1e-8
        )

        return (
            self.head(self.pre_head_act(inv_embeddings)),
            inv_embeddings,
        )
