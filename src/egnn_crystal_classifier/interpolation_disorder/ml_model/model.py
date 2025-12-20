"""
The model for predicting disorder given initial embeddings of central
atom and nearest neighbors.
"""

from typing import cast

import torch
import torch.nn as nn
from e3nn.nn import Gate
from e3nn.o3 import FullyConnectedTensorProduct, Irrep, Irreps

from egnn_crystal_classifier.config import DisorderModelConfig


class DisorderModel(nn.Module):
    """
    Equivariant model to predict the interpolation parameter (which
    could be viewed as the disorder) of some local structure.

    For each atom, initial embeddings are average spherical harmonics
    of unit vectors to all nearest neighbors.

    Note that internal operations should be E3 equivariant/invariant and (2) should
    be permutation equivariant/invariant with the output being invariant to all.

    The architecture follows the structure:
    1) Self tensor product to project onto larger irreps.
    2) Tensor product between every atom with the central one (inclusive of the
       central one) with the output being l=0 (even).
    3) Feed through MLP, mean pool, and then feed pooled result through MLP.

    The model accepts the following input (we mostly use this for training):
    1) Initial embedding of central atom
    2) Initial embeddings of all nearest neighbors

    A speedup is that (1) is unnecessarily computed many times for an atom.
    For inference, we can speed it up by doing:
    1) Compute self tensor products once across all atoms to extract features.
    2) Aggregate the features for each central atom and pass that in.
    """

    def __init__(
        self,
        config: DisorderModelConfig,
    ) -> None:
        """
        Initializes all parts of the model.

        Args:
            config: Includes all information necessary to construct the model.
                    See DisorderModelConfig class for more details.
        """
        super().__init__()

        scalars = 0
        gate_count = 0
        gated: list[tuple[int, Irrep]] = []

        for mul, ir in Irreps(config.irreps_mid):
            if ir.l == 0:
                scalars += mul
            else:
                gate_count += mul
                gated.append((mul, ir))

        self.initial_gate = Gate(
            irreps_scalars=f"{scalars}x0e" if scalars else "",
            act_scalars=[nn.SiLU()] if scalars else [],
            irreps_gates=f"{gate_count}x0e",
            act_gates=[nn.SiLU()],  # SiLU better than sigmoid in this case
            irreps_gated=Irreps(gated),
        )
        self.initial_transformation = FullyConnectedTensorProduct(
            irreps_in1=Irreps(config.irreps_in),
            irreps_in2=Irreps(config.irreps_in),
            irreps_out=Irreps(self.initial_gate.irreps_in),
        )
        self.interact_with_center = FullyConnectedTensorProduct(
            irreps_in1=Irreps(config.irreps_mid),
            irreps_in2=Irreps(config.irreps_mid),
            irreps_out=f"{config.hidden_size}x0e",
        )
        self.mlp_mid = nn.Sequential(
            nn.Linear(config.hidden_size, config.mlp_hidden),
            nn.SiLU(),
            nn.Linear(config.mlp_hidden, config.hidden_size),
            nn.SiLU(),
        )
        self.mlp_final = nn.Sequential(
            nn.Linear(config.hidden_size, config.mlp_hidden),
            nn.SiLU(),
            nn.Linear(config.mlp_hidden, 1),
        )

    def inference_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs initial tensor product and gate to transform spherical harmonic
        average into better features.

        Purpose for this is to first transform all initial embeddings and then
        aggregate based on neighbors and predict rather than recalculating each time
        when doing a regular forward pass. This applies to inference only.

        Args:
            x (B, sz_init): Initial embeddings of all B atoms.

        Returns:
            (B, sz_after) tensor of new embeddings for all B atoms.
        """
        # Although the same code, this feature extraction is meant for (N, sz)
        encoded = self.initial_gate(self.initial_transformation(x, x))
        return cast(torch.Tensor, encoded)

    def predict_from_neigh(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Takes in embeddings (post-encoding) of local structures and predicts
        disorder.

        Args:
            feats (B, N, sz): Batched inputs with each input consisting of
                              post-encoding embeddings for all N (num_neighbors + 1) atoms.

        Returns:
            Disorder (B, 1) for each atom based on its local structure.
        """
        # Tensor product with center: (B, N, sz) x (B, 1, sz)
        hidden_states = self.mlp_mid(self.interact_with_center(feats, feats[:, 0:1, :]))

        # Mean of (B, N, sz) across middle dimension to pool representations
        pooled = hidden_states.mean(dim=1)
        disorder = self.mlp_final(pooled)
        return cast(torch.Tensor, disorder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs entire model pipeline to predict disorder.

        Args:
            x (B, N, sz): Batched inputs with each input consisting of
                          initial embeddings for all N (num_neighbors + 1) atoms.
        Returns:
            Disorder (B, 1) for each atom based on its local structure.
        """
        # Although the same code, this feature extraction takes in (B, N, sz)
        feats = self.initial_gate(self.initial_transformation(x, x))
        return self.predict_from_neigh(feats)
