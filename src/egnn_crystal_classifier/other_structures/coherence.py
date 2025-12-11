"""
Handles coherence calculations. Used for solid v liquid classification
as well as a measure for disorder.
"""

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm import tqdm

from egnn_crystal_classifier.config import Config


def compute_coherence(
    config: Config,
    neighbors: NDArray[np.integer[Any]],
    embeddings: torch.Tensor,
    calc_device: torch.device,
) -> NDArray[np.float32]:
    """
    Calculates coherence as the average dot product similarity of an
    atom's embedding to that of all nearest neighbors.

    Args:
        config: Configurations for the calculation.
        neighbors (N, num_neighbors + 1): Indices of all num_neighbors + 1 nearest
                                          neighbors including the atom itself.
        embeddings (N, invariant_embedding_size): The calculated embeddings for each atom.
        calc_device: Device to perform coherence calculations on.

    Returns:
        A shape (N,) array of coherence values for each atom.
    """
    neighbors_exclude_torch = torch.from_numpy(neighbors[:, 1:]).long()
    coh_fac = torch.zeros((embeddings.shape[0],))

    for start in tqdm(
        range(0, embeddings.shape[0], config.batch_size), desc="Coherence"
    ):
        neighbor_embeddings = embeddings[
            neighbors_exclude_torch[start : start + config.batch_size]
        ].to(calc_device)

        center_embeddings = (
            embeddings[start : start + config.batch_size].unsqueeze(1).to(calc_device)
        )

        dot_prods = (center_embeddings * neighbor_embeddings).sum(dim=-1)

        sorted_vals, _ = dot_prods.sort(dim=-1, descending=True)
        coh_fac[start : start + config.batch_size] = (
            sorted_vals[:, 3:5].mean(dim=1).cpu()
        )

        # topk_vals, _ = torch.topk(dot_prods, 4, dim=1)
        # coh_fac[start : start + config.batch_size] = topk_vals.mean(dim=1).cpu()
        # coh_fac[start : start + config.batch_size] = dot_prods.mean(dim=1).cpu()

    return coh_fac.numpy().astype(np.float32)
