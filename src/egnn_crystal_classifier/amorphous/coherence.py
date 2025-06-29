from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray


def compute_coherence(
    neighbors_np: NDArray[np.number[Any]],
    embeddings: torch.Tensor,
    batch_size: int,
    calc_device: torch.device,
) -> NDArray[np.floating[Any]]:
    # important: neighbors for each atom is exclusive of the atom itself
    neighbors_torch = torch.from_numpy(neighbors_np).long()
    coh_fac = torch.zeros((embeddings.shape[0],))

    for start in range(0, embeddings.shape[0], batch_size):
        neighbor_embeddings = embeddings[
            neighbors_torch[start : start + batch_size]
        ].to(calc_device)
        center_embeddings = (
            embeddings[start : start + batch_size].unsqueeze(1).to(calc_device)
        )
        dot_prods = (center_embeddings * neighbor_embeddings).sum(dim=-1)
        coh_fac[start : start + batch_size] = dot_prods.mean(dim=1).cpu()

    return coh_fac.numpy()


def get_amorphous_mask(
    neighbors_np: NDArray[np.number[Any]],
    embeddings: torch.Tensor,
    batch_size: int,
    calc_device: torch.device,
    cutoff: int | None,
) -> NDArray[np.bool_]:
    """
    Determines which atoms are amorphous based on coherence.

    Args:
        positions (np.ndarray): Array of atomic positions in the crystal structure.
        embeddings (np.ndarray): Array of ML embeddings for each atom.
        num_neighbors (int): Number of nearest neighbors to consider for coherence.
        cutoff (float): The distance threshold for coherence.
            If cutoff is -1, then we attempt to compute it automatically
            by taking the histogram and finding value that minimizes inter-class variance.

    Returns:
        np.ndarray: Boolean array indicating whether each atom is amorphous (True) or not (False).
    """

    embedding_similarity = compute_coherence(
        neighbors_np, embeddings, batch_size, calc_device
    )

    if cutoff is None:
        # automatically determine the cutoff using Otsu's method
        hist, bin_edges = np.histogram(embedding_similarity, bins=100)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

        # class probabilities over all possible thresholds
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]

        # class means over all possible thresholds
        mean1 = np.cumsum(hist * bin_centers) / weight1
        mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

        # calculate interclass variance
        variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

        idx = np.argmax(variance12)
        cutoff = bin_centers[idx]
        print(f"Automatically determined cutoff: {cutoff}")

    return embedding_similarity < cutoff
