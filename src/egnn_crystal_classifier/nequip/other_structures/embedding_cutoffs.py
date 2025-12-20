"""
Handles outlier detection to identify if some some atom is unrecognizable.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm

from egnn_crystal_classifier.config import NequIPConfig
from egnn_crystal_classifier.nequip.data_prep.data_reader import (
    all_synthetic_data_to_loader,
    file_to_loader,
)
from egnn_crystal_classifier.nequip.ml_model.model import NequIP


def compute_perfect_embeddings(
    config: NequIPConfig,
    model: NequIP,
    perfect_lattices_path: Path,
    calc_device: torch.device,
) -> npt.NDArray[np.float32]:
    """
    Calculates the ideal embeddings for each crystal type which can be
    used as a baseline.

    Args:
        config: Configuration.
        model: NequIP model.
        perfect_lattices_path: Path to directory containing perfect lattices.
                               Should contain smth like (bcc.gz, cd.gz, ...)
        calc_device: Device to perform calculations on.

    Returns:
        (num_crystals, sz) numpy array of ideal embeddings for each crystal
        type with index determined on label type.
    """
    model = model.to(calc_device).eval()

    perfect_embeddings = np.zeros(
        (len(config.crystals), config.invariant_embedding_size), dtype=np.float32
    )

    for structure in config.crystals:
        loader = file_to_loader(config, perfect_lattices_path / f"{structure}.gz")

        # Run inference to calculate the average embedding
        embeddings_average = np.zeros(config.invariant_embedding_size, dtype=np.float32)

        for data in loader:
            with torch.inference_mode():
                embedding = model(data.to(calc_device))[1].cpu().numpy()
            embeddings_average += embedding.sum(axis=0)

        embeddings_average /= len(loader.dataset)
        perfect_embeddings[config.label_map[structure]] = embeddings_average

    return perfect_embeddings


def compute_cutoff(
    config: NequIPConfig,
    model: NequIP,
    perfect_embeddings: npt.NDArray[np.float32],
    synthetic_data_path: Path,
    calc_device: torch.device,
) -> npt.NDArray[np.float32]:
    """
    Computes the cutoff (e.g. 99% percentile distance to perfect embedding)
    using synthetic data. This cutoff is used to determine whether some atom
    has "unknown structure" rather than the given classified structure.

    Synthetic data should lie in as subdirectories in an outer directory. Each
    subdirectory should be named the structure of the data it contains.

    Args:
        config: Configuration.
        model: NequIP model.
        perfect_embeddings: Reference embeddings based on perfect lattices.
        synthetic_data_path: Path of outer directory containing all structure
                             directories which contain synthetic data.
        calc_device: Device to perform calculations on.

    Returns:
        (num_crystals,) numpy array storing the cutoffs for each crystal
        type with index determined by label map.
    """
    model = model.to(calc_device).eval()

    # For each structure, load all distances to perfect embedding
    loader = all_synthetic_data_to_loader(config, synthetic_data_path)
    dist_lists: list[list[float]] = [[] for _ in range(len(perfect_embeddings))]

    for data in tqdm(loader, desc="Calculating Embeddings"):
        with torch.inference_mode():
            embeddings = model(data.to(calc_device))[1].cpu().numpy()
        for embedding, structure in zip(embeddings, data.y):
            dist_lists[structure].append(
                (embedding * perfect_embeddings[structure]).sum(axis=-1)
            )

    # Compute percentile
    delta_cutoffs = np.zeros(len(perfect_embeddings), dtype=np.float32)

    for i, dist_list in enumerate(dist_lists):
        assert len(dist_list) > 1, "Each structure should have nonempty dist list"
        delta_cutoffs[i] = np.percentile(np.array(dist_list), config.cutoff)

    return delta_cutoffs
