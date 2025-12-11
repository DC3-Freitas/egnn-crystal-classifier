"""
TODO
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm

from egnn_crystal_classifier.config import Config
from egnn_crystal_classifier.data_prep.data_reader import (
    all_synthetic_data_to_loader,
    file_to_loader,
)
from egnn_crystal_classifier.ml_model.model import NequIP


def compute_perfect_embeddings(
    config: Config,
    model: NequIP,
    perfect_lattices_path: Path,
    calc_device: torch.device,
) -> npt.NDArray[np.float32]:
    """
    TODO
    """
    model.eval()
    model = model.to(calc_device)

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
    config: Config,
    model: NequIP,
    perfect_embeddings: npt.NDArray[np.float32],
    synthetic_data_path: Path,
    calc_device: torch.device,
) -> npt.NDArray[np.float32]:
    """
    TODO
    """
    model.eval()
    model = model.to(calc_device)

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

    # Compute 99th percentile
    delta_cutoffs = np.zeros(len(perfect_embeddings), dtype=np.float32)

    for i, dist_list in enumerate(dist_lists):
        assert len(dist_list) > 1, "Each structure should have nonempty dist list"
        delta_cutoffs[i] = np.percentile(np.array(dist_list), 1)

    return delta_cutoffs
