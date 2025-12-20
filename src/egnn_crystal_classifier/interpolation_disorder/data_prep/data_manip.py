"""
Helpers related to interacting with synthetic data for the disorder model
and preparing necessary data loaders.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from e3nn.o3 import Irreps, spherical_harmonics
from numpy.typing import NDArray
from ovito.io import import_file  # pylint: disable=no-name-in-module
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

from egnn_crystal_classifier.config import DisorderModelConfig
from egnn_crystal_classifier.utils.neighbors_and_graph import construct_graph_lists
from egnn_crystal_classifier.utils.seed import set_seed


def calculate_sph_inputs(
    config: DisorderModelConfig,
    pos_individual: NDArray[np.float32],
    cell: NDArray[np.float32] | None,
    calc_device: torch.device,
) -> tuple[torch.Tensor, NDArray[np.integer[Any]]]:
    """
    Calculates inputs for disorder model. For each atom, its initial
    embedding are the average of spherical harmonics of unit vectors to
    nearest neighbors.

    Args:
        config: Configuration for disorder model details
        pos_individual (B, 3): Coordinates of each atom.
        cell (3, 4) | None: Ovito simulation cell (we ignore the pbc flags and assume
                            pbc is applied everywhere).
        calc_device: Device we use to perform calculations on.
    Returns:
        embeddings (B, sz): Average of sphericla harmonics of unit vectors to
                            nearest neighbors for each atom (takes PBC into account
                            if available).
        neighbors (B, num_neighbors + 1): Indices of nearest neighbors with the first
                                          element being the atom itself.

        Note: you can get model inputs by doing embeddings[neighbors].
    """
    neighbors, pos_graphs = construct_graph_lists(
        pos_individual, config.num_neighbors, cell
    )

    # Get raw embeddings (compute in calc_device and put result on cpu)
    embeddings = torch.zeros((len(pos_graphs), Irreps(config.irreps_in).dim))

    for i in tqdm(range(0, len(pos_graphs), config.batch_size), desc="Calculating SPH"):
        batch = torch.from_numpy(pos_graphs[i : i + config.batch_size]).to(calc_device)

        spharm = spherical_harmonics(
            Irreps(config.irreps_in),
            batch[:, 1:, :] - batch[:, 0:1, :],
            normalize=True,
            normalization="component",
        )

        embeddings[i : i + config.batch_size] = spharm.mean(dim=1).cpu()

    # Aggregate embeddings based on neighbors to form our input
    return embeddings, neighbors


def get_disorder_loaders_for_training(
    config: DisorderModelConfig,
    synthetic_data_path: Path,
    calc_device: torch.device,
) -> tuple[
    DataLoader[tuple[torch.Tensor, ...]],
    DataLoader[tuple[torch.Tensor, ...]],
]:
    """
    Creates data loaders from synthetic data according to configs.

    The data should lie in as subdirectories in an outer directory. Each
    subdirectory should be named the structure of the data it contains.

    Warning: The current code assumes that generation does not support PBC.
             If it does in the future, edit this implementation to reflect it.

    Args:
        config: Configuration for data preparation pipeline.
        synthetic_data_path: Path for the outer directory containing the synthetic data.
        calc_device: Device to perform calculations on.

    Returns:
        train_loader: Data loader for training data.
        test_loader: Data loader for test data.
    """
    set_seed()

    all_data_list: list[torch.Tensor] = []
    all_label_list: list[float] = []

    for structure in os.listdir(synthetic_data_path):
        for f in os.listdir(synthetic_data_path / structure):
            pipeline = import_file(synthetic_data_path / structure / f)

            source = pipeline.source
            assert source is not None, "Pipeline source should exist."

            for frame in range(source.num_frames):
                lattice = pipeline.compute(frame)
                pos_individual = np.array(lattice.particles.positions, dtype=np.float32)

                # Note: current generation does not support PBC
                embeddings, neighbors = calculate_sph_inputs(
                    config, pos_individual, None, calc_device
                )
                train_inputs = embeddings[neighbors]

                all_data_list.append(train_inputs)
                all_label_list.extend(
                    frame / source.num_frames for _ in range(len(train_inputs))
                )

    # Convert to data loader
    x_data, y_data = torch.concatenate(all_data_list), torch.tensor(all_label_list)
    dataset = TensorDataset(x_data, y_data)

    use_indices = np.random.permutation(len(dataset)).tolist()
    train_section = int(config.train_split_frac * len(dataset))

    train_indices = use_indices[:train_section]
    test_indices = use_indices[train_section:]

    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=config.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices), batch_size=config.batch_size, shuffle=False
    )
    return train_loader, test_loader
