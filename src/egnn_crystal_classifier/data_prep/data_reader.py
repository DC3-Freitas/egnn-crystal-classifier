"""
Helpers for interacting with synthetic data and preparing the necessary dataloaders
to be directly used by the model.
"""

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from ovito.io import import_file  # pylint: disable=no-name-in-module

from egnn_crystal_classifier.config import Config
from egnn_crystal_classifier.data_prep.data_handler import CrystalDataset, FastLoader
from egnn_crystal_classifier.data_prep.graph_construction import construct_graph_lists
from egnn_crystal_classifier.utils import set_seed


def get_data_from_files(
    config: Config, synthetic_data_path: Path
) -> tuple[npt.NDArray[np.float32], list[str]]:
    """
    Extracts data from files and returns them in raw graph form.

    Args:
        config: Configuration for things like number of nearest neighbors.
        synthetic_data_path: Path where synthetic data lies.

    Returns:
        x_data: Shape (N, num neighbors + 1, 3) numpy array of graphs denoting
                the raw (non-normalized) positions of each atom in the graph.
        y_strs_list: The ground truth structure (as a string) corresponding to each graph.
    """
    x_data_list: list[npt.NDArray[np.float32]] = []
    y_strs_list = []

    for structure in os.listdir(synthetic_data_path):
        for f in os.listdir(synthetic_data_path / structure):
            pipeline = import_file(synthetic_data_path / structure / f)
            lattice = pipeline.compute()

            all_positions = np.copy(lattice.particles.positions).astype(np.float32)
            _, graph = construct_graph_lists(
                all_positions,
                config.num_neighbors,
                np.array(lattice.cell, dtype=np.float32),
            )

            x_data_list.append(graph)
            y_strs_list.extend([structure] * len(all_positions))

    return np.concat(x_data_list), y_strs_list


def calculate_class_weights(config: Config, y_strs_list: list[str]) -> torch.Tensor:
    """
    Calculates weights so that each class recieves equal weighting.
    This is to prevent bias from an imbalanced dataset.

    Args:
        config: Configuration for label map.
        y_strs_list: List of all classes in string form (e.g. ["fcc", "bcc", etc.]).

    Returns:
        Tensor containing weight for each class normalized so that the sum is 1.
    """
    y_ints = np.array([config.label_map[y] for y in y_strs_list])
    class_counts = np.bincount(y_ints)
    inv_freq = 1.0 / class_counts
    return torch.tensor(inv_freq / inv_freq.mean()).float()


def create_loaders(
    config: Config, synthetic_data_path: Path
) -> tuple[FastLoader, FastLoader, FastLoader, torch.Tensor]:
    """
    Creates data loaders from synthetic data according to the specified configs.

    Args:
        config: Configuration for data preparation pipeline.
        synthetic_data_path: Path for the directory containing the synthetic data.
                             Should have folders for each structure (e.g. "bcc").

    Returns:
        train_loader: Data loader for loading training data.
        train_eval_loader: Data loader for loading a subset of training data
                           used for faster train accuracy evaluations (under eval mode).
        test_loader: Data loader for test data.
        weights: Weights for each class to address data imbalance. Should sum to 1.
    """
    set_seed()

    # Prep data
    x_data, y_strs_list = get_data_from_files(config, synthetic_data_path)
    num_data_points = len(x_data)

    train_section = int(config.train_split_frac * num_data_points)
    train_eval_size = int(config.train_eval_sample_frac * train_section)

    use_indices = np.random.permutation(num_data_points)

    # Indices for dataset
    train_indices = use_indices[:train_section]
    train_eval_indices = np.random.choice(
        train_indices, size=train_eval_size, replace=False
    )
    test_indices = use_indices[train_section:num_data_points]

    # Datasets
    train_dataset = CrystalDataset(
        x_data[train_indices],
        [y_strs_list[i] for i in train_indices],
        config.label_map,
    )
    train_eval_dataset = CrystalDataset(
        x_data[train_eval_indices],
        [y_strs_list[i] for i in train_eval_indices],
        config.label_map,
    )
    test_dataset = CrystalDataset(
        x_data[test_indices],
        [y_strs_list[i] for i in test_indices],
        config.label_map,
    )

    # Dataloaders
    train_loader = FastLoader(train_dataset, config.batch_size, shuffle=True)
    train_eval_loader = FastLoader(train_eval_dataset, config.batch_size, shuffle=False)
    test_loader = FastLoader(test_dataset, config.batch_size, shuffle=False)
    weights = calculate_class_weights(config, y_strs_list)

    return (
        train_loader,
        train_eval_loader,
        test_loader,
        weights,
    )
