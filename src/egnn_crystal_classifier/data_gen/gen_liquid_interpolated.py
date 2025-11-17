import os
import numpy as np
from scipy.spatial import cKDTree
from typing import List, Tuple, Optional

from egnn_crystal_classifier.constants import NN_COUNT

def load_xyz(file_path: str) -> np.ndarray:
    """Load XYZ file and return atomic positions as a numpy array."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    frames = []
    cur_line = 0
    while cur_line < len(lines):
        num_atoms = int(lines[cur_line].strip())
        cur_line += 2  # Skip the comment line
        positions = []
        for _ in range(num_atoms):
            parts = lines[cur_line].strip().split()
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            cur_line += 1
        frames.append(np.array(positions))
    return frames

def smoothed_label(value: int, tot: int) -> np.ndarray:
    """
    Generate a smoothed one-hot label. Since here
    our labels are temperature values, we use a Gaussian smoothing.
    """
    label = np.zeros(tot, dtype=np.float32)
    sigma = tot / 20.0
    for i in range(tot):
        label[i] = np.exp(-0.5 * ((i - value) / sigma) ** 2)
    label /= label.sum()
    return label

def gen() -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Generate interpolated liquid data from XYZ files.

    Returns:
        positions: np.ndarray of shape (num_samples, num_atoms, 3)
        labels: List of smoothed labels for each sample
    """
    base_dir = os.path.join(os.path.dirname(__file__), 'interpolated_lattices')
    all_positions = []
    all_labels = []
    for structure_type in os.listdir(base_dir):
        structure_dir = os.path.join(base_dir, structure_type)
        if not os.path.isdir(structure_dir):
            continue
        for run in os.listdir(structure_dir):
            if not run.endswith('.xyz'):
                continue
            file_path = os.path.join(structure_dir, run)
            frames = load_xyz(file_path)
            num_frames = len(frames)
            for i, frame in enumerate(frames):
                tree = cKDTree(frame)
                neighbors = tree.query(frame, k=NN_COUNT + 1)[1]
                for neigh_indices in neighbors:
                    neighbor_positions = frame[neigh_indices]
                    all_positions.append(neighbor_positions)
                    temp_label = (i / num_frames)
                    # all_labels.append(smoothed_label(temp_label, 10))
                    all_labels.append(temp_label)

    positions = np.array(all_positions)
    labels = all_labels
    return positions, labels
