"""
Module for generating defective crystal structures by introducing vacancies and interstitials.
Attack plan:
- Load in all perfect/temperature preturbed synthetic structures
- Randomly remove atoms to create vacancies; all of its neighbors will be given the "vacancy" label
  (since the vacancy itself is not an atom, it cannot be given a label)
- Randomly add atoms to create interstitials; the added atom will be given the "interstitial" label
"""

import numpy as np
from egnn_crystal_classifier.constants import *
from ovito.io import import_file
from scipy.spatial import cKDTree
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from egnn_crystal_classifier.data_prep.graph_construction import construct_graph_lists

VACANCY_CUTOFF = 1.3
MIN_TEMP = 0
MAX_TEMP = 0.1
NUM_SAMPLES_PER_REFERENCE = 20 # number of times to generate defected structures per reference structure

def generate_single_defect_structure(
        positions: np.ndarray,
        num_vacancies: int = 5,
        num_interstitials: int = 5,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate defective crystal structures by introducing vacancies and interstitials.

    Args:
        positions (np.ndarray): Nx3 array of atomic positions in the perfect structure.
        num_vacancies (int): Number of vacancies to introduce.
        num_interstitials (int): Number of interstitials to introduce.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - defective_positions (np.ndarray): Mx3 array of atomic positions in the defective structure.
            - labels (np.ndarray): M-length array of integer labels for each atom in the defective structure.
                0: perfect, 1: vacancy neighbor, 2: interstitial
            - label_map (dict): Mapping from label names to integers.
    """
    # Create a copy of the original positions to modify
    defective_positions = positions.copy()
    num_atoms = len(defective_positions)

    # Initialize labels: 0 for perfect atoms
    labels = np.zeros(num_atoms, dtype=int)

    # Create a KDTree for efficient neighbor searching
    tree = cKDTree(defective_positions)
    print("Initial number of atoms:", num_atoms)
    print("average distance:", np.mean(tree.query(defective_positions, k=2)[0][:, 1]))

    # Introduce vacancies
    if num_vacancies > 0:
        vacancy_indices = np.random.choice(num_atoms, size=num_vacancies, replace=False)
        for vac_idx in vacancy_indices:
            # Find neighbors within the vacancy cutoff distance
            neighbors = tree.query_ball_point(defective_positions[vac_idx], r=VACANCY_CUTOFF)
            for neighbor in neighbors:
                if neighbor != vac_idx:
                    labels[neighbor] = 1  # Mark as vacancy neighbor
        # Remove the vacancy atoms from the structure
        defective_positions = np.delete(defective_positions, vacancy_indices, axis=0)
        labels = np.delete(labels, vacancy_indices, axis=0)
        # Rebuild the KDTree after removing atoms
        tree = cKDTree(defective_positions)

    # Introduce interstitials
    interstitial_positions = []
    for _ in range(num_interstitials):
        # Randomly select a position within the bounding box of the original structure
        min_bounds = np.min(positions, axis=0)
        max_bounds = np.max(positions, axis=0)
        interstitial_pos = np.random.uniform(min_bounds, max_bounds)
        interstitial_positions.append(interstitial_pos)
        # Add the interstitial atom to the structure
        defective_positions = np.vstack([defective_positions, interstitial_pos])
        labels = np.append(labels, 2)  # Mark as interstitial

    # Final label map
    label_map = {
        "perfect": 0,
        "vacancy_neighbor": 1,
        "interstitial": 2,
    }

    # visualize
    visualize_defects(defective_positions, labels, title="Defective Structure Example")

    return defective_positions, labels, label_map

def visualize_defects(positions, labels, title="Defective Structure"):
    """
    Visualizes the defective structure with different colors for perfect atoms,
    vacancy neighbors, and interstitials.

    Args:
        positions (np.ndarray): Nx3 array of atomic positions.
        labels (np.ndarray): N-length array of integer labels for each atom.
            0: perfect, 1: vacancy neighbor, 2: interstitial
        title (str): Title of the plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define colors for different labels
    colors = {0: 'blue', 1: 'orange', 2: 'red'}
    sizes = {0: 5, 1: 10, 2: 20}
    label_names = {0: 'Perfect', 1: 'Vacancy Neighbor', 2: 'Interstitial'}

    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(positions[mask, 0], positions[mask, 1], positions[mask, 2],
                   c=colors[label], label=label_names[label], s=sizes[label])

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
    return fig, ax

def generate_defects(
        num_vacancies: int = 5,
        num_interstitials: int = 5,
    ) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    """
    Generate multiple defective crystal structures by introducing vacancies and interstitials.
    Args:
        num_vacancies (int): Number of vacancies to introduce in each structure.
        num_interstitials (int): Number of interstitials to introduce in each structure.
    Returns:
        tuple[list[np.ndarray], list[np.ndarray], dict]: A tuple containing:
            - all_defective_positions (list[np.ndarray]): List of Mx3 arrays of atomic positions in defective structures.
            - all_labels (list[np.ndarray]): List of M-length arrays of integer labels for each atom in defective structures.
            - label_map (dict): Mapping from label names to integers.
    """
    all_defective_positions = []
    all_labels = []
    label_map = {
        "perfect": 0,
        "vacancy_neighbor": 1,
        "interstitial": 2,
    }
    for structure in os.listdir(SYNTH_DATA_PATH):
        print("Processing structure:", structure)
        for f in tqdm(os.listdir(os.path.join(SYNTH_DATA_PATH, structure))):
            temp = float(f[:-3])
            if temp > MAX_TEMP:
                continue
            if temp < MIN_TEMP:
                continue
            pipeline = import_file(os.path.join(SYNTH_DATA_PATH, structure, f))
            lattice = pipeline.compute()
            all_positions = np.copy(lattice.particles.positions)
            for _ in range(NUM_SAMPLES_PER_REFERENCE):
                defective_positions, labels, _ = generate_single_defect_structure(
                    all_positions,
                    num_vacancies=num_vacancies,
                    num_interstitials=num_interstitials,
                )
                all_defective_positions.append(defective_positions)
                all_labels.append(labels)
    print("Generating neighbor graphs ...")
    pos_graphs = []
    # for positions, labels in zip(all_defective_positions, all_labels):
    #     tree = cKDTree(positions)
    #     neighbors = tree.query(positions, k=13)[1][:, 1:]
    #     for i, neigh_indices in enumerate(neighbors):
    #         neighbor_positions = positions[neigh_indices]
    #         pos_graphs.append(neighbor_positions)
    #         y_labels.append(labels[i])
    for system in tqdm(all_defective_positions):
        _, pos_graph = construct_graph_lists(
            pos_individual=system,
            num_neighbors=12,
            cell=None,
        )
        pos_graphs.extend(pos_graph)
    y_labels = np.concatenate(all_labels)
    return np.array(pos_graphs), np.array(y_labels), label_map