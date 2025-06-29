"""
Okay, I am assuming data gen might be subject to change so it is very hacked together right now
and triggers tons of mypy errors. However, I think that having it follow some gen() which returns
x_data, y_data, map or smth is the play here.

:skull:
"""

import os

import numpy as np
from ovito.io import import_file
from scipy.spatial import cKDTree
from egnn_crystal_classifier.constants import *


def extract_center(fname):
    """
    Extracts a small sample from the center of a structure
    """

    # Load data
    pipeline = import_file(fname)
    lattice = pipeline.compute()
    positions = np.copy(lattice.particles.positions)

    # crop center
    # synth data has box size ~ n = 10, so we cut the samples of size 1
    center = np.mean(positions, axis=0) + np.random.normal(0, 0.5, size=3)
    size_xz = CHECKERBOARD_CELL_SIZE_XZ
    size_y = CHECKERBOARD_CELL_SIZE_Y
    cropped_positions = positions[
        (positions[:, 0] > center[0] - size_xz / 2)
        & (positions[:, 0] < center[0] + size_xz / 2)
        & (positions[:, 1] > center[1] - size_y / 2)
        & (positions[:, 1] < center[1] + size_y / 2)
        & (positions[:, 2] > center[2] - size_xz / 2)
        & (positions[:, 2] < center[2] + size_xz / 2)
    ]

    # normalize coordinates
    normalized_positions = cropped_positions - np.min(cropped_positions, axis=0)
    return normalized_positions


def create_checker_structures():
    """
    Creates bi-structured checkerboards from the synthetic data in order to
    help model generalize and work better in interfaces between different structures.
    """
    available_temps = os.listdir(os.path.join(SYNTH_DATA_PATH, "bcc"))
    structure_names = os.listdir(SYNTH_DATA_PATH)

    x_data = []
    y_data = []

    for temp in available_temps:
        for i in range(len(structure_names)):
            for j in range(i):
                # extract centers
                struct_a, struct_b = structure_names[i], structure_names[j]

                # build checkerboard data
                checkerboard_positions = np.array([])
                parity_ctr = 0
                sz = CHECKERBOARD_CELL_SIZE_XZ
                sz_y = CHECKERBOARD_CELL_SIZE_Y
                labels = []
                for x in range(CHECKERBOARD_CELL_COUNT):
                    for y in range(CHECKERBOARD_CELL_COUNT * 2):
                        for z in range(CHECKERBOARD_CELL_COUNT):
                            center_a = extract_center(
                                os.path.join(SYNTH_DATA_PATH, struct_a, temp)
                            )
                            center_b = extract_center(
                                os.path.join(SYNTH_DATA_PATH, struct_b, temp)
                            )
                            if parity_ctr == 0:
                                if len(checkerboard_positions) == 0:
                                    checkerboard_positions = center_a + np.array(
                                        [x * sz, y * sz, z * sz]
                                    )
                                else:
                                    checkerboard_positions = np.vstack(
                                        (
                                            checkerboard_positions,
                                            center_a
                                            + np.array([x * sz, y * sz_y, z * sz]),
                                        )
                                    )
                                labels += [struct_a] * len(center_a)
                            else:
                                checkerboard_positions = np.vstack(
                                    (
                                        checkerboard_positions,
                                        center_b + np.array([x * sz, y * sz_y, z * sz]),
                                    )
                                )
                                labels += [struct_b] * len(center_b)
                            parity_ctr = 1 - parity_ctr

                x_data.append(checkerboard_positions)
                y_data.append(labels)

    return x_data, y_data


def gen(use_checker: bool = True):
    # Extract data
    x_data = []
    y_data = []

    structure_cnts = {}

    for structure in os.listdir(SYNTH_DATA_PATH):
        for f in os.listdir(os.path.join(SYNTH_DATA_PATH, structure)):
            pipeline = import_file(os.path.join(SYNTH_DATA_PATH, structure, f))
            lattice = pipeline.compute()
            all_positions = np.copy(lattice.particles.positions)
            x_data.append(all_positions)
            y_data.append([structure] * len(all_positions))
            structure_cnts[structure] = structure_cnts.get(structure, 0) + len(
                all_positions
            )

    if use_checker:
        x_data_checker, y_data_checker = create_checker_structures()
        x_data.extend(x_data_checker)
        y_data.extend(y_data_checker)

    # Extract data
    x_data_new = []
    y_data_new = []

    for all_positions, all_labels in zip(x_data, y_data):
        # Add one to include self
        tree = cKDTree(all_positions)
        neighbors = tree.query(all_positions, k=NN_COUNT + 1)[1][:]

        # Each X will store all nearest neighbor positions with the first one being the central atom
        for i, neigh in enumerate(neighbors):
            x_data_new.append(np.array([all_positions[i] for i in neigh]))
            assert i == neigh[0]
            y_data_new.append(all_labels[i])

    x_data = np.array(x_data_new)
    y_data = y_data_new
    label_map = {label: i for i, label in enumerate(sorted(set(y_data)))}

    return x_data, y_data, label_map
