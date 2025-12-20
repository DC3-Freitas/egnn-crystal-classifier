"""
Handles processing raw positions similar to OVITO nearest neighbor finder.
Allows you to get indices for neighbors and PBC-adjusted positions of each
atom's local structure.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree


def construct_graph_lists(
    pos_individual: NDArray[np.float32],
    num_neighbors: int,
    cell: NDArray[np.float32] | None,
) -> tuple[NDArray[np.integer[Any]], NDArray[np.float32]]:
    """
    Creates neighbors (in terms of indices) and (numpy) list of (potentially modifed) positions
    of atoms for each graph.

    Each atom is treated as a graph center with its num_neighbors
    nearest neighbors being part of its graph. The positions (not indices)
    of these (num_nodes = num_neighbors + 1) make up an entry in our graph's (numpy) list.

    If a cell is provided, it will attempt to apply PBC.

    Also ensures that the central atom always comes first.

    Args:
        pos_individual (B, 3): Coordinates of each atom.
        num_neighbors: Number of nearest neighbors to consider for each atom.
        cell (3, 4) | None: Ovito simulation cell (we ignore the pbc flags and assume
                            pbc is applied everywhere).

    Returns:
        neighbors (B, num_nodes): The indices of the nearest neighbor for each atom.
        pos_graph (B, num_nodes, 3): The positions (potentially adjusted according to PBC)
                                     of each of the num_neighbors for each atom (with the
                                     first entry of the second dimension always being the
                                     atom itself).
    """
    pos_individual = np.asarray(pos_individual, dtype=np.float32)

    if pos_individual.ndim != 2 or pos_individual.shape[1] != 3:
        raise ValueError("pos_individual must be a 2D array with shape (N, 3)")

    if cell is not None:
        boxsize = np.max(cell[:, :3], axis=1)
        pos_individual -= cell[:, 3]
        pos_individual %= boxsize

        print("Constructing graph with boxsize:", boxsize)
        tree = cKDTree(pos_individual, boxsize=boxsize)
    else:
        print("Constructing graph with no boxsize provided")
        tree = cKDTree(pos_individual)

    neighbors: NDArray[np.integer[Any]] = np.asarray(
        tree.query(pos_individual, k=num_neighbors + 1)[1]
    )

    # Each data entry will store all NN positions with the first one being the central atom
    pos_graph = pos_individual[neighbors]

    # Apply PBC
    if cell is not None:
        for group in pos_graph:
            anchor = group[0]
            group -= anchor
            pbc_fact = boxsize * np.round(group / boxsize)
            group -= pbc_fact
    assert np.all(neighbors[:, 0] == np.arange(len(pos_individual)))

    return neighbors, pos_graph
