"""
Utilities for constructing per-atom local graphs in an efficient
batched manner.
"""

from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from torch_geometric.data import Data


def construct_graph_lists(
    pos_individual: NDArray[np.float32],
    num_neighbors: int,
    cell: NDArray[np.float32] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Creates (numpy) list of (potentially modifed) positions of atoms for each graph.

    Each atom is treated as a graph center with its num_neighbors
    nearest neighbors being part of its graph. The positions (not indices)
    of these (num_nodes = num_neighbors + 1) make up an entry in our graphs (numpy) list.

    If a cell is provided, it will attempt to apply PBC.

    Also ensures that the central atom always comes first.

    Args:
        pos_individual (B, 3): Coordinates of each atom.
        num_neighbors: Number of nearest neighbors to consider for each atom.
        cell (3, 4): Ovito simulation cell (we ignore the pbc flags and assume
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
        tree = cKDTree(pos_individual)

    neighbors = tree.query(pos_individual, k=num_neighbors + 1)[1]

    # Each data entry will store all NN positions with the first one being the central atom
    pos_graph = pos_individual[neighbors]
    pos_graph = cast(NDArray[np.number[Any]], pos_graph)

    # Apply PBC
    if cell is not None:
        for group in pos_graph:
            anchor = group[0]
            group -= anchor
            pbc_fact = boxsize * np.round(group / boxsize)
            group -= pbc_fact
    assert np.all(neighbors[:, 0] == np.arange(len(pos_individual)))

    return neighbors, pos_graph


def normalize_position_batch(
    pos_graphs: torch.Tensor, edge_index_single: torch.Tensor
) -> torch.Tensor:
    """
    Normalize coordinates by the mean edge length. We assume that the
    edge structure of each graph is the exact same.

    Args:
        pos_graphs (B, num_nodes, 3): Positions of atoms for each graph.
        edge_index_single (2, E): Edges of the graph structure we assume every
                                  graph to follow (e.g. complete graph).

    Returns:
        Normalized version of pos_graphs with the same shape: (B, num_nodes, 3).
    """
    diffs = (
        pos_graphs[:, edge_index_single[1], :] - pos_graphs[:, edge_index_single[0], :]
    )
    dists = diffs.norm(dim=-1)
    mean_edge_len = dists.mean(dim=-1, keepdim=True)

    if torch.any(mean_edge_len == 0):
        raise ValueError("One of the graphs has zero mean edge length.")

    return cast(torch.Tensor, pos_graphs / mean_edge_len.unsqueeze(-1))


def create_complete_graph_edges_single(num_nodes: int) -> torch.Tensor:
    """
    Creates the edge index for a single complete directed graph
    excluding self-edges.

    Args:
        num_nodes: Number of nodes (should typically be num_neighbors + 1).

    Returns:
        Tensor of shape (2, num_nodes * (num_nodes - 1)) containing edges
        in PyGeometric format.
    """
    row = torch.arange(0, num_nodes).repeat_interleave(num_nodes - 1).long()
    col = torch.cat(
        [
            torch.cat(
                (
                    torch.arange(0, i),
                    torch.arange(i + 1, num_nodes),
                )
            )
            for i in range(0, num_nodes)
        ]
    ).long()
    return torch.stack([row, col])


def create_center_mask_single(num_nodes: int) -> torch.Tensor:
    """
    Creates a boolean mask idenitfying the central atom in a local graph.
    Assumes the central atom is at position 0.

    Args:
        num_nodes: Number of nodes in the graph (inclusive of center and neighbors).

    Returns:
        Boolean tensor of shape (num_nodes,) with index 0 True (since index 0 is assumed
        to correspond to the central atom) and all others False.
    """
    center_mask = torch.zeros(num_nodes).bool()
    center_mask[0] = True
    return center_mask


def construct_batched_graph(
    pos_graphs: torch.Tensor, label_ints: torch.Tensor | None
) -> Data:
    """
    Manually builds a PyGeometric Data object representing a batch
    of graphs.

    Each graph is complete and assumed to have the same number of nodes.
    Positions of each graph are also normalized by edge length.

    Args:
        pos_graphs (B, num_nodes, 3): Pre-normalized positions of atoms for each graph.
        label_ints (B,): Integer labels (y). Can be None (usually done during inference).

    Returns:
        A PyG data object with the following fields where N = B * num_nodes:
            x (N,): Initial node species identifier for embeddings.
            pos (N, 3): Normalized positions of each atom.
            edge_index (2, B * num_nodes * (num_nodes - 1)): Complete graph edge index.
            center_mask (N,): Boolean tensor marking each graph's center node with True
            batch (N,): The batch that each node belongs to.
            y (N,) | None: Optional class labels
    """
    batches, num_nodes_single = pos_graphs.shape[0], pos_graphs.shape[1]
    num_edges = num_nodes_single * num_nodes_single - num_nodes_single

    # Complete graph with all edges relating to 0 (center) taking up the prefix
    edge_index_single = create_complete_graph_edges_single(num_nodes_single)
    offsets = (torch.arange(batches) * num_nodes_single).view(1, -1, 1)
    edge_index_all = (edge_index_single.view(2, 1, num_edges) + offsets).reshape(
        2, batches * num_edges
    )

    # Normalize positions
    pos_norm_batched = normalize_position_batch(pos_graphs, edge_index_single).reshape(
        batches * num_nodes_single, 3
    )

    # Mask
    center_mask = create_center_mask_single(num_nodes_single).repeat(batches)

    # Species for embedding
    species = torch.zeros(batches * num_nodes_single, dtype=torch.long)

    # Batch
    batch = torch.arange(0, batches).repeat_interleave(num_nodes_single)

    return Data(
        x=species,
        num_nodes=batches * num_nodes_single,
        pos=pos_norm_batched,
        edge_index=edge_index_all,
        center_mask=center_mask,
        batch=batch,
        y=label_ints,
    ).cpu()
