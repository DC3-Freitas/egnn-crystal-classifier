import math
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from torch_geometric.data import Data


def construct_graph_lists(
    pos_individual: NDArray[np.number[Any]], num_neighbors: int
) -> tuple[NDArray[np.number[Any]], NDArray[np.number[Any]]]:
    tree = cKDTree(pos_individual)
    neighbors = tree.query(pos_individual, k=num_neighbors + 1)[1]

    # Each data entry will store all nearest neighbor positions with the first one being the central atom
    pos_graph = pos_individual[neighbors]
    assert np.all(neighbors[:, 0] == np.arange(len(pos_individual)))

    return neighbors, cast(NDArray[np.number[Any]], pos_graph)


def angle_histogram_batched(
    pos_graphs: torch.Tensor, num_buckets: int, device: torch.device
) -> torch.Tensor:
    # Must be (B, N, 3) and have more than 2 atoms
    assert pos_graphs.dim() == 3 and pos_graphs.shape[1] > 2
    pos_graphs = pos_graphs.to(device)
    batches, num_atoms = pos_graphs.shape[0], pos_graphs.shape[1]

    centers = (torch.arange(num_buckets, device=device) + 0.5) / num_buckets

    # Pairwise unit vectors vec_{bij} = unit_vec(pos_{bj} - pos_{bj\i})
    # Dimensions: (B, N, N, 3)
    unit_vecs = pos_graphs[:, :, None, :] - pos_graphs[:, None, :, :]
    unit_vecs = unit_vecs / torch.linalg.norm(
        unit_vecs, dim=-1, keepdim=True
    ).clamp_min_(1e-12)

    # cos_vals_{bijk} = dot product between i-->j and i-->k unit vectors of batch b
    # Dimensions: (B, N, N, N)
    cos_vals = torch.einsum("bijc,bikc->bijk", unit_vecs, unit_vecs).clamp(-1.0, 1.0)
    angles = torch.acos(cos_vals) / math.pi

    # Broadcast last dimension of angles to get difference of every angle (b, i, j, k) with every center
    # Dimensions: (B, N, N, N, num_buckets)
    diff = angles.unsqueeze(-1) - centers
    weight = torch.clamp(1.0 - diff.abs() * num_buckets, min=0.0)

    # j == i, k == i, and k == j are bad
    # Dimensions (N, N, N)
    idx = torch.arange(num_atoms, device=device, dtype=pos_graphs.dtype)
    i_idx = idx.view(num_atoms, 1, 1)
    j_idx = idx.view(1, num_atoms, 1)
    k_idx = idx.view(1, 1, num_atoms)
    valid = ~((j_idx == i_idx) | (k_idx == i_idx) | (k_idx == j_idx))

    weight *= valid.view(1, num_atoms, num_atoms, num_atoms, 1)

    # Get histogram (B, N, N, num_buckets)
    hist = weight.sum(dim=3) / (num_atoms - 2)
    mask = (~torch.eye(num_atoms, dtype=torch.bool, device=device)).flatten()

    # Return histogram for each edge
    return hist.reshape(batches, num_atoms * num_atoms, num_buckets)[:, mask, :]


def normalize_position_batch(
    pos_graphs: torch.Tensor, edge_index_single: torch.Tensor
) -> torch.Tensor:
    diffs = (
        pos_graphs[:, edge_index_single[1], :] - pos_graphs[:, edge_index_single[0], :]
    )
    dists = diffs.norm(dim=-1)
    mean_edge_len = dists.mean(dim=-1, keepdim=True)

    if torch.any(mean_edge_len == 0):
        raise ValueError("One of the graphs has zero mean edge length.")

    return cast(torch.Tensor, pos_graphs / mean_edge_len.unsqueeze(-1))


def create_complete_graph_edges_single(
    num_nodes: int, device: torch.device
) -> torch.Tensor:
    row = (
        torch.arange(0, num_nodes, device=device)
        .repeat_interleave(num_nodes - 1)
        .long()
    )
    col = torch.cat(
        [
            torch.cat(
                (
                    torch.arange(0, i, device=device),
                    torch.arange(i + 1, num_nodes, device=device),
                )
            )
            for i in range(0, num_nodes)
        ]
    ).long()
    return torch.stack([row, col])


def create_center_mask_single(num_nodes: int, device: torch.device) -> torch.Tensor:
    center_mask = torch.zeros(num_nodes, device=device).bool()
    center_mask[0] = True
    return center_mask


def construct_batched_graph(
    pos_graphs: torch.Tensor,
    label_ints: torch.Tensor | None,
    num_buckets: int,
    calc_device: torch.device,
) -> Data:
    pos_graphs = pos_graphs.to(calc_device)
    batches, num_nodes_single = pos_graphs.shape[0], pos_graphs.shape[1]
    num_edges = num_nodes_single * num_nodes_single - num_nodes_single

    # Complete graph with all edges relating to 0 (center) taking up the prefix
    edge_index_single = create_complete_graph_edges_single(
        num_nodes_single, calc_device
    )
    offsets = (torch.arange(batches, device=calc_device) * num_nodes_single).view(
        1, -1, 1
    )
    edge_index_all = (edge_index_single.view(2, 1, num_edges) + offsets).reshape(
        2, batches * num_edges
    )

    # Normalize positions
    pos_norm_batched = normalize_position_batch(pos_graphs, edge_index_single).reshape(
        batches * num_nodes_single, 3
    )

    # Histogram
    edge_angle_hist = angle_histogram_batched(
        pos_graphs, num_buckets, calc_device
    ).reshape(-1, num_buckets)

    # Mask
    center_mask = create_center_mask_single(num_nodes_single, calc_device).repeat(
        batches
    )

    return Data(
        num_nodes=batches * num_nodes_single,
        pos_norm=pos_norm_batched,
        edge_index=edge_index_all,
        edge_angle_hist=edge_angle_hist,
        center_mask=center_mask,
        y=label_ints,
    ).cpu()
