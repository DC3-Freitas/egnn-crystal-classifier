"""
The Dataset and DataLoader portion of our pipeline.
Done manually in order to support efficient batching and graph construction.
"""

from typing import Iterator

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset
from torch_geometric.data import Data

from egnn_crystal_classifier.data_prep.graph_construction import construct_batched_graph


class CrystalDataset(Dataset[tuple[torch.Tensor, torch.Tensor | None]]):
    """
    Dataset of per-atom local graphs represented as pre-normalized
    position tensors and optional integer labels.
    """

    def __init__(
        self,
        pos_graphs: NDArray[np.float32],
        label_strs: list[str] | None,
        label_map: dict[str, int] | None,
    ) -> None:
        """
        Initializes dataset with graphs. Providing labels is optional
        but if label_strs is provided, label_map must also be provided.

        Args:
            pos_graphs (B, num_nodes, 3): Pre-normalized positions of atoms for each graph.
            label_strs: Optional labels for each atom in the form of a string (e.g. "bcc").
            label_map: Optional mapping from string to integer. If label_strs provided, label_map
                       must also be provided.
        """
        self.pos_graphs = torch.from_numpy(pos_graphs).float()
        self.label_map = label_map

        assert not (
            label_strs is not None and self.label_map is None
        ), "if labels are present, a label map must also be present"

        self.label_ints = (
            torch.tensor([self.label_map[label] for label in label_strs]).long()
            if label_strs is not None and self.label_map is not None
            else None
        )

    def __len__(self) -> int:
        """Returns the number of graphs."""
        return self.pos_graphs.shape[0]

    def __getitem__(
        self, idx: int | slice | list[int] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Retrieves positions and optional labels for one or more graphs.

        Args:
            idx: Index or indices selecting which graphs to return.

        Returns:
            pos_graphs_ret (B, num_nodes, 3) | (num_nodes, 3): Selected graphs of positions.
            label_ints_ret (B,) | (1,) | None: Labels of the graphs if provided.
        """
        pos_graphs_ret = self.pos_graphs[idx].contiguous()
        label_ints_ret = (
            self.label_ints[idx].contiguous() if self.label_ints is not None else None
        )
        return pos_graphs_ret, label_ints_ret


class FastLoader:
    """
    DataLoader for CrystalDataset. Handles batching efficiently,
    converts to graphs on the fly and supports shuffling (via
    a custom iterator).
    """

    def __init__(
        self,
        dataset: CrystalDataset,
        batch_size: int,
        shuffle: bool,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_batches = (len(dataset) + batch_size - 1) // batch_size

    def __len__(self) -> int:
        """Returns the number of batches."""
        return self.num_batches

    def __iter__(self) -> Iterator[Data]:
        """
        Iterates over dataset batches in a potentially randomized order.

        Yields:
            Next batch of data. See construct_batched_graph for what to
            expect to be yielded.
        """
        indices = torch.arange(len(self.dataset))

        if self.shuffle:
            indices = indices[torch.randperm(indices.shape[0])]

        for start in range(0, indices.shape[0], self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            pos_graphs, label_ints = self.dataset[batch_indices]

            yield construct_batched_graph(pos_graphs, label_ints)
