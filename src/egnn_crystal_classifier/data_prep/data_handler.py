from typing import Any, Iterator

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset
from torch_geometric.data import Data

from egnn_crystal_classifier.data_prep.graph_construction import construct_batched_graph


class CrystalDataset(Dataset[tuple[torch.Tensor, torch.Tensor | None]]):
    def __init__(
        self,
        pos_graphs: NDArray[np.number[Any]],
        label_strs: list[str] | None,
        label_map: dict[str, int] | None,
    ) -> None:
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
        return self.pos_graphs.shape[0]

    def __getitem__(
        self, idx: int | slice | list[int] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pos_graphs_ret = self.pos_graphs[idx].contiguous()
        label_ints_ret = (
            self.label_ints[idx].contiguous() if self.label_ints is not None else None
        )
        return pos_graphs_ret, label_ints_ret


class FastLoader:
    def __init__(
        self,
        dataset: CrystalDataset,
        batch_size: int,
        num_buckets: int,
        calc_device: torch.device,
        shuffle: bool,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_buckets = num_buckets
        self.calc_device = calc_device
        self.shuffle = shuffle
        self.num_batches = (len(dataset) + batch_size - 1) // batch_size

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[Data]:
        indices = torch.arange(len(self.dataset))

        if self.shuffle:
            indices = indices[torch.randperm(indices.shape[0])]

        for start in range(0, indices.shape[0], self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            pos_graphs, label_ints = self.dataset[batch_indices]

            yield construct_batched_graph(
                pos_graphs, label_ints, self.num_buckets, self.calc_device
            )
