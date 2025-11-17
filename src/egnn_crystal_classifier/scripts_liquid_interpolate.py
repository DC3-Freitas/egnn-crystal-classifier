import os
import math
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from egnn_crystal_classifier.data_gen import gen_liquid_interpolated
from egnn_crystal_classifier.ml_model.model import EGNN
from egnn_crystal_classifier.ml_train.hparams import HParams
from egnn_crystal_classifier.constants import BASE_DIR

from torch.utils.data import Dataset
from torch_geometric.data import Data
from numpy.typing import NDArray
from typing import Any, Iterator
import numpy as np
from egnn_crystal_classifier.data_prep.graph_construction import construct_batched_graph
from tqdm import tqdm

class CrystalDatasetNumeric(Dataset[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]):
    def __init__(self, 
        pos_graphs: NDArray[np.number[Any]], 
        labels: NDArray[np.number[Any]] | None,
    ) -> None:
        self.pos_graphs = torch.from_numpy(pos_graphs).float()
        self.labels = torch.from_numpy(labels).float() if labels is not None else None
    def __len__(self) -> int:
        return self.pos_graphs.shape[0]
    def __getitem__(self, idx: int | slice | list[int] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        pos_graphs_ret = self.pos_graphs[idx].contiguous()
        labels_ret = self.labels[idx].contiguous() if self.labels is not None else None
        return pos_graphs_ret, labels_ret
    
class FastLoaderNumeric:
    def __init__(self,
        dataset: CrystalDatasetNumeric,
        batch_size: int,
        num_buckets: int,
        device: torch.device,
        shuffle: bool,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_buckets = num_buckets
        self.shuffle = shuffle
        self.num_batches = (len(dataset) + batch_size - 1) // batch_size
        self.calc_device = device
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


def create_data_loaders_liquid_interpolated(
    batch_size: int,
    num_buckets: int,
    calc_device: torch.device,
) -> tuple[FastLoaderNumeric, FastLoaderNumeric]:
    positions, labels = gen_liquid_interpolated.gen()
    labels = np.array(labels, dtype=np.float32)
    print(positions.shape, labels.shape)
    indices = np.random.permutation(positions.shape[0])
    split_idx = int(0.8 * positions.shape[0])
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    train_dataset = CrystalDatasetNumeric(
        positions[train_indices], labels[train_indices]
    )
    test_dataset = CrystalDatasetNumeric(
        positions[test_indices], labels[test_indices]
    )
    train_loader = FastLoaderNumeric(
        train_dataset, batch_size, num_buckets, calc_device, shuffle=True
    )
    test_loader = FastLoaderNumeric(
        test_dataset, batch_size, num_buckets, calc_device, shuffle=False
    )
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"Position graph shape: {positions.shape}, Labels shape: {labels.shape}")
    print(f"Sample position graph: {positions[0]}, Sample label: {labels[0]}")
    return train_loader, test_loader

def create_model_liquid_interpolated(
    hp: HParams,
    device: torch.device,
) -> EGNN:
    model = EGNN(
        num_buckets=hp.num_buckets,
        hidden=hp.num_hidden,
        num_reg_layers=hp.num_reg_layers,
        num_classes=hp.num_classes,
        dropout_prob=hp.dropout_prob,
    ).to(device)
    print(model)
    return model
            
def create_optimizer_scheduler_liquid_interpolated(
    model: EGNN,
    hp: HParams,
) -> tuple[AdamW, CosineAnnealingLR]:
    optimizer = AdamW(model.parameters(), lr=hp.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=hp.epochs)
    return optimizer, scheduler

def train(hp: HParams) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = create_data_loaders_liquid_interpolated(
        hp.batch_size, hp.num_buckets, device
    )
    model = create_model_liquid_interpolated(hp, device)
    optimizer, scheduler = create_optimizer_scheduler_liquid_interpolated(model, hp)
    criterion = MSELoss() # CrossEntropyLoss()
    # prepare model saving
    model_dir = os.path.join(BASE_DIR, "ml_model")
    os.makedirs(model_dir, exist_ok=True)
    best_loss = math.inf
    best_model_path = os.path.join(model_dir, "liquid_interpolated_model_best.pth")
    for epoch in range(hp.epochs):
        # training
        print(f"Starting epoch {epoch+1}/{hp.epochs}")
        model.train()
        total_loss = 0.0
        for data in tqdm(train_loader):
            optimizer.zero_grad()
            logits, _ = model(data.to(next(model.parameters()).device))
            loss = criterion(torch.sigmoid(logits).squeeze(), data.y)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * data.y.shape[0]
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{hp.epochs}, Training Loss: {avg_loss:.4f}")
        scheduler.step()

        # evaluation
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            # accuracy = 0.0
            for data in test_loader:
                logits, _ = model(data.to(next(model.parameters()).device))
                loss = criterion(torch.sigmoid(logits).squeeze(), data.y)
                total_loss += loss.item() * data.y.shape[0]
                # accuracy += (logits.argmax(dim=1) == torch.argmax(data.y, dim=1)).sum().item()
            # accuracy /= len(test_loader.dataset)
            avg_loss = total_loss / len(test_loader.dataset)
            print(f"Epoch {epoch+1}/{hp.epochs}, Test Loss: {avg_loss:.4f}")
            # save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model (loss {best_loss:.4f}) -> {best_model_path}")

    # also save final model
    final_model_path = os.path.join(model_dir, "liquid_interpolated_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model -> {final_model_path}")

if __name__ == "__main__":
    hparams = HParams(
        num_classes=1,
        lr=1e-3,
        batch_size=512,
        epochs=20
    )
    train(hparams)