import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from modal import Volume
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from egnn_crystal_classifier.data_prep.data_handler import CrystalDataset, FastLoader
from egnn_crystal_classifier.ml_model.model import EGNN
from egnn_crystal_classifier.ml_train.hparams import HParams


def get_loaders(
    coord_path: Path,
    label_path: Path,
    label_map_path: Path,
    device: torch.device,
    hp: HParams,
) -> tuple[FastLoader, FastLoader, FastLoader]:
    # Load data and do some quick checks
    pos_graphs = np.load(coord_path)
    with label_path.open("r", encoding="utf-8") as f:
        label_strs = json.load(f)
    with label_map_path.open("r", encoding="utf-8") as f:
        label_map = json.load(f)

    assert pos_graphs.shape[0] == len(
        label_strs
    ), "lengths of pos_graphs and label_strs must be the same"
    assert set(label_strs) == set(
        label_map
    ), "label_strs must be consistent with label_map"

    # Prep data
    num_data_points = pos_graphs.shape[0]
    train_section = int(hp.train_split_frac * num_data_points)
    train_eval_size = int(hp.train_eval_sample_frac * train_section)

    use_indices = np.random.permutation(num_data_points)

    # Indices for dataset
    train_indices = use_indices[0:train_section]
    train_eval_indices = np.random.choice(
        train_indices, size=train_eval_size, replace=False
    )
    test_indices = use_indices[train_section:num_data_points]

    # Datasets
    train_dataset = CrystalDataset(
        pos_graphs[train_indices], [label_strs[i] for i in train_indices], label_map
    )
    train_eval_dataset = CrystalDataset(
        pos_graphs[train_eval_indices],
        [label_strs[i] for i in train_eval_indices],
        label_map,
    )
    test_dataset = CrystalDataset(
        pos_graphs[test_indices], [label_strs[i] for i in test_indices], label_map
    )

    # Dataloaders
    train_loader = FastLoader(
        train_dataset, hp.batch_size, hp.num_buckets, device, shuffle=True
    )
    train_eval_loader = FastLoader(
        train_eval_dataset, hp.batch_size, hp.num_buckets, device, shuffle=False
    )
    test_loader = FastLoader(
        test_dataset, hp.batch_size, hp.num_buckets, device, shuffle=False
    )
    return train_loader, train_eval_loader, test_loader


def train_epoch(
    model: EGNN,
    loader: FastLoader,
    criterion: CrossEntropyLoss,
    optimizer: AdamW,
) -> tuple[float, float]:
    model.train()
    total_loss = 0
    total_correct = 0

    for data in loader:
        optimizer.zero_grad()
        logits, _ = model(data.to(next(model.parameters()).device))
        pred = logits.argmax(dim=1)

        loss = criterion(logits, data.y)
        loss.backward()

        clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * pred.shape[0]
        total_correct += int((pred == data.y).sum())

    avg_loss = total_loss / len(loader.dataset)
    avg_accuracy = total_correct / len(loader.dataset)

    return avg_loss, avg_accuracy


@torch.no_grad()
def evaluate_model(
    model: EGNN, loader: FastLoader, criterion: CrossEntropyLoss
) -> tuple[float, float]:
    model.eval()
    total_loss = 0
    total_correct = 0

    for data in loader:
        logits, _ = model(data.to(next(model.parameters()).device))
        pred = logits.argmax(dim=1)
        total_loss += criterion(logits, data.y).item() * pred.shape[0]
        total_correct += int((pred == data.y).sum())

    avg_loss = total_loss / len(loader.dataset)
    avg_accuracy = total_correct / len(loader.dataset)

    return avg_loss, avg_accuracy


def plot_training_curves(
    exp_path: Path,
    vol: Volume | None,
    train_losses: list[float],
    test_losses: list[float],
    train_accuracies: list[float],
    test_accuracies: list[float],
) -> None:
    assert (
        len(train_losses)
        == len(test_losses)
        == len(train_accuracies)
        == len(test_accuracies)
    ), "train and test info must have same length"

    epochs = list(range(1, len(train_losses) + 1))

    fig, ax1 = plt.subplots()

    # Loss curves
    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, test_losses, label="Test Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, which="both", axis="x", linestyle=":", linewidth=0.5)

    # Accuracy curves on twin y-axis
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_accuracies, label="Train Acc", linestyle="--")
    ax2.plot(epochs, test_accuracies, label="Test Acc", linestyle="--")
    ax2.set_ylabel("Accuracy")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("Training Loss & Accuracy")
    plt.tight_layout()

    out_path = exp_path / "training_curves.png"
    plt.savefig(out_path)
    plt.close(fig)

    if vol is not None:
        vol.commit()


def train(
    exp_path: Path,
    coord_path: Path,
    label_path: Path,
    label_map_path: Path,
    vol: Volume | None,
    device: torch.device,
    hp: HParams,
) -> None:
    # Prepare info
    train_loader, train_eval_loader, test_loader = get_loaders(
        coord_path, label_path, label_map_path, device, hp
    )

    model = EGNN(
        hp.num_buckets,
        hp.num_hidden,
        hp.num_reg_layers,
        hp.num_classes,
        hp.dropout_prob,
    ).to(device)

    criterion = CrossEntropyLoss(label_smoothing=hp.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=hp.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=hp.epochs)

    train_losses: list[float] = []
    test_losses: list[float] = []

    train_accuracies: list[float] = []
    test_accuracies: list[float] = []

    best_test_acc: float = 0.0

    # Initialize directories
    exp_path.mkdir(parents=True, exist_ok=True)
    (exp_path / "models").mkdir(parents=True, exist_ok=True)

    with open(exp_path / "log.txt", "w", encoding="utf-8") as f:
        f.write(f"{hp}\n")
    if vol is not None:
        vol.commit()

    # Training loop
    for epoch in range(1, hp.epochs + 1):
        # Train loss from training and everything else on eval mode
        train_loss, _ = train_epoch(model, train_loader, criterion, optimizer)

        _, train_acc = evaluate_model(model, train_eval_loader, criterion)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion)

        scheduler.step()

        # Log info
        log_line = (
            f"Epoch {epoch:03d}: "
            f"Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, "
            f"Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}"
        )
        with open(exp_path / "log.txt", "a", encoding="utf-8") as f:
            f.write(f"{log_line}\n")

        # Save the model at every checkpoint or when we have a new best accuracy
        state_dict_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(state_dict_cpu, exp_path / "models" / f"model_best.pth")

        # Checkpoint
        if epoch % hp.checkpoint_freq == 0:
            print(log_line)
            torch.save(state_dict_cpu, exp_path / "models" / f"model_{epoch}.pth")

        # Save for plotting
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # Commit all changes to volume
        if vol is not None:
            vol.commit()

    plot_training_curves(
        exp_path, vol, train_losses, test_losses, train_accuracies, test_accuracies
    )
