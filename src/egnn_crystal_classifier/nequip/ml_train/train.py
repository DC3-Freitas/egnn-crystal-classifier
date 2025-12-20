"""
All training utilities and pipeline for training nequip.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from egnn_crystal_classifier.config import NequIPConfig
from egnn_crystal_classifier.nequip.data_prep.data_handler import FastLoader
from egnn_crystal_classifier.nequip.ml_model.model import NequIP
from egnn_crystal_classifier.utils.seed import set_seed


def train_epoch(
    model: NequIP,
    loader: FastLoader,
    criterion: CrossEntropyLoss,
    optimizer: AdamW,
) -> tuple[float, float]:
    """
    Runs a single epoch through the data loader and reports loss and
    accuracy on train mode.

    Args:
        model: The model being trained.
        loader: Data loader providing batches of PyGeometric Data objects.
        criterion: Classification loss function (CrossEntropyLoss).
        optimizer: Optimizer for the model (AdamW).

    Returns:
        avg_loss: Mean loss over the entire dataset on train mode.
        avg_accuracy: Fraction of correctly classified samples over the dataset.
        (These values are representative of the model during train mode)
    """
    model.train()

    total_loss = 0
    total_correct = 0

    pbar = tqdm(loader, desc="Training", leave=False)

    for data in pbar:
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


def evaluate_model(
    model: NequIP, loader: FastLoader, criterion: CrossEntropyLoss
) -> tuple[float, float]:
    """
    Evaluates the model on all the data in the given data loader
    and reports loss and accuracy on eval mode.

    Args:
        model: The model being trained.
        loader: Data loader providing batches of PyGeometric Data objects.
        criterion: Classification loss function (CrossEntropyLoss).

    Returns:
        avg_loss: Mean loss over the entire dataset.
        avg_accuracy: Fraction of correctly classified samples over the dataset.
        (These values are representative of the model during eval mode)
    """
    model.eval()

    total_loss = 0
    total_correct = 0

    pbar = tqdm(loader, desc="Evaluating", leave=False)

    for data in pbar:
        with torch.inference_mode():
            logits, _ = model(data.to(next(model.parameters()).device))

        loss = criterion(logits, data.y)
        pred = logits.argmax(dim=1)

        total_loss += loss.item() * pred.shape[0]
        total_correct += int((pred == data.y).sum())

    avg_loss = total_loss / len(loader.dataset)
    avg_accuracy = total_correct / len(loader.dataset)

    pbar.close()

    return avg_loss, avg_accuracy


def plot_training_curves(
    exp_path: Path,
    train_losses: list[float],
    test_losses: list[float],
    train_accuracies: list[float],
    test_accuracies: list[float],
) -> None:
    """
    Plots and saves loss and accuracy curves (all on the same graph).

    Args:
        exp_path: Outer directory of the expirement which is where the
                  figure will be saved.
        train_losses: Losses from each epoch for train set.
        test_losses: Losses from each epoch for test set.
        train_accuracies: Accuricies from each epoch for train set.
        test_accuracies: Accuricies from each epoch for test set.
    """
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


# pylint: disable=too-many-locals, too-many-arguments
def train_nequip(
    config: NequIPConfig,
    exp_path: Path,
    train_loader: FastLoader,
    train_eval_loader: FastLoader,
    test_loader: FastLoader,
    class_weights: torch.Tensor,
    device: torch.device,
) -> None:
    """
    Runs the full training pipeline given the necessary
    information. Saves all information (checkpoint, logs, and
    graph).

    Args:
        config: Contains all training hyperparameter information.
                See Config class for more details.
        exp_path: Outer directory for experiments which will contain
                  model checkpoints and logs.
        train_loader: Training set data loader.
        train_eval_loader: Training set evaluaiton data loader (subset of
                           training set).
        test_loader: Test set data loader.
        class_weights: Class weights to remove bias from imbalanced data.
        device: Device on which the model is trained (e.g. cuda).
    """
    # Prepare info
    set_seed()
    model = NequIP(config).to(device)

    criterion = CrossEntropyLoss(
        class_weights.to(device), label_smoothing=config.label_smoothing
    )
    optimizer = AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.eta_min
    )

    train_losses: list[float] = []
    test_losses: list[float] = []

    train_accuracies: list[float] = []
    test_accuracies: list[float] = []

    # Initialize directories
    exp_path.mkdir(parents=True, exist_ok=True)
    (exp_path / "models").mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(1, config.epochs + 1):
        # Train loss from training and everything else on eval mode
        train_loss, _ = train_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()

        _, train_acc = evaluate_model(model, train_eval_loader, criterion)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion)

        # Log info
        log_line = (
            f"Epoch {epoch:03d}: "
            f"Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, "
            f"Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}"
        )
        with open(exp_path / "log.txt", "a", encoding="utf-8") as f:
            f.write(f"{log_line}\n")

        print(log_line)

        # Save the model at every checkpoint or when we have a new best accuracy
        state_dict_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        # Checkpoint or final epoch
        if epoch % config.checkpoint_freq == 0 or epoch == config.epochs:
            torch.save(state_dict_cpu, exp_path / "models" / f"model_{epoch}.pth")

        # Save for plotting
        train_losses.append(train_loss)  # Train mode
        test_losses.append(test_loss)  # Eval mode
        train_accuracies.append(train_acc)  # Eval mode
        test_accuracies.append(test_acc)  # Eval mode

    plot_training_curves(
        exp_path, train_losses, test_losses, train_accuracies, test_accuracies
    )
