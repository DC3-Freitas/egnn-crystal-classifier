"""
All training utilities and pipeline for training disorder model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from egnn_crystal_classifier.config import DisorderModelConfig
from egnn_crystal_classifier.interpolation_disorder.ml_model.model import DisorderModel
from egnn_crystal_classifier.utils.seed import set_seed


def train_epoch(
    model: DisorderModel,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    criterion: MSELoss,
    optimizer: AdamW,
) -> float:
    """
    Runs a single epoch through the data loader and reports loss on train mode.

    Args:
        model: The model being trained.
        loader: Data loader providing batches of data.
        criterion: Classification loss function (MSE).
        optimizer: Optimizer for the model (AdamW).

    Returns:
        Mean loss over the dataset on train mode.
    """
    model.train()

    total_loss = 0
    pbar = tqdm(loader, desc="Training", leave=False)

    for data, y in pbar:
        optimizer.zero_grad()

        device = next(model.parameters()).device
        data, y = data.to(device), y.to(device)

        out = model(data)

        loss = criterion(torch.sigmoid(out).squeeze(-1), y)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_model(
    model: DisorderModel,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    criterion: MSELoss,
) -> float:
    """
    Evaluates the model on all the data in the given data loader
    and reports loss and accuracy on eval mode.

    Args:
        model: The model being trained.
        loader: Data loader providing batches of data.
        criterion: Classification loss function (MSE).

    Returns:
        Mean loss over the dataset on eval mode.
    """
    model.eval()

    total_loss = 0
    pbar = tqdm(loader, desc="Evaluating", leave=False)

    for data, y in pbar:
        device = next(model.parameters()).device
        data, y = data.to(device), y.to(device)

        with torch.inference_mode():
            out = model(data)

        loss = criterion(torch.sigmoid(out).squeeze(-1), y)
        total_loss += loss.item()

    pbar.close()

    return total_loss / len(loader)


def plot_training_curves(
    exp_path: Path,
    train_losses: list[float],
    test_losses: list[float],
) -> None:
    """
    Plots and saves loss curves.

    Args:
        exp_path: Outer directory of the expirement which is where the
                  figure will be saved.
        train_losses: Losses from each epoch for train set.
        test_losses: Losses from each epoch for test set.
    """
    assert len(train_losses) == len(
        test_losses
    ), "train and test info must have same length"

    epochs = list(range(1, len(train_losses) + 1))

    fig, ax = plt.subplots()

    # Loss curves
    ax.plot(epochs, train_losses, label="Train Loss")
    ax.plot(epochs, test_losses, label="Test Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax.grid(True, which="both", axis="x", linestyle=":", linewidth=0.5)
    ax.legend(loc="best")

    plt.tight_layout()

    out_path = exp_path / "training_curves.png"
    plt.savefig(out_path)
    plt.close(fig)


def train_disorder_model(
    config: DisorderModelConfig,
    exp_path: Path,
    train_loader: DataLoader[tuple[torch.Tensor, ...]],
    test_loader: DataLoader[tuple[torch.Tensor, ...]],
    device: torch.device,
) -> None:
    """
    Runs the full training pipeline given the necessary
    information. Saves all information (checkpoint, logs, and
    graph).

    Args:
        config: Contains all training hyperparameter information.
                See DisorderModelConfig class for more details.
        exp_path: Outer directory for experiments which will contain
                  model checkpoints and logs.
        train_loader: Training set data loader.
        test_loader: Test set data loader.
        device: Device on which the model is trained (e.g. cuda).
    """
    # Prepare info
    set_seed()

    model = DisorderModel(config).to(device)

    criterion = MSELoss()
    optimizer = AdamW(model.parameters(), lr=config.lr)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.eta_min
    )

    train_losses: list[float] = []
    test_losses: list[float] = []

    # Initialize directories
    exp_path.mkdir(parents=True, exist_ok=True)
    (exp_path / "models").mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(1, config.epochs + 1):
        # Train loss from training and everything else on eval mode
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()

        test_loss = evaluate_model(model, test_loader, criterion)

        # Log info
        log_line = (
            f"Epoch {epoch:03d}: "
            f"Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}"
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

    plot_training_curves(exp_path, train_losses, test_losses)
