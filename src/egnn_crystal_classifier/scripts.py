# pylint: disable=unused-import

"""
Utility scripts for the complete DC4 model pipeline.
Used to test and save the output of components of the pipeline.
"""

import os
from pathlib import Path
from typing import Callable

import torch
from ovito.data import DataCollection  # pylint: disable=no-name-in-module
from ovito.io import export_file, import_file  # pylint: disable=no-name-in-module

from egnn_crystal_classifier.config import ConfigAll
from egnn_crystal_classifier.dc4 import DC4
from egnn_crystal_classifier.interpolation_disorder.data_prep.data_manip import (
    get_disorder_loaders_for_training,
)
from egnn_crystal_classifier.interpolation_disorder.ml_model.model import DisorderModel
from egnn_crystal_classifier.interpolation_disorder.ml_train.train import (
    train_disorder_model,
)
from egnn_crystal_classifier.nequip.data_prep.data_reader import (
    get_nequip_loaders_for_training,
)
from egnn_crystal_classifier.nequip.ml_model.model import NequIP
from egnn_crystal_classifier.nequip.ml_train.train import train_nequip
from egnn_crystal_classifier.nequip.other_structures.embedding_cutoffs import (
    compute_cutoff,
    compute_perfect_embeddings,
)
from egnn_crystal_classifier.nequip.other_structures.outlier_data import OutlierData

BASE_DIR = Path(os.path.abspath(os.path.dirname(__file__)))

SYNTH_DATA_DISORDER_PATH = BASE_DIR / "resources" / "interpolated_lattices"
SYNTH_DATA_NEQUIP_PATH = BASE_DIR / "resources" / "synthetic_data"

EXP_DISORDER_PATH = BASE_DIR / "resources" / "train_disorder_model"
EXP_NEQUIP_PATH = BASE_DIR / "resources" / "train_nequip"

DISORDER_MODEL_PATH = (
    BASE_DIR / "resources" / "train_disorder_model" / "models" / "model_20.pth"
)
NEQUIP_PATH = BASE_DIR / "resources" / "train_nequip" / "models" / "model_20.pth"

PERFECT_LATTICES_PATH = BASE_DIR / "resources" / "perfect_lattices"

DC4_PATH = BASE_DIR / "resources" / "complete_models" / "dc4.ckpt"


def run_disorder_train(config_all: ConfigAll, device: torch.device) -> None:
    """
    Trains the disorder model.

    Args:
        config_all: Global configuration containing disorder model settings.
        device: Device used for calculations.
    """
    train_loader, test_loader = get_disorder_loaders_for_training(
        config_all.disorder_model_config, SYNTH_DATA_DISORDER_PATH, device
    )
    train_disorder_model(
        config_all.disorder_model_config,
        EXP_DISORDER_PATH,
        train_loader,
        test_loader,
        device,
    )


def run_nequip_train(config_all: ConfigAll, device: torch.device) -> None:
    """
    Trains the NequIP model.

    Args:
        config_all: Global configuration containing NequIP settings.
        device: Device used for computations.
    """
    train_loader, train_eval_loader, test_loader, class_weights = (
        get_nequip_loaders_for_training(
            config_all.nequip_config, SYNTH_DATA_NEQUIP_PATH
        )
    )
    train_nequip(
        config_all.nequip_config,
        EXP_NEQUIP_PATH,
        train_loader,
        train_eval_loader,
        test_loader,
        class_weights,
        device,
    )


def save_dc4_from_models(config_all: ConfigAll, device: torch.device) -> None:
    """
    Bundles configs, disorder model, NequIP, and outlier info
    into a single DC4 checkpoint.

    Args:
        config_all: Global configuration spanning both models.
        device: Device used for computations.
    """
    disorder_model = DisorderModel(config_all.disorder_model_config)
    disorder_model.load_state_dict(torch.load(DISORDER_MODEL_PATH))

    nequip = NequIP(config_all.nequip_config)
    nequip.load_state_dict(torch.load(NEQUIP_PATH))

    perfect = compute_perfect_embeddings(
        config_all.nequip_config,
        nequip,
        PERFECT_LATTICES_PATH,
        device,
    )
    cutoffs = compute_cutoff(
        config_all.nequip_config, nequip, perfect, SYNTH_DATA_NEQUIP_PATH, device
    )

    dc4 = DC4(config_all, disorder_model, nequip, OutlierData(perfect, cutoffs))
    dc4.save_dc4(DC4_PATH)


def get_modifier(
    disorder_cutoff: float | None = None,
) -> Callable[[int, DataCollection], None]:
    """
    Creates an OVITO modifier that runs DC4 inference per frame.

    Args:
        disorder_cutoff: Optional disorder threshold for amorphous
                         classification.
    """
    dc4 = DC4.from_saved(DC4_PATH)

    def modify(frame: int, data: DataCollection) -> None:
        """
        Runs DC4 inference on a single OVITO frame and
        attaches results to the pipeline.

        Args:
            frame: Frame being evaluated.
            data: OVITO data collection for the current frame.
        """
        print(f"Running model on frame: {frame}")
        disorder, predictions, _ = dc4.calculate_all(data, disorder_cutoff)

        data.particles_.create_property(
            name="disorder",
            data=disorder,
        )

        data.particles_.create_property(
            name="structure",
            data=predictions,
        )

    return modify


def main() -> None:
    """
    Place to put code to run parts of the pipeline.

    Example code that runs the entire pipeline:

    config_all = ConfigAll()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_disorder_train(config_all, device)
    run_nequip_train(config_all, device)
    save_dc4_from_models(config_all, device)

    Example code that annotates a structure (without explicitely labelling amorphous):



    """
    # Testing code

    pipeline = import_file(BASE_DIR / "resources" / "copper_structures.gz")
    pipeline.modifiers.append(get_modifier())

    export_file(
        pipeline,
        BASE_DIR / "resources" / "copper_structures.gz",
        format="lammps/dump",
        columns=[
            "Particle Identifier",
            "Position.X",
            "Position.Y",
            "Position.Z",
            "disorder",
            "structure",
        ],
    )


if __name__ == "__main__":
    main()
