import json
import os
from pathlib import Path

import modal
import numpy as np
import torch
from ovito.io import export_file, import_file

from egnn_crystal_classifier.config import Config
from egnn_crystal_classifier.data_prep.data_reader import get_loaders_for_training
from egnn_crystal_classifier.dc4 import DC4
from egnn_crystal_classifier.ml_model.model import NequIP
from egnn_crystal_classifier.ml_train.train import train
from egnn_crystal_classifier.other_structures.embedding_cutoffs import (
    compute_cutoff,
    compute_perfect_embeddings,
)
from egnn_crystal_classifier.other_structures.outlier_data import OutlierData

BASE_DIR = Path(os.path.abspath(os.path.dirname(__file__)))

SYNTH_DATA_PATH = BASE_DIR / "resources" / "synthetic_data"
EXP_PATH = BASE_DIR / "resources" / "training_smoothing"
NEQUIP_PATH = BASE_DIR / "resources" / "training_smoothing" / "models" / "model_20.pth"
PERFECT_LATTICES_PATH = BASE_DIR / "resources" / "perfect_lattices"
DC4_PATH = BASE_DIR / "resources" / "complete_models" / "dc4_smoothing.ckpt"


def main() -> None:
    config = Config()
    device = "cuda"

    """
    train_loader, train_eval_loader, test_loader, class_weights = (
        get_loaders_for_training(config, SYNTH_DATA_PATH)
    )
    train(
        config,
        EXP_PATH,
        train_loader,
        train_eval_loader,
        test_loader,
        class_weights,
        torch.device("cuda"),
    )
    return
    """

    dc4 = DC4.from_saved(DC4_PATH)
    # print(dc4.outlier_data)
    # print(dc4.config)
    # return
    pipeline = import_file(BASE_DIR / "dump.r_1100K_1.lammpstrj")
    data = pipeline.compute(300)
    pred, coh, _ = dc4.calculate(data)

    data.particles_.create_property("structure", data=pred)
    data.particles_.create_property("coherence", data=coh)

    # -----------------------------------
    # 4. Save only this frame
    # -----------------------------------
    export_file(
        data,
        "dump.r_1100K_1.lammpstrj_two.gz",
        format="lammps/dump",
        columns=[
            "Particle Identifier",
            "Position.X",
            "Position.Y",
            "Position.Z",
            "structure",
            "coherence",
        ],
    )
    return

    """
    model = NequIP(config)
    model.load_state_dict(torch.load(NEQUIP_PATH))
    perfect = compute_perfect_embeddings(
        config,
        model,
        PERFECT_LATTICES_PATH,
        device,
    )
    cutoffs = compute_cutoff(config, model, perfect, SYNTH_DATA_PATH, device)
    print(cutoffs)
    dc4 = DC4(config, model, OutlierData(perfect, cutoffs))
    dc4.save_dc4(DC4_PATH)
    """
    return


if __name__ == "__main__":
    main()

"""
Desired:
Epoch 001: Train Loss = 0.7355, Test Loss = 0.1593, Train Acc = 0.9671, Test Acc = 0.9657
Epoch 002: Train Loss = 0.1240, Test Loss = 0.0546, Train Acc = 0.9841, Test Acc = 0.9830
Epoch 003: Train Loss = 0.0652, Test Loss = 0.0315, Train Acc = 0.9918, Test Acc = 0.9902
Epoch 004: Train Loss = 0.0502, Test Loss = 0.0234, Train Acc = 0.9927, Test Acc = 0.9917
Epoch 005: Train Loss = 0.0426, Test Loss = 0.0200, Train Acc = 0.9930, Test Acc = 0.9914
Epoch 006: Train Loss = 0.0364, Test Loss = 0.0206, Train Acc = 0.9917, Test Acc = 0.9898
Epoch 007: Train Loss = 0.0329, Test Loss = 0.0131, Train Acc = 0.9968, Test Acc = 0.9954
Epoch 008: Train Loss = 0.0299, Test Loss = 0.0111, Train Acc = 0.9971, Test Acc = 0.9957
Epoch 009: Train Loss = 0.0275, Test Loss = 0.0102, Train Acc = 0.9977, Test Acc = 0.9964
Epoch 010: Train Loss = 0.0253, Test Loss = 0.0120, Train Acc = 0.9952, Test Acc = 0.9939
Epoch 011: Train Loss = 0.0239, Test Loss = 0.0084, Train Acc = 0.9978, Test Acc = 0.9965
Epoch 012: Train Loss = 0.0229, Test Loss = 0.0082, Train Acc = 0.9980, Test Acc = 0.9964
Epoch 013: Train Loss = 0.0216, Test Loss = 0.0078, Train Acc = 0.9981, Test Acc = 0.9972
Epoch 014: Train Loss = 0.0203, Test Loss = 0.0075, Train Acc = 0.9980, Test Acc = 0.9964
Epoch 015: Train Loss = 0.0198, Test Loss = 0.0077, Train Acc = 0.9981, Test Acc = 0.9972
Epoch 016: Train Loss = 0.0188, Test Loss = 0.0073, Train Acc = 0.9981, Test Acc = 0.9970
Epoch 017: Train Loss = 0.0185, Test Loss = 0.0069, Train Acc = 0.9984, Test Acc = 0.9974
Epoch 018: Train Loss = 0.0177, Test Loss = 0.0066, Train Acc = 0.9985, Test Acc = 0.9973
Epoch 019: Train Loss = 0.0177, Test Loss = 0.0066, Train Acc = 0.9987, Test Acc = 0.9972
Epoch 020: Train Loss = 0.0175, Test Loss = 0.0065, Train Acc = 0.9985, Test Acc = 0.9972
"""
