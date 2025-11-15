"""
Train a defect classifier. We use the same architecture as the crystal
classifier, but train it on a dataset of defected structures. We will
first generate the requisite data, and store it in /data. Then we will
train the model on this data. The model will use the same EGNN architecture
as the crystal classifier. The model will be stored as /ml_model/defect_model.pth.
ml_train/train.py will not be modified for this purpose, but it will be used.

This script can be run directly, or via the command line using:
python -m egnn_crystal_classifier.scripts_defect
"""

import os
import json
from pathlib import Path
import numpy as np
import torch

import matplotlib.pyplot as plt

from egnn_crystal_classifier.constants import *
from egnn_crystal_classifier.data_gen.gen_defect import (
    generate_defects,
)
from egnn_crystal_classifier.ml_train.hparams import HParams
from egnn_crystal_classifier.ml_train.train import train

""" DATA GENERATION """

def save_data(
    local_path: Path,
    num_vacancies: int,
    num_interstitials: int,
) -> None:
    x_data, y_data, label_map = generate_defects(
        num_vacancies=num_vacancies,
        num_interstitials=num_interstitials,
    )

    print(f"Generated {len(x_data)} defective structures.")

    local_path.mkdir(parents=True, exist_ok=True)
    coords_file = local_path / "coords.npy"
    y_data_file = local_path / "labels.json"
    labels_file = local_path / "label_map.json"
    np.save(coords_file, x_data)
    label_backmap = {v: k for k, v in label_map.items()}
    label_strs = [label_backmap[int(y)] for y in y_data]
    with open(labels_file, "w") as f:
        json.dump(label_map, f)
    with open(y_data_file, "w") as f:
        json.dump(label_strs, f)

    print(f"Saved {len(x_data)} structures to {local_path}")

""" TRAINING """

def train_defect_classifier(
    data_path: Path,
    model_path: Path,
    hparams: HParams,
) -> None:
    train(
        exp_path=model_path / "defect_training",
        coord_path=data_path / "coords.npy",
        label_path=data_path / "labels.json",
        label_map_path=data_path / "label_map.json",
        vol=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        hp=hparams,
    )
    # copy the best model to the specified model_path
    best_model_path = model_path / "defect_training" / "models" / "model_best.pth"
    if best_model_path.exists():
        os.makedirs(model_path, exist_ok=True)
        torch.save(torch.load(best_model_path), model_path / "defect_model.pth")
        # copy the label map as well
        with open(data_path / "label_map.json", "r") as f:
            label_map = json.load(f)
        with open(model_path / "defect_label_map.json", "w") as f:
            json.dump(label_map, f)
    

if __name__ == "__main__":
    BASE_DIR = Path("egnn_crystal_classifier")
    DATA_DIR = BASE_DIR / "data" / "defect_data"
    MODEL_DIR = BASE_DIR / "ml_model"

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Step 1: Generate data
    print("[PART I] Generating defect data ...")
    save_data(
        local_path=DATA_DIR,
        num_vacancies=15,
        num_interstitials=15,
    )

    # Step 2: Train model
    print("[PART II] Training defect classifier ...")
    # hparams = HParams(
    #     num_buckets=100,
    #     num_hidden=64,
    #     num_reg_layers=4,
    #     num_classes=3,  # vacancy, interstitial, none
    #     dropout_prob=0.1,
    #     lr=1e-3,
    #     batch_size=32,
    # )
    
    hparams = HParams(
        num_classes=3,  # vacancy, interstitial, none
        lr=1e-3,
        batch_size=512,
        epochs=20
    )
    train_defect_classifier(
        data_path=DATA_DIR,
        model_path=MODEL_DIR,
        hparams=hparams,
    )
    print(f"Model saved to {MODEL_DIR / 'defect_model.pth'}")