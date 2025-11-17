"""
DC4 liquid inference for interpolated-lattices dataset.

This mirrors `dc4_liquid.py` but looks for the interpolated-trained
liquid model and label map produced by the `scripts_liquid_interpolated`
training pipeline.
"""

import json
import os
from typing import Tuple

import numpy as np
import torch
from ovito.data import DataCollection
from scipy.spatial import cKDTree

from egnn_crystal_classifier.constants import BASE_DIR
from egnn_crystal_classifier.data_prep.data_handler import CrystalDataset, FastLoader
from egnn_crystal_classifier.data_prep.graph_construction import construct_graph_lists
from egnn_crystal_classifier.ml_model.model import EGNN


# Default paths for interpolated model and label map (may be overridden by user)
INTERP_MODEL_PATH = os.path.join(BASE_DIR, "ml_model", "liquid_interpolated_model.pth")
INTERP_MODEL_BEST_PATH = os.path.join(BASE_DIR, "ml_model", "liquid_interpolated_model_best.pth")


class DC4LiquidInterpolated:
    def __init__(
        self,
        model: EGNN | None = None,
        label_map: dict | None = None,
        model_path: str | None = None,
        label_map_path: str | None = None,
        device: torch.device | None = None,
        confidence_threshold: float = 0.5,
        hparams=None,
    ) -> None:
        """Initialize DC4 inference for interpolated dataset.

        Args:
            model: optional EGNN instance to use (if provided, will not be loaded)
            label_map: optional mapping str->int for labels
            model_path: optional path to a saved model file
            label_map_path: optional path to label map json
            device: torch device to run inference on
            confidence_threshold: unused currently but kept for compatibility
            hparams: optional hyperparameters (used to build model if loading)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold

        # load or use provided model
        if model is None:
            # decide model path: prefer best snapshot if available
            mp = model_path or (INTERP_MODEL_BEST_PATH if os.path.exists(INTERP_MODEL_BEST_PATH) else INTERP_MODEL_PATH)
            if not os.path.exists(mp):
                raise FileNotFoundError(f"Interpolated liquid model not found at {mp}")

            if hparams is None:
                # Minimal default hyperparams expected by EGNN constructor used in this repo
                # We assume the saved state_dict matches the EGNN signature from ml_model.model
                hparams = type("HP", (), {"num_buckets": 24, "num_hidden": 64, "num_reg_layers": 2, "num_classes": 1, "dropout_prob": 0.05})()

            self.model = EGNN(
                num_buckets=getattr(hparams, "num_buckets", 24),
                hidden=getattr(hparams, "num_hidden", 64),
                num_reg_layers=getattr(hparams, "num_reg_layers", 2),
                num_classes=getattr(hparams, "num_classes", 1),
                dropout_prob=getattr(hparams, "dropout_prob", 0.05),
            )
            self.model.load_state_dict(torch.load(mp, map_location=self.device))
            print(f"Loaded interpolated liquid model from {mp}")
        else:
            self.model = model

        self.model.to(self.device)
        self.model.eval()

    def calculate(self, data: DataCollection, return_probabilities: bool = False) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """Run inference on an OVITO DataCollection and return predictions.

        Args:
            data: OVITO DataCollection with `particles.positions` and `cell`
            return_probabilities: whether to return probabilities (liquid prob)

        Returns:
            predictions or (predictions, probabilities)
        """
        neighbors, pos_graphs = construct_graph_lists(
            pos_individual=data.particles.positions,
            num_neighbors=16,
            cell=data.cell[...],
        )

        dataset = CrystalDataset(pos_graphs=pos_graphs, label_strs=None, label_map=None)
        loader = FastLoader(dataset=dataset, batch_size=512, num_buckets=24, calc_device=self.device, shuffle=False)

        with torch.no_grad():
            for i, graphs in enumerate(loader):
                graphs = graphs.to(self.device)
                batch_output, embeddings = self.model(graphs)
                if i == 0:
                    output = batch_output
                    embeddings_list = embeddings
                else:
                    output = torch.cat((output, batch_output), dim=0)
                    embeddings_list = torch.cat((embeddings_list, embeddings), dim=0)

        # probs = torch.softmax(output, dim=1).cpu().numpy()
        # preds = output.argmax(dim=1).cpu().numpy()
        probs = torch.sigmoid(output).squeeze().cpu().numpy()
        preds = (probs > 0.3).astype(np.int64)

        # Local smoothing using neighbor majority (same as DC4Liquid)
        N_SAME = 8
        N_NEIGH = 16
        tree = cKDTree(data.particles.positions)
        for idx in range(len(preds)):
            _, indices = tree.query(data.particles.positions[idx], k=N_NEIGH + 1)
            neighbor_preds = preds[indices[1:]]
            if np.sum(neighbor_preds == preds[idx]) < N_SAME:
                preds[idx] = 1 - preds[idx]

        if return_probabilities:
            return probs * 100
        return preds


__all__ = ["DC4LiquidInterpolated"]
