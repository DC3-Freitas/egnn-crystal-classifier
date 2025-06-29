"""
Core class for DC4 inference.
"""

import json
import os

import numpy as np
import torch
from ovito.data import DataCollection

from egnn_crystal_classifier.amorphous.coherence import get_amorphous_mask
from egnn_crystal_classifier.amorphous.outlier import get_outlier_mask
from egnn_crystal_classifier.constants import *
from egnn_crystal_classifier.data_prep.data_handler import CrystalDataset, FastLoader
from egnn_crystal_classifier.data_prep.graph_construction import (
    construct_batched_graph,
    construct_graph_lists,
)
from egnn_crystal_classifier.ml_model.model import EGNN
from egnn_crystal_classifier.ml_train.hparams import HParams


class DC4:
    def __init__(
        self,
        model: EGNN = None,
        label_map: dict[str, int] = None,
        run_amorphous: bool = True,
        coherence_cutoff: float | None = None,
        hparams: HParams = HParams(),
    ) -> None:
        """
        Initialize DC4 inference class. Loads pretrained model and
        preset labelmap if not provided. Auto-detects device (CPU or GPU).

        Args:
            model (EGNN, optional): Pretrained EGNN model. Defaults to None.
            label_map (dict[str, int], optional): Mapping of labels to integers.
                Defaults to None, which uses a preset label map.
            hparams (HParams, optional): Hyperparameters for the model.
                Defaults to HParams() with preset values.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            # Load pretrained model
            self.model = EGNN(
                num_buckets=hparams.num_buckets,
                hidden=hparams.num_hidden,
                num_reg_layers=hparams.num_reg_layers,
                num_classes=hparams.num_classes,
                dropout_prob=hparams.dropout_prob,
            )
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            print("No model provided. I will use my pretrained model.")
        else:
            assert isinstance(model, EGNN), "Model must be an EGNN instance."
            self.model = model

        self.model.to(self.device)
        self.model.eval()
        self.hparams = hparams

        # Mapping
        if label_map is None:
            label_map = json.loads((open(LABEL_MAP_PATH).read()))
            print("No label map provided, using defaults:", label_map)

        # Inject additional labels
        label_map = label_map.copy()
        label_map["amorphous"] = len(label_map)
        label_map["unknown"] = len(label_map) + 1

        self.label_to_number = label_map
        self.number_to_label = {v: k for k, v in label_map.items()}

        self.coherence_cutoff = coherence_cutoff
        self.run_amorphous = run_amorphous

        # TODO make these paths configurable
        self.perfect_embeddings = np.load(PERFECT_EMBEDDINGS_PATH)
        self.perfect_embeddings = torch.from_numpy(self.perfect_embeddings).to(
            self.device
        )
        self.delta_cutoffs = np.load(DELTA_CUTOFFS_PATH)
        self.delta_cutoffs = torch.from_numpy(self.delta_cutoffs).to(self.device)

    def calculate(
        self,
        data: DataCollection,
    ) -> np.ndarray:
        """
        Calculate the crystal structure types for the given data.

        Args:
            data (DataCollection): The input data collection.
            hparams (HParams): Hyperparameters for the model.

        Returns:
            np.ndarray: Predicted crystal structure types.
        """

        neighbors, pos_graphs = construct_graph_lists(
            pos_individual=data.particles.positions,
            num_neighbors=self.hparams.num_neighbors,
        )
        dataset = CrystalDataset(
            pos_graphs=pos_graphs,
            label_strs=None,
            label_map=self.label_to_number,
        )
        loader = FastLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_buckets=self.hparams.num_buckets,
            calc_device=self.device,
            shuffle=False,
        )
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

        predictions = output.argmax(dim=1).cpu().numpy()
        if self.run_amorphous:
            amorphous_mask = get_amorphous_mask(
                neighbors_raw=neighbors,
                embeddings=embeddings_list,
                batch_size=self.hparams.batch_size,
                calc_device=self.device,
                cutoff=self.coherence_cutoff,
            )
            outlier_mask = get_outlier_mask(
                embeddings=embeddings_list,
                predictions=predictions,
                ref_embeddings=self.perfect_embeddings,
                delta_cutoffs=self.delta_cutoffs,
            )
            predictions[np.where(amorphous_mask == 1)] = self.label_to_number[
                "amorphous"
            ]
            predictions[np.where(outlier_mask == 1)] = self.label_to_number["unknown"]
        return predictions
