"""
Core class for DC4 inference.
"""

import numpy as np
import torch
import torch_geometric.data
from ovito.data import DataCollection
from scipy.spatial import cKDTree
import json
import os

from egnn_crystal_classifier.ml_model.model import EGNN
from egnn_crystal_classifier.ml_train.hparams import HParams
from egnn_crystal_classifier.data_prep.graph_construction import construct_graph_lists
from egnn_crystal_classifier.data_prep.data_handler import CrystalDataset, FastLoader
from egnn_crystal_classifier.amorphous.coherence import get_amorphous_mask


class DC4:
    def __init__(
        self,
        model: EGNN = None,
        label_map: dict[str, int] = None,
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
        base_dir = os.path.abspath(os.path.dirname(__file__))
        if model is None:
            # Load pretrained model
            self.model = EGNN(
                num_buckets=hparams.num_buckets,
                hidden=hparams.num_hidden,
                num_reg_layers=hparams.num_reg_layers,
                num_classes=hparams.num_classes,
                dropout_prob=0.0,
            )
            self.model.load_state_dict(
                torch.load(base_dir + "/ml_model/model_best.pth")
            )
            print("No model provided. I will use my pretrained model.")
        else:
            assert isinstance(model, EGNN), "Model must be an EGNN instance."
            self.model = model

        self.model.to(self.device)
        self.model.eval()
        self.hparams = hparams

        # Mapping
        if label_map is None:
            label_map = json.loads(
                (open(base_dir + "/ml_model/label_map.json", "r").read())
            )
            print("No label map provided, using defaults:", label_map)

        # Inject additional labels
        label_map = label_map.copy()
        label_map["amorphous"] = len(label_map)
        label_map["unknown"] = len(label_map) + 1

        self.label_to_number = label_map
        self.number_to_label = {v: k for k, v in label_map.items()}

        self.coherence_cutoff = coherence_cutoff

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
            num_neighbors=self.hparams.nn_count,
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
        amorphous_mask = get_amorphous_mask(
            neighbors_raw=neighbors,
            embeddings=embeddings_list,
            batch_size=self.hparams.batch_size,
            calc_device=self.device,
            cutoff=self.coherence_cutoff,
        )
        predictions[np.where(amorphous_mask == 1)] = self.label_to_number["amorphous"]
        return predictions
