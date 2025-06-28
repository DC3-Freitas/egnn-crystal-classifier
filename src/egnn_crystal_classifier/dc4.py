"""
Core class for DC4 inference.
"""

import numpy as np
import torch
import torch_geometric.data
from ovito.data import DataCollection
from scipy.spatial import cKDTree
import json

from egnn_crystal_classifier.ml_model.model import EGNN
from egnn_crystal_classifier.ml_train.hparams import HParams
from egnn_crystal_classifier.data_prep.graph_construction import construct_batched_graph


class DC4:
    def __init__(
        self,
        model: EGNN = None,
        label_map: dict[str, int] = None,
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
                dropout_prob=0.0,
            )
            self.model.load_state_dict(
                torch.load("egnn_crystal_classifier/ml_model/model_best.pth")
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
                (open("egnn_crystal_classifier/ml_model/label_map.json", "r").read())
            )
            print("No label map provided, using defaults:", label_map)
        self.label_to_number = label_map
        self.number_to_label = {v: k for k, v in label_map.items()}

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

        pos_graphs = self.process_for_inference(data)
        with torch.no_grad():
            for i in range(0, pos_graphs.shape[0], self.hparams.batch_size):
                batch_graphs = pos_graphs[i : i + self.hparams.batch_size].to(
                    self.device
                )
                graphs = construct_batched_graph(
                    pos_graphs=batch_graphs,
                    label_ints=None,
                    num_buckets=self.hparams.num_buckets,
                    calc_device=self.device,
                )
                graphs = graphs.to(self.device)

                batch_output, embeddings = self.model(graphs)
                if i == 0:
                    output = batch_output
                else:
                    output = torch.cat((output, batch_output), dim=0)

        predictions = output.argmax(dim=1).cpu().numpy()
        return predictions

    def process_for_inference(self, data: DataCollection) -> torch_geometric.data.Data:
        """
        Process OVITO datacollection into individual position graphs
        in preparation for inference.

        Args:
            data (DataCollection): The input data collection.

        Returns:
            torch.Tensor: A tensor containing the position graphs.
        """

        positions = data.particles.positions

        pos_graphs = []
        tree = cKDTree(positions)
        neighbors = tree.query(positions, k=self.hparams.nn_count + 1)[1][:]

        for i, neigh in enumerate(neighbors):
            pos_graphs.append(np.array([positions[j] for j in neigh]))
            assert i == neigh[0]
        pos_graphs = np.array(pos_graphs)
        pos_graphs = torch.tensor(pos_graphs, dtype=torch.float32, device=self.device)

        return pos_graphs
