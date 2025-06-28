"""
Core class for DC4 inference.
"""

import numpy as np
import torch
import torch_geometric.data
from ovito.data import DataCollection
from scipy.spatial import cKDTree

from egnn_crystal_classifier.ml_model.model import EGNN
from egnn_crystal_classifier.ml_train.hparams import HParams
from egnn_crystal_classifier.data_prep.graph_construction import construct_batched_graph

class DC4:
    def __init__(
        self,
        model: EGNN = None,
        label_map: dict[str, int] = {},
        hparams: HParams = HParams()
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            # Load pretrained model
            model = EGNN(
                num_buckets=hparams.num_buckets,
                num_features=hparams.num_features,
                num_classes=len(label_map),
                hidden_dim=hparams.hidden_dim,
                num_layers=hparams.num_layers,
            )
            model.load_state_dict("egnn_crystal_classifier/ml_model/model_best.pth")
            print("No model provided. I will use my pretrained model.")
        else:
            assert isinstance(model, EGNN), "Model must be an EGNN instance."
            self.model = model

        self.model.to(self.device)
        self.model.eval()
        self.hparams = hparams

        # Mapping
        self.label_to_number = label_map
        self.number_to_label = {v: k for k, v in label_map.items()}

    def calculate(
        self,
        data: DataCollection,
        batch_size: int = 64,
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
            for i in range(0, pos_graphs.shape[0], batch_size):
                batch_graphs = pos_graphs[i:i + batch_size].to(self.device)
                graphs = construct_batched_graph(
                    pos_graphs=batch_graphs,
                    label_ints=None,
                    num_buckets=self.hparams.num_buckets,
                    device=self.device,
                )
                graphs = graphs.to(self.device)

                if i == 0:
                    output = self.model(graphs)
                else:
                    output = torch.cat((output, self.model(graphs)), dim=0)

        predictions = output.argmax(dim=1).cpu().numpy()
        return predictions
    
    def process_for_inference(
        self,
        data: DataCollection
    ) -> torch_geometric.data.Data:
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
        neighbors = tree.query(positions, k=self.hparams.num_buckets + 1)[1][:]

        for i, neigh in enumerate(neighbors):
            pos_graphs.append(np.array([positions[i] for i in neigh]))
            assert i == neigh[0]
        pos_graphs = np.array(pos_graphs)
        pos_graphs = torch.tensor(pos_graphs, dtype=torch.float32, device=self.device)

        return pos_graphs