"""
TODO
"""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from ovito.data import DataCollection  # pylint: disable=no-name-in-module
from tqdm import tqdm

from egnn_crystal_classifier.config import Config
from egnn_crystal_classifier.data_prep.data_reader import raw_positions_to_loader
from egnn_crystal_classifier.data_prep.graph_construction import construct_graph_lists
from egnn_crystal_classifier.ml_model.model import NequIP
from egnn_crystal_classifier.other_structures.coherence import compute_coherence
from egnn_crystal_classifier.other_structures.outlier_data import OutlierData


class DC4:
    """
    TODO
    """

    def __init__(
        self,
        config: Config,
        model: NequIP,
        outlier_data: OutlierData,
    ) -> None:
        """
        TODO

        maybe warn about rep exposure
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        self.model = model
        self.model.to(self.device)
        self.model.eval()

        self.outlier_data = outlier_data

    @classmethod
    def from_saved(cls, path: Path) -> "DC4":
        """
        TODO
        """
        info = torch.load(path, weights_only=False)

        config = Config(**info["config"])

        # Rebuild the model
        model = NequIP(config)
        model.load_state_dict(info["state_dict"])

        outlier_data = OutlierData(**info["outlier_data"])

        return cls(config, model, outlier_data)

    def save_dc4(self, path: Path) -> None:
        """
        Saves entire model (NequIP, config, and outlier info) into single file.

        Args:
            path: Path to save the model to.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        info = {
            "state_dict": self.model.state_dict(),
            "config": asdict(self.config),
            "outlier_data": asdict(self.outlier_data),
        }
        torch.save(info, path)

    def calculate(
        self,
        data: DataCollection,
    ) -> tuple[
        npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]
    ]:
        """
        Performs structural classification on the given data and
        determines local disorder via coherence.

        Args:
            data: The input data collection.

        Returns:
            predictions: Predicted crystal structure types.
            coherence: How different the atom's local structure is from that of its neighbors
            embeddings: Embeddings for each atom.
        """
        loader = raw_positions_to_loader(
            self.config,
            np.array(data.particles.positions, dtype=np.float32),
            np.array(data.cell, dtype=np.float32),
        )

        # Get predictions and embeddings
        output_list: list[int] = []
        embeddings_list: list[torch.Tensor] = []

        for graphs in tqdm(loader, desc="Forward Pass"):
            with torch.inference_mode():
                batch_output, embeddings = self.model(graphs.to(self.device))

            output_list.extend(batch_output.cpu())
            embeddings_list.append(embeddings.cpu())

        predictions = np.array(output_list, dtype=np.float32).argmax(axis=1)
        embeddings_torch = torch.cat(embeddings_list, axis=0)
        embeddings_np = embeddings_torch.numpy().astype(np.float32)

        # Outlier detection
        neighbors, _ = construct_graph_lists(
            np.array(data.particles.positions, dtype=np.float32),
            self.config.num_neighbors,
            np.array(data.cell, dtype=np.float32),
        )
        coherence = compute_coherence(
            self.config, neighbors, embeddings_torch, self.device
        )
        amorphous_mask = coherence <= self.outlier_data.alpha_cutoff

        similarity_to_ref = (
            embeddings_np * self.outlier_data.perfect_embeddings[predictions]
        ).sum(axis=-1)
        unknown_crystal_mask = (
            similarity_to_ref <= self.outlier_data.delta_cutoffs[predictions]
        )

        # Modify predictions

        # predictions[np.where(amorphous_mask == 1)] = self.config.label_map["amorphous"]
        predictions[np.where(unknown_crystal_mask == 1)] = self.config.label_map[
            "unknown_crystal"
        ]

        return predictions, coherence, embeddings_np
