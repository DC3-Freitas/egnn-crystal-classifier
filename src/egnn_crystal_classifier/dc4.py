"""
Core DC4 implementation for classification. Combines features of
both models to assign a disorder score as well as perform structure classification.
"""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from ovito.data import DataCollection  # pylint: disable=no-name-in-module
from tqdm import tqdm

from egnn_crystal_classifier.config import ConfigAll, DisorderModelConfig, NequIPConfig
from egnn_crystal_classifier.interpolation_disorder.data_prep.data_manip import (
    calculate_sph_inputs,
)
from egnn_crystal_classifier.interpolation_disorder.ml_model.model import DisorderModel
from egnn_crystal_classifier.nequip.data_prep.data_reader import raw_positions_to_loader
from egnn_crystal_classifier.nequip.ml_model.model import NequIP
from egnn_crystal_classifier.nequip.other_structures.outlier_data import OutlierData


class DC4:
    """
    Core DC4 model that supports assigning disorder score and
    performing structural classification.

    Also contains own logic to save and load entire DC4 model
    as one file which includes the configurations (which span both
    the disorder model and nequip), models, and outlier data.
    """

    def __init__(
        self,
        config_all: ConfigAll,
        disorder_model: DisorderModel,
        nequip: NequIP,
        outlier_data: OutlierData,
    ) -> None:
        """
        Initializes all parts of the DC4 model. No gaurentees
        that the arguments will be copied so be careful of rep exposure.

        Args:
            config_all: Configurations spanning both the disorder model configs
                        and nequip configs.
            disorder_model: The model for predicting disorder.
            nequip: The model for predicting structures.
            outlier_data: Information used for unknown structure classification.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config_all = config_all

        self.disorder_model = disorder_model.to(self.device).eval()
        self.nequip = nequip.to(self.device).eval()

        self.outlier_data = outlier_data

    @classmethod
    def from_saved(cls, path: Path) -> "DC4":
        """
        Loads a fully-specified (NequIP, disorder model, config, and outlier info)
        DC4 model from path.

        Args:
            path: Path to a DC4 checkpoint file (e.g. *.ckpt).

        Returns:
            A DC4 instance equivalent to the one that produced the saved model.
        """
        info = torch.load(path, weights_only=False)

        config_dict = info["config"]
        config = ConfigAll(
            disorder_model_config=DisorderModelConfig(
                **config_dict["disorder_model_config"]
            ),
            nequip_config=NequIPConfig(**config_dict["nequip_config"]),
        )

        # Rebuild the models
        disorder_model = DisorderModel(config.disorder_model_config)
        disorder_model.load_state_dict(info["disorder_model_state_dict"])

        nequip = NequIP(config.nequip_config)
        nequip.load_state_dict(info["nequip_state_dict"])

        outlier_data = OutlierData(**info["outlier_data"])

        return cls(config, disorder_model, nequip, outlier_data)

    def save_dc4(self, path: Path) -> None:
        """
        Saves entire model (NequIP, disorder model, config, and outlier info)
        into single file.

        Args:
            path: Path to save the model to (e.g. *.ckpt).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        info = {
            "nequip_state_dict": self.nequip.state_dict(),
            "disorder_model_state_dict": self.disorder_model.state_dict(),
            "config": asdict(self.config_all),
            "outlier_data": asdict(self.outlier_data),
        }
        torch.save(info, path)

    def calculate_disorder(self, data: DataCollection) -> NDArray[np.float32]:
        """
        Computes a per-atom disorder score which is equal to the interpolation parameter
        (i.e. 0 being perfect lattice and 1 being basically random).

        This can be used for amorphous classification.

        Args:
            data: The input data collection.

        Returns:
            Disorder (N,) for each atom.
        """
        disorder_batch_size = self.config_all.disorder_model_config.batch_size
        embeddings, neighbors = calculate_sph_inputs(
            self.config_all.disorder_model_config,
            np.array(data.particles.positions, dtype=np.float32),
            np.array(data.cell, dtype=np.float32),
            self.device,
        )

        # 1) Pass through encoder
        encoded_list: list[torch.Tensor] = []

        for i in tqdm(
            range(0, len(embeddings), disorder_batch_size), desc="Encoding for Disorder"
        ):
            with torch.inference_mode():
                out = self.disorder_model.inference_encode(
                    embeddings[i : i + disorder_batch_size].to(self.device)
                ).cpu()
            encoded_list.append(out)
        encoded = torch.concat(encoded_list)

        # 2) Use encoded to predict features
        disorder_np = np.zeros((len(encoded),), dtype=np.float32)

        for i in tqdm(
            range(0, len(encoded), disorder_batch_size), "Predicting Disorder"
        ):
            neigh_batch = encoded[neighbors[i : i + disorder_batch_size]]
            with torch.inference_mode():
                out = torch.sigmoid(
                    self.disorder_model.predict_from_neigh(neigh_batch.to(self.device))
                )
            disorder_np[i : i + disorder_batch_size] = out.squeeze(1).cpu().numpy()

        return disorder_np

    def calculate_nequip_info(
        self, data: DataCollection
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Uses the NequIP model to predict raw structures and provide invariant
        embeddings. Does not include amorphous or unknown structure predictions.

        Args:
            data: The input data collection.

        Returns:
            predictions (N,): Predicted crystal structure types.
            embeddings (N, sz): Embeddings for each atom.
        """
        loader = raw_positions_to_loader(
            self.config_all.nequip_config,
            np.array(data.particles.positions, dtype=np.float32),
            np.array(data.cell, dtype=np.float32),
        )

        output_list: list[int] = []
        embeddings_list: list[torch.Tensor] = []

        for graphs in tqdm(loader, desc="Forward Pass NequIP"):
            with torch.inference_mode():
                batch_output, embeddings = self.nequip(graphs.to(self.device))

            output_list.extend(batch_output.cpu())
            embeddings_list.append(embeddings.cpu())

        predictions = np.array(output_list, dtype=np.float32).argmax(axis=1)
        embeddings_torch = torch.cat(embeddings_list)
        embeddings_np = embeddings_torch.numpy().astype(np.float32)

        return predictions, embeddings_np

    def calculate_all(
        self,
        data: DataCollection,
        disorder_cutoff: float | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """
        Performs structural classification on the given data and
        determines local disorder via coherence.

        Args:
            data: The input data collection.

        Returns:
            disorder (N,): Interpolation parameter (i.e. how disordered and close to random
                           atom's local structure is).
            predictions (N,): Predicted crystal structure types.
            embeddings (N, sz): Embeddings for each atom.
        """
        disorder = self.calculate_disorder(data)
        predictions, embeddings = self.calculate_nequip_info(data)

        # Adjust nequip with outlier info
        similarity_to_ref = (
            embeddings * self.outlier_data.perfect_embeddings[predictions]
        ).sum(axis=-1)
        unknown_crystal_mask = (
            similarity_to_ref <= self.outlier_data.delta_cutoffs[predictions]
        )
        if disorder_cutoff is not None:
            predictions[np.where(disorder >= disorder_cutoff)] = (
                self.config_all.nequip_config.label_map["amorphous"]
            )
        predictions[np.where(unknown_crystal_mask == 1)] = (
            self.config_all.nequip_config.label_map["unknown_crystal"]
        )
        return disorder, predictions, embeddings
