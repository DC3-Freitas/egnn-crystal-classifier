"""
Core class for DC4 liquid inference.
Performs liquid-solid classification using the trained liquid classifier.
"""

import json
import os

import numpy as np
import torch
from ovito.data import DataCollection

from egnn_crystal_classifier.constants import *
from egnn_crystal_classifier.data_prep.data_handler import CrystalDataset, FastLoader
from egnn_crystal_classifier.data_prep.graph_construction import (
    construct_batched_graph,
    construct_graph_lists,
)
from egnn_crystal_classifier.ml_model.model import EGNN
from egnn_crystal_classifier.ml_train.hparams import HParams


class LiquidTrainingHParams(HParams):
    """Hyperparameters for liquid-solid classification."""
    
    def __init__(self):
        super().__init__()
        # Binary classification (liquid vs solid)
        self.num_classes = 2


class DC4Liquid:
    def __init__(
        self,
        model: EGNN = None,
        label_map: dict[str, int] = None,
        confidence_threshold: float = 0.5,
        hparams: LiquidTrainingHParams = None,
    ) -> None:
        """
        Initialize DC4 liquid inference class. Loads pretrained liquid model and
        preset labelmap if not provided. Auto-detects device (CPU or GPU).

        Args:
            model (EGNN, optional): Pretrained EGNN model. Defaults to None.
            label_map (dict[str, int], optional): Mapping of labels to integers.
                Defaults to None, which uses the preset liquid label map.
            confidence_threshold (float): Minimum confidence for liquid classification.
                Defaults to 0.5.
            hparams (LiquidTrainingHParams, optional): Hyperparameters for the model.
                Defaults to LiquidTrainingHParams() with preset values.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        
        # Set up hyperparameters
        if hparams is None:
            hparams = LiquidTrainingHParams()
        self.hparams = hparams
        
        if model is None:
            # Load pretrained liquid model
            if not os.path.exists(LIQUID_MODEL_PATH):
                raise FileNotFoundError(
                    f"Liquid model not found at {LIQUID_MODEL_PATH}. "
                    "Please train the liquid model first using scripts_liquid.py"
                )
            
            self.model = EGNN(
                num_buckets=hparams.num_buckets,
                hidden=hparams.num_hidden,
                num_reg_layers=hparams.num_reg_layers,
                num_classes=hparams.num_classes,  # Binary classification
                dropout_prob=hparams.dropout_prob,
            )
            self.model.load_state_dict(torch.load(LIQUID_MODEL_PATH, map_location=self.device))
            print(f"Loaded pretrained liquid model from {LIQUID_MODEL_PATH}")
        else:
            assert isinstance(model, EGNN), "Model must be an EGNN instance."
            self.model = model
            print("Using provided model for liquid inference.")

        self.model.to(self.device)
        self.model.eval()

        # Load label mapping
        if label_map is None:
            if not os.path.exists(LIQUID_LABEL_MAP_PATH):
                raise FileNotFoundError(
                    f"Liquid label map not found at {LIQUID_LABEL_MAP_PATH}. "
                    "Please train the liquid model first using scripts_liquid.py"
                )
            
            with open(LIQUID_LABEL_MAP_PATH, 'r') as f:
                label_map = json.load(f)
            print("Using liquid label map:", label_map)

        # Add uncertainty label for low-confidence predictions
        label_map = label_map.copy()
        label_map["uncertain"] = len(label_map)

        self.label_to_number = label_map
        self.number_to_label = {v: k for k, v in label_map.items()}

    def calculate(
        self,
        data: DataCollection,
        return_probabilities: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Calculate the liquid-solid classification for the given data.

        Args:
            data (DataCollection): The input data collection from OVITO.
            return_probabilities (bool): Whether to return prediction probabilities.
                Defaults to False.

        Returns:
            np.ndarray: Predicted liquid-solid types (0=liquid, 1=solid, 2=uncertain).
            tuple[np.ndarray, np.ndarray]: If return_probabilities=True, returns 
                (predictions, probabilities) where probabilities is N x 2 array.
        """

        # Construct graphs from atomic positions
        neighbors, pos_graphs = construct_graph_lists(
            pos_individual=data.particles.positions,
            num_neighbors=self.hparams.num_neighbors,
            cell=data.cell[...]
        )
        
        # Create dataset and loader
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
        
        # Run inference
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

        # Convert logits to probabilities
        probabilities = torch.softmax(output, dim=1).cpu().numpy()
        
        # Get predictions
        predictions = output.argmax(dim=1).cpu().numpy()
        # predictions = probabilities[:, 1] * 100
        
        # Apply confidence threshold - mark low-confidence predictions as uncertain
        # max_probs = np.max(probabilities, axis=1)
        # uncertain_mask = max_probs < self.confidence_threshold
        # predictions[uncertain_mask] = self.label_to_number["uncertain"]
        
        if return_probabilities:
            return probabilities[:, 1] * 100
        else:
            return predictions

    def calculate_with_confidence(
        self,
        data: DataCollection,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate liquid-solid classification with confidence scores and uncertainties.

        Args:
            data (DataCollection): The input data collection from OVITO.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: 
                - predictions: Predicted labels (0=liquid, 1=solid, 2=uncertain)
                - confidence_scores: Maximum probability for each prediction
                - liquid_probabilities: Probability of being liquid for each atom
        """
        predictions, probabilities = self.calculate(data, return_probabilities=True)
        
        # Calculate confidence scores (maximum probability)
        confidence_scores = np.max(probabilities, axis=1)
        
        # Extract liquid probabilities (class 0)
        liquid_probabilities = probabilities[:, 0]
        
        return predictions, confidence_scores, liquid_probabilities

    def get_label_name(self, label_number: int) -> str:
        """
        Convert label number to label name.
        
        Args:
            label_number (int): Numerical label
            
        Returns:
            str: String label name
        """
        return self.number_to_label.get(label_number, "unknown")
    
    def get_label_number(self, label_name: str) -> int:
        """
        Convert label name to label number.
        
        Args:
            label_name (str): String label name
            
        Returns:
            int: Numerical label
        """
        return self.label_to_number.get(label_name, -1)
    
    def summary_statistics(self, predictions: np.ndarray) -> dict:
        """
        Calculate summary statistics for predictions.
        
        Args:
            predictions (np.ndarray): Array of predictions
            
        Returns:
            dict: Dictionary with statistics
        """
        unique, counts = np.unique(predictions, return_counts=True)
        total = len(predictions)
        
        stats = {
            "total_atoms": total,
            "liquid_count": 0,
            "solid_count": 0,
            "uncertain_count": 0,
            "liquid_fraction": 0.0,
            "solid_fraction": 0.0,
            "uncertain_fraction": 0.0
        }
        
        for label_num, count in zip(unique, counts):
            label_name = self.get_label_name(label_num)
            if label_name == "liquid":
                stats["liquid_count"] = count
                stats["liquid_fraction"] = count / total
            elif label_name == "solid":
                stats["solid_count"] = count
                stats["solid_fraction"] = count / total
            elif label_name == "uncertain":
                stats["uncertain_count"] = count
                stats["uncertain_fraction"] = count / total
        
        return stats


# Backward compatibility alias
DC4 = DC4Liquid
