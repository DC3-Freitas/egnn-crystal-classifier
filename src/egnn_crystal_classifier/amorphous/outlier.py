"""
UNTESTED CODE
"""

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray


def get_outlier_mask(
    embeddings: torch.Tensor,
    predictions: NDArray[np.number[Any]],
    ref_embeddings: torch.Tensor,
    delta_cutoffs: NDArray[np.floating[Any]],
) -> NDArray[np.bool_]:
    """
    Determines which atoms are outliers based on a threshold.

    Args:
        embeddings (torch.Tensor): Tensor of ML embeddings for each atom.
        predictions (NDArray[np.number[Any]]): Array of predicted labels for each atom.
        ref_embeddings (torch.Tensor): Reference embeddings for each predicted label.
        delta_cutoffs (NDArray[np.floating[Any]]): Array of delta cutoffs for each predicted label.

    Returns:
        NDArray[np.bool_]: Boolean array indicating whether each atom is an outlier (True) or not (False).
    """

    return (
        torch.norm(embeddings - ref_embeddings[predictions])
        > delta_cutoffs[predictions]
    )
