"""
TODO
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class OutlierData:
    """
    TODO
    """

    perfect_embeddings: npt.NDArray[np.float32]
    delta_cutoffs: npt.NDArray[np.float32]
    alpha_cutoff: float = 0.88  # Based on observation
