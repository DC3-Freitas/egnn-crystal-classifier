"""
Class for storing outlier (specifically unknown structure) data.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class OutlierData:
    """
    Information used for unknown structure classification.

    Attributes:
        perfect_embeddings (num_crystals, sz): ideal embeddings for each crystal type

        (num_crystals, sz) numpy array of ideal embeddings for each crystal
        type with index determined on label type.
    """

    perfect_embeddings: npt.NDArray[np.float32]
    delta_cutoffs: npt.NDArray[np.float32]
