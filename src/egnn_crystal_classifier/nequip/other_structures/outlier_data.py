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
        perfect_embeddings (num_crystals, sz): Ideal embeddings for each crystal type

        delta_cutoffs (num_crystals,): Distance cutoffs for each crystal type (i.e. if distance
                                       is >= cutoff then it is unknown).
    """

    perfect_embeddings: npt.NDArray[np.float32]
    delta_cutoffs: npt.NDArray[np.float32]
