"""
OVITO modifier for crystal structure classification using the DC4 model.
"""

from pathlib import Path

from ovito.data import DataCollection  # pylint: disable=no-name-in-module
from ovito.pipeline import ModifierInterface  # pylint: disable=no-name-in-module
from traits.api import Bool

from egnn_crystal_classifier.dc4 import DC4


class DC4Modifier(ModifierInterface):
    """
    OVITO modifier for crystal structure classification using DC4.
    Predicts crystal structure types and coherence based on atomic positions
    and updates DataCollection in-place.
    """

    model_path: str
    run = Bool(False, help="Click to start model processing.")

    # pylint: disable=unused-argument
    def modify(self, data: DataCollection, frame: int, **_: object) -> None:
        """
        Modify the DataCollection in-place by predicting crystal structure types
        and calculating coherence using the DC4 model.

        Args:
            data: The data to modify.
            frame: The current frame number.
        """
        model = DC4.from_saved(Path(self.model_path))
        predictions, coherence, __ = model.calculate(data)

        data.particles_.create_property("Particle Type", data=predictions)
        data.particles_.create_property("Coherence", data=coherence)
