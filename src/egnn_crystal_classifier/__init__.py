"""
OVITO modifier for crystal structure classification using the DC4 model.
"""

from pathlib import Path

from ovito.data import DataCollection  # pylint: disable=no-name-in-module
from ovito.pipeline import ModifierInterface  # pylint: disable=no-name-in-module
from traits.api import Bool

from egnn_crystal_classifier.dc4 import DC4


# pylint: disable=too-few-public-methods
class DC4Modifier(ModifierInterface):
    """
    OVITO modifier for crystal structure classification using DC4.
    Predicts crystal structure types and disorder based on atomic positions
    and updates DataCollection in-place.

    Attributes:
        model_path: Path of the DC4 model (should be the one including
                    all components from the state dicts to the outlier information).
        disorder_cutoff: Cutoff for amorphous classification (e.g. any
                         disorder >= disorder_cutoff) is classified as amorphous.
                         None if no cutoff should be set.
        run: Whether to run the DC4 model.
    """

    model_path: str
    disorder_cutoff: float | None

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

        # The "__" is so that it doesn't overlap with the parameter **_
        disorder, predictions, __ = model.calculate_all(data, self.disorder_cutoff)

        data.particles_.create_property("structure", data=predictions)
        data.particles_.create_property("disorder", data=disorder)
