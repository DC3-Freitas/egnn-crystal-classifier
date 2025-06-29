"""
OVITO modifier for crystal structure classification using the DC4 model.
"""

from egnn_crystal_classifier.dc4 import DC4
from egnn_crystal_classifier.ml_train.hparams import HParams
from ovito.data import DataCollection
from ovito.pipeline import ModifierInterface
from traits.api import Any, Bool, Float


class DC4Modifier(ModifierInterface):
    """
    OVITO modifier for crystal structure classification using the DC4 equivariant
    graph neural network model. Predicts crystal structure types based on atomic positions
    and updates DataCollection in-place. Amorphous and unknown crystal structures
    will be implemented in the future.
    """

    model_info = Any()
    run = Bool(False, help="Click to start model processing.")
    coherence_cutoff = Any(help="Coherence cutoff for amorphous structure detection.")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def modify(self, data: DataCollection, frame: int, **kwargs) -> None:
        """
        Modify the DataCollection in-place by predicting crystal structure types
        using the DC4 model.

        Args:
            data (DataCollection): The data to modify.
            frame (int): The current frame number.
            **kwargs: Additional keyword arguments (not used).
        """
        if not self.run:
            return
        assert isinstance(
            self.model_info, (str, type(None))
        ), "Model must be a string path to the model or None for default model."
        assert isinstance(
            self.coherence_cutoff, (float, type(None))
        ), "Coherence cutoff must be a float or None."
        if isinstance(self.model_info, str):
            self.model = DC4(
                model_path=self.model_info,
            )
        else:
            self.model = DC4(
                coherence_cutoff=self.coherence_cutoff,
            )
        outputs = self.model.calculate(data)
        data.particles_.create_property("Particle Type", data=outputs)
