"""
OVITO modifier for crystal structure classification using the DC4 model.
"""

from egnn_crystal_classifier.dc4 import DC4
from egnn_crystal_classifier.ml_train.hparams import HParams
from ovito.data import DataCollection
from ovito.pipeline import ModifierInterface
from traits.api import Any, Bool

class DC4Modifier(ModifierInterface):
    """
    OVITO modifier for crystal structure classification using the DC4 equivariant
    graph neural network model. Predicts crystal structure types based on atomic positions
    and updates DataCollection in-place. Amorphous and unknown crystal structures
    will be implemented in the future.
    """

    model: Any
    run: Bool = (False, {"desc": "Click to start model processing."})
    device: str = "cpu"
    label_map: dict[str, int] | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(self.model, (str, type(None))), (
            "Model must be a string path to the model or None for default model."
        )
        if isinstance(self.model, str):
            self.model = DC4(
                model_path=self.model,
                label_map=self.label_map,
                hparams=HParams(),
            )
        else:
            self.model = DC4()
    
    def modify(self, data: DataCollection, frame: int, **kwargs) -> None:
        """
        Modify the DataCollection in-place by predicting crystal structure types
        using the DC4 model.

        Args:
            data (DataCollection): The data to modify.
            frame (int): The current frame number.
            **kwargs: Additional keyword arguments (not used).
        """
        outputs = self.model.calculate(data)
        data.particles_.create_property("Particle Type", data=outputs)