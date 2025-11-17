"""
OVITO modifier for crystal structure classification using the DC4 model.
"""

from egnn_crystal_classifier.dc4 import DC4
from egnn_crystal_classifier.dc4_liquid import DC4Liquid
from egnn_crystal_classifier.dc4_defect import DC4Defect
from egnn_crystal_classifier.dc4_liquid_interpolated import DC4LiquidInterpolated
from egnn_crystal_classifier.ml_train.hparams import HParams
from ovito.data import DataCollection
from ovito.pipeline import ModifierInterface
from traits.api import Any, Bool, Float, Enum


class DC4Modifier(ModifierInterface):
    """
    OVITO modifier for crystal structure classification using the DC4 equivariant
    graph neural network model. Predicts crystal structure types based on atomic positions
    and updates DataCollection in-place. Amorphous and unknown crystal structures
    will be implemented in the future.
    """

    model_info = Any()
    run = Bool(False, help="Click to start model processing.")
    run_amorphous_outlier = Bool(True, help="Run amorphous and outlier detection")
    display_probability = Bool(
        False, help="Display prediction probabilities instead of class labels."
    )
    coherence_cutoff = Any(help="Coherence cutoff for amorphous structure detection.")

    model_select = Enum(
        "default", "liquid", "defect", "liquid_interp", help="Select which pretrained model to use."
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.model_setup = None

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
        if self.model is None or self.model_setup != (self.model_info, self.coherence_cutoff, self.model_select):
            self.model_setup = (self.model_info, self.coherence_cutoff, self.model_select)
            if isinstance(self.model_info, str):
                self.model = DC4(
                    model_path=self.model_info,
                    coherence_cutoff=self.coherence_cutoff,
                    run_amorphous=self.run_amorphous_outlier,
                )
            else:
                if self.model_select == "default":
                    self.model = DC4(
                        coherence_cutoff=self.coherence_cutoff,
                        run_amorphous=self.run_amorphous_outlier,
                    )
                elif self.model_select == "liquid":
                    self.model = DC4Liquid()
                elif self.model_select == "defect":
                    self.model = DC4Defect()
                elif self.model_select == "liquid_interp":
                    self.model = DC4LiquidInterpolated()
        outputs = self.model.calculate(data, return_probabilities=self.display_probability)
        data.particles_.create_property("Particle Type", data=outputs)
