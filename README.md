# e3gnn-crystal-classifier
![image](https://github.com/user-attachments/assets/890dc8bb-6e30-4a09-a014-171f26ee5136)

E(3)-equivariant graph neural network for crystal structure prediction. Our model achieves state-of-the-art performance especially at high temperatures (T≈T_m), and runs up to 10x faster than previous data-driven methods. It is trained on a synthetic dataset, eliminating the need for large datasets of real-world structures. By default, it can classify six types of crystal structures: BCC, FCC, HCP, simple cubic, hexagonal diamond, and diamond cubic. 

## Usage
If installed in OVITO, `DC4 Classification` will appear in the Python subsection of the Modifiers panel. The module can also be used directly in a standard Python environment:
```python
from ovito.io import import_file
from egnn_crystal_classifier.dc4 import DC4

dc4_model = DC4()
pipeline = import_file("path/to/simulation/file")
data = pipeline.compute()
inferences = dc4_model.calculate(data)
```
To add DC4 to an OVITO scripting pipeline, use the DC4Modifier interface:
```python
dc4_modifier = DC4Modifier()
modifier.run = True              # Run must be set to true to activate inference
modifier.model_input = None      # Use default pretrained model

pipeline = import_file("path/to/simulation/file")
pipeline.modifiers.append(dc4_modifier)
```
The model can be trained with `python -m egnn_crystal_classifier.scripts`. This will attempt to start a training run on Modal.
To run training locally, `egnn_crystal_classifier.ml_train.train` can be used directly:
```python
from egnn_crystal_classifier.ml_train.train import train
from egnn_crystal_classifier.ml_train.hparams import HParams

train(
    exp_path=Path("path/to/experiment"),
    coord_path=Path("path/to/position/graphs"),
    label_path=Path("path/to/labels"),
    label_map_path=Path("path/to/label_map.json"),
    vol=None,           # Optional, can be a Volume object
    device="cuda:0",    # or "cpu"
    hp=HParams(),       # Hyperparameters for training
)
```

## Installation
OVITO Pro Integrated Python Interpreter
```bash
ovitos -m pip install --user https://github.com/DC3-Freitas/egnn-crystal-classifier/archive/refs/heads/main.zip --find-links https://data.pyg.org/whl/torch-2.7.0+cu126.html --extra-index-url https://download.pytorch.org/whl/cu126 --prefer-binary --only-binary=:all:
```
Python
```bash
pip install https://github.com/DC3-Freitas/egnn-crystal-classifier/archive/refs/heads/main.zip --find-links https://data.pyg.org/whl/torch-2.7.0+cu126.html --extra-index-url https://download.pytorch.org/whl/cu126 --prefer-binary --only-binary=:all:
```