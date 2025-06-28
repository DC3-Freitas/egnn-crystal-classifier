# e3gnn-crystal-classifier
![image](https://github.com/user-attachments/assets/890dc8bb-6e30-4a09-a014-171f26ee5136)
E(3)-equivariant graph convolutional network for crystal structure prediction.

## Usage
If installed in OVITO, `DC4 Classification` will appear in the Python subsection of the Modifiers panel. The module can also be used directly in Python:
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

## Installation
OVITO Pro Integrated Python Interpreter
```bash
ovitos -m pip install --user https://github.com/DC3-Freitas/egnn-crystal-classifier/archive/refs/heads/main.zip --find-links https://data.pyg.org/whl/torch-2.7.0+cu126.html --extra-index-url https://download.pytorch.org/whl/cu126 --prefer-binary --only-binary=:all:
```
Python
```bash
pip install https://github.com/DC3-Freitas/egnn-crystal-classifier/archive/refs/heads/main.zip --find-links https://data.pyg.org/whl/torch-2.7.0+cu126.html --extra-index-url https://download.pytorch.org/whl/cu126 --prefer-binary --only-binary=:all:
```