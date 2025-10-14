# Liquid-Solid Classifier Training

This document describes the liquid-solid classifier training script (`scripts_liquid.py`) that extends the EGNN crystal classifier to distinguish between liquid and solid atomic structures.

## Overview

The liquid-solid classifier uses the same EGNN architecture as the original crystal classifier but is trained for binary classification (liquid vs solid). It combines:

1. **Synthetic liquid structures** generated using Poisson disk sampling
2. **Solid crystal structures** from the existing synthetic crystal dataset
3. **Binary classification** training to distinguish liquid from solid phases

## Features

- **Data Generation**: Automatically generates balanced liquid/solid datasets
- **Realistic Physics**: Uses physically meaningful parameters for liquid generation
- **Reuses Existing Architecture**: Same EGNN model as crystal classifier
- **Flexible Configuration**: Customizable hyperparameters for different use cases
- **Evaluation Metrics**: Per-class accuracy reporting for liquid/solid classification

## Usage

### Basic Usage

Run the training script directly:

```bash
python -m egnn_crystal_classifier.scripts_liquid
```

Or execute the script file:

```bash
python src/egnn_crystal_classifier/scripts_liquid.py
```

### Configuration

The script uses `LiquidTrainingHParams` class which extends the base `HParams`:

```python
class LiquidTrainingHParams(HParams):
    def __init__(self):
        super().__init__()
        # Binary classification
        self.num_classes = 2
        
        # Data generation
        self.liquid_structures = 100  # Number of liquid structures
        self.solid_structures = 200   # Number of solid structures to use
        self.liquid_solid_ratio = 0.5  # Target liquid/solid ratio
        
        # Training parameters
        self.epochs = 80
        self.lr = 0.008
        self.batch_size = 512
```

### Custom Training

For custom configurations, modify the hyperparameters:

```python
from egnn_crystal_classifier.scripts_liquid import main, LiquidTrainingHParams

# Custom hyperparameters
hp = LiquidTrainingHParams()
hp.liquid_structures = 200
hp.epochs = 100
hp.lr = 0.01

# Run training (you would need to modify main() to accept hp parameter)
main()
```

## Training Process

### 1. Data Generation
- **Liquid Structures**: Generated using Poisson disk sampling with realistic density and minimum distance constraints
- **Solid Structures**: Extracted from existing synthetic crystal data (BCC, FCC, HCP, etc.)
- **Balancing**: Dataset balanced according to `liquid_solid_ratio` parameter

### 2. Data Processing
- All structures converted to atomic neighborhood graphs
- Each atom represented by its nearest neighbors (default: 16 neighbors)
- Binary labels: "liquid" → 0, "solid" → 1

### 3. Model Training
- Uses EGNN (E(3)-equivariant Graph Neural Network) architecture
- Binary cross-entropy loss with label smoothing
- AdamW optimizer with cosine annealing learning rate schedule
- Gradient clipping for stability

### 4. Model Output
- Trained model saved as `ml_model/liquid_model.pth`
- Label mapping saved as `ml_model/liquid_label_map.json`
- Training logs and checkpoints in `ml_model/liquid_classifier_training/`

## Output Files

After training, the following files are created:

```
ml_model/
├── liquid_model.pth              # Trained model weights
├── liquid_label_map.json         # Label mapping (liquid: 0, solid: 1)
└── liquid_classifier_training/   # Training artifacts
    ├── log.txt                   # Training log
    ├── training_curves.png       # Loss/accuracy plots
    └── models/                   # Model checkpoints
        ├── model_best.pth        # Best model
        ├── model_5.pth           # Checkpoint at epoch 5
        └── ...

data/
├── liquid_coords.npy             # Generated coordinate data
├── liquid_labels.json            # Generated labels
└── liquid_label_map.json         # Label mapping
```

## Performance Metrics

The script automatically evaluates the trained model and reports:

- **Overall Accuracy**: Total correct predictions / total samples
- **Liquid Accuracy**: Correct liquid predictions / total liquid samples  
- **Solid Accuracy**: Correct solid predictions / total solid samples

Example output:
```
Overall Accuracy: 0.9234 (1847/2000)
Liquid Accuracy:  0.9156 (915/1000)
Solid Accuracy:   0.9320 (932/1000)
```

## Testing

Test the training script without running full training:

```bash
python test_liquid_training.py
```

This validates:
- All imports work correctly
- Hyperparameters are configured properly
- Data generation produces valid output
- Script structure is correct

## Integration with Existing Code

The liquid classifier integrates with the existing codebase:

- **Uses same model architecture**: `ml_model/model.py`
- **Uses same training loop**: `ml_train/train.py`
- **Uses same data handling**: `data_prep/data_handler.py`
- **Extends existing patterns**: Similar to `scripts.py`

## Hardware Requirements

- **GPU Recommended**: CUDA-compatible GPU for training acceleration
- **Memory**: ~4GB RAM for default dataset sizes
- **Storage**: ~500MB for generated data and model files
- **Time**: ~30-60 minutes for default training (80 epochs)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   hp.batch_size = 256  # Reduce batch size
   ```

2. **Too Few Liquid Structures Generated**
   ```python
   hp.liquid_structures = 50  # Reduce if generation is slow
   ```

3. **Training Takes Too Long**
   ```python
   hp.epochs = 50  # Reduce epochs
   ```

4. **Import Errors**
   - Ensure all dependencies are installed
   - Check that the src/ directory is in your Python path

### Debug Mode

For debugging, you can run individual components:

```python
from egnn_crystal_classifier.scripts_liquid import generate_liquid_solid_dataset, LiquidTrainingHParams

hp = LiquidTrainingHParams()
hp.liquid_structures = 5  # Small test
x_data, y_data, label_map = generate_liquid_solid_dataset(hp, Path("debug_data"))
```

## Dependencies

Required packages:
- `torch` (PyTorch)
- `torch_geometric` (Graph neural networks)
- `numpy` (Numerical computation)
- `scipy` (Scientific computing)
- `matplotlib` (Plotting)
- `ovito` (Structure file handling)

Install with:
```bash
pip install torch torch_geometric numpy scipy matplotlib ovito
```
