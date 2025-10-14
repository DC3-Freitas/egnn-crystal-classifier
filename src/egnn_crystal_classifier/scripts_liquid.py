"""
Train a liquid-solid classifier. We use the same architecture as the crystal
classifier, but train it on a dataset of liquid and solid structures. We will
first generate the requisite data, and store it in /data. Then we will
train the model on this data. The model will use the same EGNN architecture
as the crystal classifier. The model will be stored as /ml_model/liquid_model.pth.
ml_train/train.py will not be modified for this purpose, but it will be used.

This script can be run directly, or via the command line using:
python -m egnn_crystal_classifier.scripts_liquid
"""

import json
import os
from pathlib import Path
from typing import Tuple, List, Dict, Any
import numpy as np
import torch

from egnn_crystal_classifier.data_gen.gen_liquid import gen as gen_liquid
from egnn_crystal_classifier.data_gen.gen import gen as gen_solid
from egnn_crystal_classifier.ml_train.train import train
from egnn_crystal_classifier.ml_train.hparams import HParams
from egnn_crystal_classifier.constants import BASE_DIR


class LiquidTrainingHParams(HParams):
    """Extended hyperparameters for liquid-solid classification."""
    
    def __init__(self):
        super().__init__()
        # Override for binary classification (liquid vs solid)
        self.num_classes = 2
        
        # Liquid-specific parameters
        self.liquid_structures = 50  # Number of liquid structures to generate
        self.solid_structures = 200   # Number of solid structures to use
        self.liquid_solid_ratio = 0.5  # Target ratio of liquid to solid samples
        
        # Training adjustments for binary classification
        self.epochs = 20  # Slightly fewer epochs for binary task
        self.lr = 0.008   # Slightly lower learning rate
        self.batch_size = 512  # Smaller batches for better convergence


def generate_liquid_solid_dataset(hp: LiquidTrainingHParams, 
                                data_dir: Path) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """
    Generate a combined dataset of liquid and solid structures.
    
    Args:
        hp: Hyperparameters containing generation settings
        data_dir: Directory to save the generated data
        
    Returns:
        Tuple of (x_data, y_data, label_map)
    """
    print("Generating liquid-solid classification dataset...")
    print("=" * 50)
    
    # Generate liquid structures
    print(f"1. Generating {hp.liquid_structures} liquid structures...")
    x_liquid, y_liquid, _ = gen_liquid(
        n_structures=hp.liquid_structures,
        use_realistic_params=True
    )
    print(f"   Generated {len(x_liquid)} liquid atomic environments")
    
    # Generate solid structures  
    print(f"2. Generating solid structures...")
    x_solid, y_solid, _ = gen_solid(use_checker=True, max_temp=0.1)
    x_liquid_crystal, _, _ = gen_solid(use_checker=False, min_temp=0.2)
    x_liquid = np.vstack([x_liquid, x_liquid_crystal])
    y_liquid = y_liquid + ["liquid"] * len(x_liquid_crystal)
    
    # Take a subset of solid structures to balance the dataset
    total_solid_samples = len(x_solid)
    target_solid_samples = int(len(x_liquid) / hp.liquid_solid_ratio - len(x_liquid))
    target_solid_samples = min(target_solid_samples, total_solid_samples)
    
    if target_solid_samples < total_solid_samples:
        print(f"   Using {target_solid_samples} out of {total_solid_samples} solid samples for balance")
        solid_indices = np.random.choice(total_solid_samples, target_solid_samples, replace=False)
        x_solid = x_solid[solid_indices]
        y_solid = [y_solid[i] for i in solid_indices]
    
    print(f"   Using {len(x_solid)} solid atomic environments")
    
    # Combine datasets
    print("3. Combining liquid and solid datasets...")
    
    # Convert all solid labels to "solid"
    y_solid_binary = ["solid"] * len(y_solid)
    
    # Combine data
    x_combined = np.vstack([x_liquid, x_solid])
    y_combined = y_liquid + y_solid_binary
    
    # Create binary label map
    binary_label_map = {"liquid": 0, "solid": 1}
    
    print(f"   Combined dataset: {len(x_combined)} samples")
    print(f"   Liquid samples: {len(x_liquid)} ({len(x_liquid)/len(x_combined)*100:.1f}%)")
    print(f"   Solid samples: {len(x_solid)} ({len(x_solid)/len(x_combined)*100:.1f}%)")
    
    # Shuffle the combined dataset
    print("4. Shuffling combined dataset...")
    shuffle_indices = np.random.permutation(len(x_combined))
    x_combined = x_combined[shuffle_indices]
    y_combined = [y_combined[i] for i in shuffle_indices]
    
    return x_combined, y_combined, binary_label_map


def save_liquid_solid_data(x_data: np.ndarray, 
                          y_data: List[str], 
                          label_map: Dict[str, int],
                          data_dir: Path) -> Tuple[Path, Path, Path]:
    """
    Save the liquid-solid dataset to disk.
    
    Args:
        x_data: Coordinate data
        y_data: Labels
        label_map: Label mapping
        data_dir: Directory to save data
        
    Returns:
        Tuple of (coords_path, labels_path, label_map_path)
    """
    print("5. Saving dataset to disk...")
    
    # Create data directory
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # File paths
    coords_path = data_dir / "liquid_coords.npy"
    labels_path = data_dir / "liquid_labels.json"
    label_map_path = data_dir / "liquid_label_map.json"
    
    # Save files
    np.save(coords_path, x_data)
    
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(y_data, f, indent=2)
    
    with open(label_map_path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, indent=2)
    
    print(f"   Saved coordinates: {coords_path}")
    print(f"   Saved labels: {labels_path}")
    print(f"   Saved label map: {label_map_path}")
    print(f"   Dataset size: {x_data.nbytes / 1024 / 1024:.1f} MB")
    
    return coords_path, labels_path, label_map_path


def train_liquid_solid_classifier(coords_path: Path,
                                 labels_path: Path, 
                                 label_map_path: Path,
                                 hp: LiquidTrainingHParams,
                                 device: torch.device) -> None:
    """
    Train the liquid-solid classifier.
    
    Args:
        coords_path: Path to coordinate data
        labels_path: Path to labels
        label_map_path: Path to label map
        hp: Hyperparameters
        device: Training device
    """
    print("\nTraining liquid-solid classifier...")
    print("=" * 50)
    
    # Set up output directory
    model_dir = Path(BASE_DIR) / "ml_model"
    exp_path = model_dir / "liquid_classifier_training"
    
    # Train the model
    train(
        exp_path=exp_path,
        coord_path=coords_path,
        label_path=labels_path,
        label_map_path=label_map_path,
        vol=None,  # No modal volume for local training
        device=device,
        hp=hp
    )
    
    # Copy the best model to the expected location
    best_model_path = exp_path / "models" / "model_best.pth"
    target_model_path = model_dir / "liquid_model.pth"
    
    if best_model_path.exists():
        import shutil
        shutil.copy2(best_model_path, target_model_path)
        print(f"\n✅ Model saved to: {target_model_path}")
        
        # Also save the label map to the model directory
        target_label_map = model_dir / "liquid_label_map.json"
        shutil.copy2(label_map_path, target_label_map)
        print(f"✅ Label map saved to: {target_label_map}")
    else:
        print(f"❌ Best model not found at: {best_model_path}")


def evaluate_model_performance(coords_path: Path,
                              labels_path: Path,
                              label_map_path: Path,
                              model_path: Path,
                              device: torch.device) -> None:
    """
    Evaluate the trained model and print performance statistics.
    
    Args:
        coords_path: Path to coordinate data
        labels_path: Path to labels
        label_map_path: Path to label map  
        model_path: Path to trained model
        device: Device for evaluation
    """
    try:
        from egnn_crystal_classifier.ml_model.model import EGNN
        from egnn_crystal_classifier.data_prep.data_handler import CrystalDataset, FastLoader
        
        print("\nEvaluating model performance...")
        print("=" * 30)
        
        # Load data
        pos_graphs = np.load(coords_path)
        with open(labels_path, 'r') as f:
            label_strs = json.load(f)
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
        
        # Create test dataset (use last 20% of data)
        test_size = int(0.2 * len(pos_graphs))
        test_indices = np.arange(len(pos_graphs) - test_size, len(pos_graphs))
        
        test_dataset = CrystalDataset(
            pos_graphs[test_indices],
            [label_strs[i] for i in test_indices],
            label_map
        )
        
        test_loader = FastLoader(
            test_dataset, batch_size=512, num_buckets=24, 
            calc_device=device, shuffle=False
        )
        
        # Load model
        model = EGNN(24, 64, 2, 2, 0.05).to(device)  # Binary classification
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Evaluate
        total_correct = 0
        total_samples = 0
        liquid_correct = 0
        liquid_total = 0
        solid_correct = 0
        solid_total = 0
        
        with torch.no_grad():
            for data in test_loader:
                logits, _ = model(data.to(device))
                pred = logits.argmax(dim=1)
                
                # Overall accuracy
                correct = (pred == data.y).sum().item()
                total_correct += correct
                total_samples += len(pred)
                
                # Per-class accuracy
                for i in range(len(pred)):
                    true_label = data.y[i].item()
                    pred_label = pred[i].item()
                    
                    if true_label == 0:  # liquid
                        liquid_total += 1
                        if pred_label == 0:
                            liquid_correct += 1
                    else:  # solid
                        solid_total += 1
                        if pred_label == 1:
                            solid_correct += 1
        
        # Print results
        overall_acc = total_correct / total_samples
        liquid_acc = liquid_correct / liquid_total if liquid_total > 0 else 0
        solid_acc = solid_correct / solid_total if solid_total > 0 else 0
        
        print(f"Overall Accuracy: {overall_acc:.4f} ({total_correct}/{total_samples})")
        print(f"Liquid Accuracy:  {liquid_acc:.4f} ({liquid_correct}/{liquid_total})")
        print(f"Solid Accuracy:   {solid_acc:.4f} ({solid_correct}/{solid_total})")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")


def main() -> None:
    """Main training script for liquid-solid classifier."""
    print("EGNN Liquid-Solid Classifier Training")
    print("=" * 60)
    
    # Set up
    np.random.seed(42)  # For reproducibility
    torch.manual_seed(42)
    
    # Configuration
    hp = LiquidTrainingHParams()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(BASE_DIR) / "data"
    
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Hyperparameters: {hp}")
    print()
    
    try:
        # Generate dataset
        x_data, y_data, label_map = generate_liquid_solid_dataset(hp, data_dir)
        
        # Save dataset
        coords_path, labels_path, label_map_path = save_liquid_solid_data(
            x_data, y_data, label_map, data_dir
        )
        
        # Train model
        train_liquid_solid_classifier(
            coords_path, labels_path, label_map_path, hp, device
        )
        
        # Evaluate model
        model_path = Path(BASE_DIR) / "ml_model" / "liquid_model.pth"
        if model_path.exists():
            evaluate_model_performance(
                coords_path, labels_path, label_map_path, model_path, device
            )
        
        print("\n🎉 Training completed successfully!")
        print(f"Model saved at: {model_path}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
