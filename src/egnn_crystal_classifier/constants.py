import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# DATA GENERATION
SYNTH_DATA_PATH = os.path.join(BASE_DIR, "synthetic_data")
NN_COUNT = 16
CHECKERBOARD_CELL_SIZE_XZ = 4
CHECKERBOARD_CELL_SIZE_Y = 3
CHECKERBOARD_CELL_COUNT = 3

# INFERENCE
MODEL_PATH = os.path.join(BASE_DIR, "ml_model", "model_best.pth")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "ml_model", "label_map.json")

# OUTLIER DETECTION
PERFECT_LATTICES_PATH = os.path.join(BASE_DIR, "amorphous", "perfect_lattices")
PERFECT_EMBEDDINGS_PATH = os.path.join(BASE_DIR, "amorphous", "perfect_embeddings.npy")
DELTA_CUTOFFS_PATH = os.path.join(BASE_DIR, "amorphous", "delta_cutoffs.npy")
