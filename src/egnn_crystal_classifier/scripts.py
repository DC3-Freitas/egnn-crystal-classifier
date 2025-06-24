import json
from pathlib import Path

import modal
import numpy as np
from modal import Volume

from egnn_crystal_classifier.ml_model.model import EGNN
from egnn_crystal_classifier.ml_train.hparams import HParams
from egnn_crystal_classifier.ml_train.train import train

app = modal.App()

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch", "torch_geometric", "tqdm", "ovito", "matplotlib", "numpy", "scipy"
    )
    .pip_install("torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv")
).add_local_python_source("egnn_crystal_classifier")

vol = modal.Volume.from_name("egnn", create_if_missing=True)


def save_data(
    local_path: Path,
    modal_path: Path | None,
    vol: Volume,
) -> None:
    from egnn_crystal_classifier.data_gen.gen import gen

    x_data, y_data, label_map = gen()

    local_path.mkdir(parents=True, exist_ok=True)
    coords_file = local_path / "coords.npy"
    labels_file = local_path / "labels.json"
    label_map_file = local_path / "label_map.json"

    np.save(coords_file, x_data)
    labels_file.write_text(json.dumps(y_data), encoding="utf-8")
    label_map_file.write_text(json.dumps(label_map), encoding="utf-8")

    if modal_path is not None:
        remote_base = modal_path.as_posix()
        with vol.batch_upload(force=True) as batch:
            for local_file in (coords_file, labels_file, label_map_file):
                remote_file = f"{remote_base}/{local_file.name}"
                batch.put_file(str(local_file), remote_file)


@app.function(gpu="A100", image=image, volumes={"/root/egnn": vol}, timeout=18000)
def run_train(
    exp_path: str,
    coord_path: str,
    label_path: str,
    label_map_path: str,
    vol: Volume | None,
    device: str,
    hp: HParams,
) -> None:

    train(
        Path(exp_path),
        Path(coord_path),
        Path(label_path),
        Path(label_map_path),
        vol,
        device,
        hp,
    )


def main() -> None:
    """
    Save data to local and modal with this function
    """
    # save_data(Path("outputs/data"), Path("data"), vol)
    """
    Run training with this function
    """
    hp = HParams()
    with app.run():
        run_train.remote(
            ("/root/egnn/dropout_05_epochs_100_updated_logging"),
            ("/root/egnn/data/coords.npy"),
            ("/root/egnn/data/labels.json"),
            ("/root/egnn/data/label_map.json"),
            vol,
            "cuda",
            hp,
        )


if __name__ == "__main__":
    main()
