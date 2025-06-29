import json
import os
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from ovito.data import DataCollection
from ovito.io import import_file
from tqdm import tqdm

from egnn_crystal_classifier.constants import *
from egnn_crystal_classifier.data_gen.gen import gen
from egnn_crystal_classifier.data_prep.data_handler import CrystalDataset, FastLoader
from egnn_crystal_classifier.data_prep.graph_construction import construct_graph_lists
from egnn_crystal_classifier.ml_model.model import EGNN
from egnn_crystal_classifier.ml_train.hparams import HParams


def compute_perfect_embeddings(
    structure: DataCollection,
    model: EGNN,
    batch_size: int,
    calc_device: torch.device,
    hparams: HParams = HParams(),
):
    """
    Computes the perfect embeddings for the given positions using the provided model.

    Args:
        pos_graphs (NDArray): Array of atomic positions in the crystal structure.
        model (EGNN): The machine learning model to compute embeddings.
        batch_size (int): Batch size for processing.
        calc_device (torch.device): Device for computation.
        hparams (HParams): Hyperparameters for the model, including nn_count and embedding_dim.

    Returns:
        torch.Tensor: Embeddings for each atom in the crystal structure.
    """
    _, pos_graphs = construct_graph_lists(
        structure.particles.positions, num_neighbors=hparams.num_neighbors
    )
    dataset = CrystalDataset(
        pos_graphs=pos_graphs,
        label_strs=None,
        label_map=None,
    )
    loader = FastLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_buckets=hparams.num_buckets,
        calc_device=calc_device,
        shuffle=False,
    )
    embeddings_avg = torch.zeros(hparams.num_hidden, device=calc_device)
    with torch.no_grad():
        for graphs in loader:
            graphs = graphs.to(calc_device)
            _, embeddings = model(graphs)
            embeddings_avg += embeddings.mean(dim=0)
    embeddings_avg /= len(loader)
    return embeddings_avg


def compute_delta_cutoffs(
    perfect_embeddings: NDArray[np.floating[Any]],
    x_data: NDArray[np.number[Any]],
    y_data: list[str],
    label_map: dict[str, int] | None,
    model: EGNN,
    hparams: HParams,
    calc_device: torch.device,
) -> NDArray[np.floating[Any]]:
    """
    Computes the 99-percentile cutoff between lattices in training data and perfect lattices.
    Used for determining the delta cutoffs for outlier detection.

    Args:
        perfect_embeddings (torch.Tensor): The perfect embeddings for the crystal structure.
        x_data: Input features for each crystal structure.
        y_data: Labels for each crystal structure.
        label_map (dict[str, int] | None): Mapping from label strings to integers.
        model (EGNN): The machine learning model to compute embeddings.
        batch_size (int): Batch size for processing.
        calc_device (torch.device): Device for computation.

    Returns:
        NDArray[np.floating[Any]]: The delta cutoffs for each crystal structure.
    """

    dataset = CrystalDataset(
        pos_graphs=x_data,
        label_strs=y_data,
        label_map=label_map,
    )
    loader = FastLoader(
        dataset=dataset,
        batch_size=hparams.batch_size,
        num_buckets=hparams.num_buckets,
        calc_device=calc_device,
        shuffle=False,
    )
    distances = [[] for _ in range(len(perfect_embeddings))]
    with torch.no_grad():
        for i, graphs in tqdm(enumerate(loader)):
            graphs = graphs.to(calc_device)
            _, embeddings = model(graphs)
            class_label_nums = graphs.y.cpu().numpy()
            for j in range(embeddings.shape[0]):
                class_label_num = class_label_nums[j]
                distances[class_label_num].append(
                    torch.norm(
                        embeddings[j] - perfect_embeddings[class_label_num]
                    ).item()
                )
    delta_cutoffs = np.zeros(len(perfect_embeddings))
    for i, dist_list in enumerate(distances):
        if len(dist_list) > 0:
            delta_cutoffs[i] = np.percentile(np.array(dist_list), 99)
        else:
            delta_cutoffs[i] = 0.0
    return delta_cutoffs


def run_outlier_precompute(
    model_path: str = MODEL_PATH,
    label_map_path: str = LABEL_MAP_PATH,
) -> None:
    """
    Precomputes the perfect embeddings and delta cutoffs for outlier detection.
    Stores the results in `perfect_embeddings.npy` and `delta_cutoffs.npy`.
    """

    # TODO Should fix gen to have customization options,
    # so we can disable noisy data generation that may interfere

    x_data, y_data, label_map = gen()
    calc_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hparams = HParams()
    model = EGNN(
        num_buckets=hparams.num_buckets,
        hidden=hparams.num_hidden,
        num_reg_layers=hparams.num_reg_layers,
        num_classes=hparams.num_classes,
        dropout_prob=0.0,
    )

    label_map = json.loads(open(label_map_path, "r").read())
    model.load_state_dict(torch.load(model_path, map_location=calc_device))
    model.to(calc_device)
    model.eval()

    perfect_structure_names = os.listdir(PERFECT_LATTICES_PATH)
    perfect_embeddings = [0] * len(perfect_structure_names)
    for structure_name in perfect_structure_names:
        structure_path = os.path.join(PERFECT_LATTICES_PATH, structure_name)
        structure = import_file(structure_path).compute()
        perfect_embeddings[label_map[structure_name.split(".")[0]]] = (
            compute_perfect_embeddings(
                structure, model, hparams.batch_size, calc_device, hparams
            )
        )

    perfect_embeddings = torch.stack(perfect_embeddings).to(calc_device)
    print("Computed perfect embeddings for", len(perfect_embeddings), "structures.")
    delta_cutoffs = compute_delta_cutoffs(
        perfect_embeddings,
        x_data,
        y_data,
        label_map,
        model,
        hparams,
        calc_device,
    )
    np.save(PERFECT_EMBEDDINGS_PATH, perfect_embeddings.cpu().numpy())
    np.save(DELTA_CUTOFFS_PATH, delta_cutoffs)
    print("Computed cutoffs:", delta_cutoffs)


if __name__ == "__main__":
    run_outlier_precompute()
