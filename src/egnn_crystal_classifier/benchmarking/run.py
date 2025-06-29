import os
import time

import numpy as np
import pandas as pd
from DC3.dc3 import DC3
from DC3.dc3 import create_model as create_dc3_model
from ovito.io import import_file

from egnn_crystal_classifier.benchmarking.heuristics import compute_heuristic_accuracy
from egnn_crystal_classifier.dc4 import DC4

SIM_TEMPERATURE_FRACTIONS = np.round(np.arange(0.04, 1.60 + 0.04, 0.04), 6)
EXP_TO_CLASSIFIERS = {
    "al_fcc": [
        "DC4",
        "DC3",
        "Common Neighbor Analysis (Non-Diamond)",
        "Interval Common Neighbor Analysis",
        "Ackland-Jones Analysis",
    ],
    "li_bcc": [
        "DC4",
        "DC3",
        "Common Neighbor Analysis (Non-Diamond)",
        "Interval Common Neighbor Analysis",
        "Ackland-Jones Analysis",
        "VoroTop Analysis",
    ],
    "ti_hcp": [
        "DC4",
        "DC3",
        "Common Neighbor Analysis (Non-Diamond)",
        "Interval Common Neighbor Analysis",
        "Ackland-Jones Analysis",
    ],
    "ge_cd": ["DC4", "DC3", "Common Neighbor Analysis (Diamond)", "Chill+"],
}

dc4_model = DC4()
dc3_model = create_dc3_model(None)


def apply_dcmodel(data_path: str, model) -> np.ndarray:
    pipeline = import_file(data_path)
    all_outputs = []
    for frame in range(pipeline.source.num_frames):
        data = pipeline.compute(frame)
        outputs = model.calculate(data)
        all_outputs.append(outputs)
    return np.concatenate(all_outputs)


def run_benchmark(exp_name: str) -> None:
    """
    Run the benchmark for the given experiment name and data path.

    Args:
        exp_name (str): The name of the experiment.
    """
    classifiers = EXP_TO_CLASSIFIERS.get(exp_name, [])

    CORRECT_MAP_DC4 = {
        "al_fcc": dc4_model.label_to_number["fcc"],
        "li_bcc": dc4_model.label_to_number["bcc"],
        "ti_hcp": dc4_model.label_to_number["hcp"],
        "ge_cd": dc4_model.label_to_number["cd"],
    }
    CORRECT_MAP_DC3 = {"al_fcc": "fcc", "li_bcc": "bcc", "ti_hcp": "hcp", "ge_cd": "cd"}

    inference_results = [[] for _ in range(len(SIM_TEMPERATURE_FRACTIONS))]
    time_results = [[] for _ in range(len(SIM_TEMPERATURE_FRACTIONS))]

    for test_file in os.listdir(f"egnn_crystal_classifier/benchmarking/md/{exp_name}"):
        data_path = f"egnn_crystal_classifier/benchmarking/md/{exp_name}/{test_file}"
        if "relaxed" in data_path or not data_path.endswith(".gz"):
            continue
        sim_temp_id = test_file[5:-3]
        print(
            f"Processing {data_path} for simulation temperature fraction {sim_temp_id}..."
        )
        try:
            idx = np.where(SIM_TEMPERATURE_FRACTIONS == float(sim_temp_id))[0][0]
        except IndexError:
            print(
                f"Simulation temperature fraction {sim_temp_id} not found in predefined fractions. Skipping..."
            )
            continue

        for classifier in classifiers:
            start_time = time.time()
            if classifier == "DC4":
                preds = apply_dcmodel(data_path, dc4_model)
                acc = (preds == CORRECT_MAP_DC4[exp_name]).sum().item() / len(preds)
            elif classifier == "DC3":
                preds = apply_dcmodel(data_path, dc3_model)
                acc = (preds == CORRECT_MAP_DC3[exp_name]).sum().item() / len(preds)
            else:
                acc = compute_heuristic_accuracy(exp_name, data_path, classifier)
            elapsed_time = time.time() - start_time

            print(
                f"{classifier} accuracy on {sim_temp_id}: {acc:.2f}, elapsed time: {elapsed_time:.2f}s"
            )
            inference_results[idx].append(acc)
            time_results[idx].append(elapsed_time)

    # Save results
    output_path = f"egnn_crystal_classifier/benchmarking/results/{exp_name}_results.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(
        inference_results, index=SIM_TEMPERATURE_FRACTIONS, columns=classifiers
    )
    df.to_csv(output_path, index_label="Simulation Temperature Fraction")

    # Save timings
    time_output_path = (
        f"egnn_crystal_classifier/benchmarking/results/{exp_name}_timings.csv"
    )
    time_df = pd.DataFrame(
        time_results, index=SIM_TEMPERATURE_FRACTIONS, columns=classifiers
    )
    time_df.to_csv(time_output_path, index_label="Simulation Temperature Fraction")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run benchmarking for crystal structure classification."
    )
    parser.add_argument(
        "--exp",
        type=str,
        choices=["al_fcc", "li_bcc", "ti_hcp", "ge_cd"],
        help="Experiment name to run the benchmark for.",
    )
    args = parser.parse_args()

    # Detect if md simulation data exists
    md_data_path = "egnn_crystal_classifier/benchmarking/md"
    if not os.path.exists(md_data_path):
        print(
            "I can't find the MD simulation data directory. Please make sure it's",
            "present in the expected location: 'egnn_crystal_classifier/benchmarking/md'.",
        )

    # If an experiment is specified, run only that one
    if args.exp:
        print(f"== Running benchmark for {args.exp}...")
        run_benchmark(args.exp)
        return

    # Otherwise, run benchmarks for all experiments
    experiments = ["al_fcc", "li_bcc", "ti_hcp", "ge_cd"]
    for exp_name in experiments:
        print(f"== Running benchmark for {exp_name}...")
        run_benchmark(exp_name)


if __name__ == "__main__":
    main()
