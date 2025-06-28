import matplotlib.pyplot as plt
import pandas as pd

def plot_results(exp_name: str) -> None:
    data = pd.read_csv(
        f"egnn_crystal_classifier/benchmarking/results/{exp_name}_results.csv",
        index_col=0
    )

    plt.figure(figsize=(6, 4))
    plt.style.use("bmh")
    plt.rcParams["font.family"] = "Times New Roman"

    plt.title(exp_name)
    plt.ylabel("Accuracy")
    plt.xlabel("Simulation Temperature Fraction T/Tm")

    for column in data.columns:
        plt.plot(data.index, data[column], label=column)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize benchmarking results.")
    parser.add_argument("exp_name", type=str, help="Experiment name to visualize results for.")
    args = parser.parse_args()

    plot_results(args.exp_name)