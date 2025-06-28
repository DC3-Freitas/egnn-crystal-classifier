import matplotlib.pyplot as plt
import pandas as pd


def plot_results(exp_names: list[str]) -> None:
    """
    Plot the benchmarking results for the given experiment names.
    Args:
        exp_names (list[str]): List of experiment names to visualize results for.
    """

    plt.style.use("bmh")
    plt.rcParams["font.family"] = "Segoe UI"

    x_dim, y_dim = len(exp_names), 1
    if x_dim > 3:
        y_dim = 2
        x_dim = (x_dim + 1) // 2
    fig, axs = plt.subplots(x_dim, y_dim, figsize=(6 * x_dim, 4 * y_dim))
    if x_dim == 1 and y_dim == 1:
        axs = [axs]
    elif y_dim != 1:
        axs = axs.flatten()

    fig.suptitle("MD Benchmarking Results", fontsize=16, fontweight="bold")

    for i, exp_name in enumerate(exp_names):
        ax = axs[i]

        data = pd.read_csv(
            f"egnn_crystal_classifier/benchmarking/results/{exp_name}_results.csv",
            index_col=0,
        )

        ax.set_title(exp_name)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Simulation Temperature Fraction (T/Tm)")

        for column in data.columns:
            ax.plot(data.index, data[column], label=column)
        ax.legend(fontsize=8, loc="lower left")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize benchmarking results.")
    parser.add_argument(
        "exp_names", nargs="+", help="Experiment name to visualize results for."
    )
    args = parser.parse_args()

    plot_results(args.exp_names)
