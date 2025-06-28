import matplotlib.pyplot as plt
import pandas as pd


def plot_results(exp_names: list[str]) -> None:
    """
    Plot the benchmarking results for the given experiment names.
    Args:
        exp_names (list[str]): List of experiment names to visualize results for.
    """

    plt.style.use("seaborn-v0_8-deep")
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

    fig.suptitle("MD Benchmarking Results", fontsize=16, fontstyle="italic")

    for i, exp_name in enumerate(exp_names):
        ax = axs[i]

        data = pd.read_csv(
            f"egnn_crystal_classifier/benchmarking/results/{exp_name}_results.csv",
            index_col=0,
        )

        ax.set_title(exp_name)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Simulation Temperature Fraction (T/Tm)")

        # set x and y range
        ax.set_xlim(0.04, 1.2)
        ax.set_ylim(0.8, 1.01)

        # line at T/Tm = 1
        ax.axvline(x=1, color="darkgray", linestyle="--")

        for column in data.columns:
            ax.plot(data.index, data[column], label=column)
        ax.legend(fontsize=8, loc="lower left")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_timings(files):
    """
    Ingest all timing files and plot average timings for each method.
    """
    times = {}
    for file in files:
        df = pd.read_csv(file)
        for column in df.columns:
            if column == "Simulation Temperature Fraction":
                continue
            if column not in times:
                times[column] = []
            times[column].append(df[column].mean())

    plt.style.use("seaborn-v0_8-deep")
    plt.rcParams["font.family"] = "Segoe UI"

    replace_map = {
        "DC4": "DC4",
        "DC3": "DC3",
        "Common Neighbor Analysis (Non-Diamond)": "CNA (ND)",
        "Common Neighbor Analysis (Diamond)": "CNA (D)",
        "Interval Common Neighbor Analysis": "ICNA",
        "Ackland-Jones Analysis": "AJ",
        "VoroTop Analysis": "VoroTop",
        "Chill+": "Chill+",
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        [replace_map[k] for k in times.keys()],
        [sum(t) / len(t) for t in times.values()],
        width=0.4,
        color="crimson",
    )
    ax.set_title("Average Classification Time", fontstyle="italic")
    ax.set_ylabel("Average Time (s)")
    ax.set_xlabel("Method")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize benchmarking results.")
    parser.add_argument(
        "exp_names", nargs="+", help="Experiment name to visualize results for."
    )
    parser.add_argument(
        "--timings", action="store_true", help="Plot timings instead of accuracies."
    )
    args = parser.parse_args()

    if args.timings:
        timing_files = [
            f"egnn_crystal_classifier/benchmarking/results/{exp_name}_timings.csv"
            for exp_name in args.exp_names
        ]
        plot_timings(timing_files)
    else:
        plot_results(args.exp_names)
