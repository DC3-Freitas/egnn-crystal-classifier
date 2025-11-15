import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from egnn_crystal_classifier.constants import *
from egnn_crystal_classifier.data_gen.gen_defect import generate_single_defect_structure

def visualize_defects(positions, labels, title="Defective Structure"):
    """
    Visualizes the defective structure with different colors for perfect atoms,
    vacancy neighbors, and interstitials.

    Args:
        positions (np.ndarray): Nx3 array of atomic positions.
        labels (np.ndarray): N-length array of integer labels for each atom.
            0: perfect, 1: vacancy neighbor, 2: interstitial
        title (str): Title of the plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define colors for different labels
    colors = {0: 'blue', 1: 'orange', 2: 'red'}
    label_names = {0: 'Perfect', 1: 'Vacancy Neighbor', 2: 'Interstitial'}

    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(positions[mask, 0], positions[mask, 1], positions[mask, 2],
                   c=colors[label], label=label_names[label], s=50)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
    return fig, ax

if __name__ == "__main__":
    # Example usage
    # Generate a perfect crystal structure (e.g., simple cubic)
    lattice_constant = 1.0
    grid_size = 5
    x, y, z = np.meshgrid(np.arange(grid_size) * lattice_constant,
                          np.arange(grid_size) * lattice_constant,
                          np.arange(grid_size) * lattice_constant)
    perfect_positions = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    # Generate defects
    defective_positions, labels, label_map = generate_single_defect_structure(
        perfect_positions,
        num_vacancies=5,
        num_interstitials=5
    )

    # Visualize the defective structure
    visualize_defects(defective_positions, labels, title="Defective Crystal Structure")