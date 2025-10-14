"""
3D visualization script for liquid structure generation.
Provides various visualization functions to analyze and display
the generated liquid structures.

Features:
- 3D scatter plots of atomic positions with optional bond visualization
- Radial distribution function (RDF) analysis
- Parameter study visualizations comparing multiple structures
- Density and coordination analysis plots
- Grid comparison views for multiple structures

Usage Examples:
    
    # Basic single structure visualization
    from egnn_crystal_classifier.data_gen.visualize_liquid_gen import visualize_single_structure
    visualize_single_structure(density=0.1, min_distance=2.5, show_bonds=True)
    
    # Parameter study with multiple structures  
    from egnn_crystal_classifier.data_gen.visualize_liquid_gen import visualize_parameter_study
    visualize_parameter_study(n_structures=9)
    
    # Advanced usage with LiquidVisualizer class
    from egnn_crystal_classifier.data_gen.visualize_liquid_gen import LiquidVisualizer
    from egnn_crystal_classifier.data_gen.gen_liquid import generate_liquid_structure
    
    positions = generate_liquid_structure(0.1, 2.5, (20, 20, 20))
    visualizer = LiquidVisualizer()
    fig = visualizer.plot_3d_structure(positions, show_bonds=True)
    fig = visualizer.plot_radial_distribution(positions)

Dependencies:
- matplotlib (with 3D support)
- numpy
- scipy
- seaborn (for enhanced color schemes)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from typing import List, Tuple, Optional

from egnn_crystal_classifier.data_gen.gen_liquid import generate_liquid_structure, generate_multiple_liquid_structures


class LiquidVisualizer:
    """
    Class for visualizing liquid structures and their properties.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Figure size for matplotlib plots
        """
        self.figsize = figsize
        # Set up color schemes
        self.colors = plt.cm.viridis
        plt.style.use('default')
        
    def plot_3d_structure(self, positions: np.ndarray, 
                         title: str = "Liquid Structure",
                         atom_size: float = 50.0,
                         show_bonds: bool = False,
                         bond_cutoff: float = 3.5,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot a 3D visualization of the atomic structure.
        
        Args:
            positions: Array of atomic positions (N x 3)
            title: Plot title
            atom_size: Size of atom markers
            show_bonds: Whether to show bonds between nearby atoms
            bond_cutoff: Maximum distance for bond visualization
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot atoms
        scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                           c=positions[:, 2], cmap=self.colors, 
                           s=atom_size, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Add bonds if requested
        if show_bonds and len(positions) > 1:
            tree = cKDTree(positions)
            pairs = tree.query_pairs(bond_cutoff)
            for i, j in pairs:
                ax.plot([positions[i, 0], positions[j, 0]],
                       [positions[i, 1], positions[j, 1]],
                       [positions[i, 2], positions[j, 2]], 
                       'gray', alpha=0.3, linewidth=0.5)
        
        # Customize plot
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(title)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Z coordinate (Å)', shrink=0.8)
        
        # Make axes equal
        max_range = np.array([positions[:,0].max()-positions[:,0].min(),
                             positions[:,1].max()-positions[:,1].min(),
                             positions[:,2].max()-positions[:,2].min()]).max() / 2.0
        mid_x = (positions[:,0].max()+positions[:,0].min()) * 0.5
        mid_y = (positions[:,1].max()+positions[:,1].min()) * 0.5
        mid_z = (positions[:,2].max()+positions[:,2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_radial_distribution(self, positions: np.ndarray,
                               r_max: float = 10.0,
                               n_bins: int = 100,
                               title: str = "Radial Distribution Function",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the radial distribution function (RDF) for the structure.
        
        Args:
            positions: Array of atomic positions
            r_max: Maximum radius for RDF calculation
            n_bins: Number of bins for histogram
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        # Calculate all pairwise distances
        distances = pdist(positions)
        
        # Create histogram
        bin_edges = np.linspace(0, r_max, n_bins + 1)
        hist, _ = np.histogram(distances, bins=bin_edges)
        r = (bin_edges[1:] + bin_edges[:-1]) / 2
        
        # Normalize by volume and density
        volume = (4 * np.pi * r**2) * (bin_edges[1] - bin_edges[0])
        box_volume = np.prod(np.ptp(positions, axis=0))
        density = len(positions) / box_volume
        n_pairs = len(positions) * (len(positions) - 1) / 2
        
        # Avoid division by zero
        volume[volume == 0] = 1
        rdf = hist / (volume * density * n_pairs / len(positions))
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(r, rdf, 'b-', linewidth=2, label='RDF')
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Random distribution')
        
        ax.set_xlabel('Distance (Å)')
        ax.set_ylabel('g(r)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_density_analysis(self, positions_list: List[np.ndarray],
                            densities: List[float],
                            min_distances: List[float],
                            title: str = "Density vs Structure Analysis",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot analysis of density vs structural properties.
        
        Args:
            positions_list: List of position arrays
            densities: List of densities used for generation
            min_distances: List of minimum distances used
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Calculate structural properties
        n_atoms = [len(pos) for pos in positions_list]
        avg_distances = []
        coord_numbers = []
        
        for positions in positions_list:
            if len(positions) > 1:
                distances = pdist(positions)
                avg_distances.append(np.mean(distances))
                
                # Calculate coordination numbers (neighbors within 1.5 * min_distance)
                tree = cKDTree(positions)
                coord_nums = []
                for i in range(len(positions)):
                    neighbors = tree.query_ball_point(positions[i], 4.0)  # 4 Å cutoff
                    coord_nums.append(len(neighbors) - 1)  # Exclude self
                coord_numbers.append(np.mean(coord_nums))
            else:
                avg_distances.append(0)
                coord_numbers.append(0)
        
        # Plot 1: Number of atoms vs density
        ax1.scatter(densities, n_atoms, c=min_distances, cmap='viridis', s=50, alpha=0.7)
        ax1.set_xlabel('Density (atoms/Ų)')
        ax1.set_ylabel('Number of atoms')
        ax1.set_title('Atoms vs Density')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Average distance vs density
        scatter2 = ax2.scatter(densities, avg_distances, c=min_distances, cmap='viridis', s=50, alpha=0.7)
        ax2.set_xlabel('Density (atoms/Ų)')
        ax2.set_ylabel('Average distance (Å)')
        ax2.set_title('Average Distance vs Density')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Coordination number vs density
        ax3.scatter(densities, coord_numbers, c=min_distances, cmap='viridis', s=50, alpha=0.7)
        ax3.set_xlabel('Density (atoms/Ų)')
        ax3.set_ylabel('Average coordination number')
        ax3.set_title('Coordination vs Density')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Min distance vs coordination
        scatter4 = ax4.scatter(min_distances, coord_numbers, c=densities, cmap='plasma', s=50, alpha=0.7)
        ax4.set_xlabel('Minimum distance (Å)')
        ax4.set_ylabel('Average coordination number')
        ax4.set_title('Coordination vs Min Distance')
        ax4.grid(True, alpha=0.3)
        
        # Add colorbars
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Min distance (Å)')
        cbar4 = plt.colorbar(scatter4, ax=ax4)
        cbar4.set_label('Density (atoms/Ų)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_comparison_grid(self, positions_list: List[np.ndarray],
                           parameters_list: List[Tuple[float, float, float]],
                           titles: Optional[List[str]] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot a grid comparison of multiple structures.
        
        Args:
            positions_list: List of position arrays
            parameters_list: List of (density, min_distance, temperature) tuples
            titles: Optional list of titles for each subplot
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        n_structures = len(positions_list)
        cols = min(3, n_structures)
        rows = (n_structures + cols - 1) // cols
        
        fig = plt.figure(figsize=(5 * cols, 5 * rows))
        
        for i, (positions, params) in enumerate(zip(positions_list, parameters_list)):
            ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
            
            # Plot structure
            scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                               c=positions[:, 2], cmap=self.colors, 
                               s=30, alpha=0.8, edgecolors='black', linewidth=0.3)
            
            # Set title
            if titles and i < len(titles):
                title = titles[i]
            else:
                density, min_dist, temp = params
                title = f'ρ={density:.3f}, d={min_dist:.2f}, T={temp:.2f}'
            
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('X (Å)', fontsize=8)
            ax.set_ylabel('Y (Å)', fontsize=8)
            ax.set_zlabel('Z (Å)', fontsize=8)
            
            # Make axes equal
            max_range = np.array([positions[:,0].max()-positions[:,0].min(),
                                 positions[:,1].max()-positions[:,1].min(),
                                 positions[:,2].max()-positions[:,2].min()]).max() / 2.0
            mid_x = (positions[:,0].max()+positions[:,0].min()) * 0.5
            mid_y = (positions[:,1].max()+positions[:,1].min()) * 0.5
            mid_z = (positions[:,2].max()+positions[:,2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def visualize_single_structure(density: float = 0.1, 
                             min_distance: float = 2.5,
                             temperature: float = 1.0,
                             box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0),
                             show_bonds: bool = True,
                             show_rdf: bool = True,
                             save_prefix: Optional[str] = None):
    """
    Generate and visualize a single liquid structure.
    
    Args:
        density: Target density
        min_distance: Minimum interatomic distance
        temperature: Temperature parameter
        box_size: Simulation box size
        show_bonds: Whether to show bonds in 3D plot
        show_rdf: Whether to show radial distribution function
        save_prefix: Prefix for saved files
    """
    print(f"Generating liquid structure with density={density}, min_distance={min_distance}, temperature={temperature}")
    
    # Generate structure
    positions = generate_liquid_structure(density, min_distance, box_size, temperature)
    
    print(f"Generated {len(positions)} atoms")
    
    # Create visualizer
    visualizer = LiquidVisualizer()
    
    # Plot 3D structure
    title = f"Liquid Structure (ρ={density:.3f}, d={min_distance:.2f}Å, T={temperature:.2f})"
    save_path_3d = f"{save_prefix}_3d.png" if save_prefix else None
    fig1 = visualizer.plot_3d_structure(positions, title=title, show_bonds=show_bonds, save_path=save_path_3d)
    
    # Plot RDF if requested
    if show_rdf:
        save_path_rdf = f"{save_prefix}_rdf.png" if save_prefix else None
        fig2 = visualizer.plot_radial_distribution(positions, save_path=save_path_rdf)
    
    plt.show()


def visualize_parameter_study(n_structures: int = 12,
                            density_range: Tuple[float, float] = (0.05, 0.15),
                            min_distance_range: Tuple[float, float] = (2.0, 3.0),
                            temperature_range: Tuple[float, float] = (0.5, 2.0),
                            save_prefix: Optional[str] = None):
    """
    Generate and visualize multiple structures with varying parameters.
    
    Args:
        n_structures: Number of structures to generate
        density_range: Range of densities to explore
        min_distance_range: Range of minimum distances
        temperature_range: Range of temperatures
        save_prefix: Prefix for saved files
    """
    print(f"Generating {n_structures} structures for parameter study...")
    
    # Generate structures
    positions_list, _ = generate_multiple_liquid_structures(
        n_structures=n_structures,
        density_range=density_range,
        min_distance_range=min_distance_range,
        temperature_range=temperature_range,
        box_size=(15.0, 15.0, 15.0)  # Smaller box for faster visualization
    )
    
    # Store parameters used
    np.random.seed(42)  # For reproducible parameter generation
    parameters_list = []
    for _ in range(n_structures):
        density = np.random.uniform(*density_range)
        min_distance = np.random.uniform(*min_distance_range)
        temperature = np.random.uniform(*temperature_range)
        parameters_list.append((density, min_distance, temperature))
    
    # Create visualizer
    visualizer = LiquidVisualizer()
    
    # Plot comparison grid
    save_path_grid = f"{save_prefix}_grid.png" if save_prefix else None
    fig1 = visualizer.plot_comparison_grid(positions_list, parameters_list, save_path=save_path_grid)
    
    # Plot density analysis
    densities = [p[0] for p in parameters_list]
    min_distances = [p[1] for p in parameters_list]
    save_path_analysis = f"{save_prefix}_analysis.png" if save_prefix else None
    fig2 = visualizer.plot_density_analysis(positions_list, densities, min_distances, save_path=save_path_analysis)
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Liquid Structure Visualization Demo")
    print("=" * 40)
    
    # Visualize a single structure
    print("1. Single structure visualization...")
    visualize_single_structure(
        density=0.1,
        min_distance=2.5,
        temperature=1.0,
        show_bonds=True,
        show_rdf=True,
        save_prefix="liquid_demo"
    )
    
    # Parameter study
    print("\n2. Parameter study visualization...")
    visualize_parameter_study(
        n_structures=9,
        save_prefix="liquid_study"
    )