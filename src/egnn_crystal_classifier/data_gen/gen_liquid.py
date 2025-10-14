"""
We need to generate synthetic liquid structures that closely resemble real liquids.
This means that generated structures must obey certain physical constraints,
such as density and minimum interatomic distances. We will use poisson disk sampling
to generate structures that obey these constraints.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import List, Tuple, Optional
import random

from egnn_crystal_classifier.constants import NN_COUNT


class PoissonDiskSampler:
    """
    Generates liquid structures using Poisson disk sampling to ensure
    realistic atomic distributions with proper minimum distances.
    """
    
    def __init__(self, min_distance: float, box_size: Tuple[float, float, float]):
        """
        Initialize the Poisson disk sampler.
        
        Args:
            min_distance: Minimum distance between atoms
            box_size: Size of the simulation box (x, y, z)
        """
        self.min_distance = min_distance
        self.box_size = np.array(box_size)
        self.cell_size = min_distance / np.sqrt(3)  # Ensures max one point per cell
        self.grid_size = np.ceil(self.box_size / self.cell_size).astype(int)
        
    def generate_points(self, target_density: float, max_attempts: int = 30) -> np.ndarray:
        """
        Generate points using Poisson disk sampling.
        
        Args:
            target_density: Target number density (atoms per unit volume)
            max_attempts: Maximum attempts to place each point
            
        Returns:
            Array of 3D positions
        """
        # Calculate target number of points
        volume = np.prod(self.box_size)
        target_points = int(target_density * volume)
        
        # Initialize grid and point lists
        grid = np.full(self.grid_size, -1, dtype=int)
        points = []
        active_list = []
        
        # Generate first point randomly
        first_point = np.random.uniform(0, self.box_size)
        points.append(first_point)
        active_list.append(0)
        
        grid_pos = (first_point / self.cell_size).astype(int)
        grid[tuple(grid_pos)] = 0
        
        while active_list and len(points) < target_points:
            # Choose random point from active list
            idx = random.choice(range(len(active_list)))
            active_idx = active_list[idx]
            center = points[active_idx]
            
            # Try to generate new point around this center
            point_found = False
            for _ in range(max_attempts):
                # Generate candidate point in annulus
                angle = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, np.pi)
                r = np.random.uniform(self.min_distance, 2 * self.min_distance)
                
                # Convert to Cartesian coordinates
                candidate = center + r * np.array([
                    np.sin(phi) * np.cos(angle),
                    np.sin(phi) * np.sin(angle),
                    np.cos(phi)
                ])
                
                # Check if point is within bounds
                if np.any(candidate < 0) or np.any(candidate >= self.box_size):
                    continue
                
                # Check if point is valid (far enough from existing points)
                if self._is_valid_point(candidate, points, grid):
                    points.append(candidate)
                    active_list.append(len(points) - 1)
                    
                    grid_pos = (candidate / self.cell_size).astype(int)
                    grid[tuple(grid_pos)] = len(points) - 1
                    point_found = True
                    break
            
            if not point_found:
                active_list.pop(idx)
        
        return np.array(points)
    
    def _is_valid_point(self, candidate: np.ndarray, points: List[np.ndarray], 
                       grid: np.ndarray) -> bool:
        """Check if a candidate point is valid (maintains minimum distance)."""
        grid_pos = (candidate / self.cell_size).astype(int)
        
        # Check surrounding grid cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    check_pos = grid_pos + np.array([dx, dy, dz])
                    
                    # Skip if outside grid
                    if np.any(check_pos < 0) or np.any(check_pos >= self.grid_size):
                        continue
                    
                    point_idx = grid[tuple(check_pos)]
                    if point_idx >= 0:
                        distance = np.linalg.norm(candidate - points[point_idx])
                        if distance < self.min_distance:
                            return False
        
        return True


def generate_liquid_structure(density: float, min_distance: float, 
                            box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0),
                            temperature: float = 1.0) -> np.ndarray:
    """
    Generate a liquid structure with realistic atomic distribution.
    
    Args:
        density: Number density (atoms per unit volume)
        min_distance: Minimum distance between atoms (in Angstroms)
        box_size: Simulation box dimensions
        temperature: Temperature parameter (affects disorder)
        
    Returns:
        Array of atomic positions
    """
    sampler = PoissonDiskSampler(min_distance, box_size)
    positions = sampler.generate_points(density)
    
    # Add thermal disorder based on temperature
    if temperature > 0:
        thermal_noise = np.random.normal(0, temperature * 0.1, positions.shape)
        positions += thermal_noise
        
        # Keep atoms within box bounds
        positions = np.mod(positions, box_size)
    
    return positions


def generate_multiple_liquid_structures(n_structures: int = 10,
                                      density_range: Tuple[float, float] = (0.8, 1.2),
                                      min_distance_range: Tuple[float, float] = (2.0, 3.0),
                                      temperature_range: Tuple[float, float] = (0.5, 2.0),
                                      box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0)) -> Tuple[List[np.ndarray], List[str]]:
    """
    Generate multiple liquid structures with varying parameters.
    
    Args:
        n_structures: Number of structures to generate
        density_range: Range of densities to sample from
        min_distance_range: Range of minimum distances
        temperature_range: Range of temperatures
        box_size: Simulation box size
        
    Returns:
        Tuple of (positions_list, labels_list)
    """
    positions_list = []
    labels_list = []
    
    for i in range(n_structures):
        # Sample random parameters
        density = np.random.uniform(*density_range)
        min_distance = np.random.uniform(*min_distance_range)
        temperature = np.random.uniform(*temperature_range)
        
        # Generate structure
        positions = generate_liquid_structure(density, min_distance, box_size, temperature)
        
        # Create labels (all atoms are "liquid")
        labels = ["liquid"] * len(positions)
        
        positions_list.append(positions)
        labels_list.append(labels)
        
        print(f"Generated liquid structure {i+1}/{n_structures}: "
              f"{len(positions)} atoms, density={density:.3f}, "
              f"min_dist={min_distance:.3f}, temp={temperature:.3f}")
    
    return positions_list, labels_list


def gen(n_structures: int = 50, 
        use_realistic_params: bool = True) -> Tuple[np.ndarray, List[str], dict]:
    """
    Main function to generate synthetic liquid data compatible with the main data pipeline.
    
    Args:
        n_structures: Number of liquid structures to generate
        use_realistic_params: Whether to use realistic physical parameters
        
    Returns:
        Tuple of (x_data, y_data, label_map) where:
        - x_data: Array of neighbor configurations for each atom
        - y_data: List of labels for each atom
        - label_map: Mapping from labels to integers
    """
    print(f"Generating {n_structures} synthetic liquid structures...")
    
    if use_realistic_params:
        # Realistic parameters for liquid metals
        density_range = (0.05, 0.15)  # atoms per cubic Angstrom
        min_distance_range = (2.2, 2.8)  # typical metallic bond lengths
        temperature_range = (0.8, 1.5)  # thermal disorder
        box_size = (25.0, 25.0, 25.0)  # larger box for more atoms
    else:
        # More varied parameters for training diversity
        density_range = (0.03, 0.2)
        min_distance_range = (1.8, 3.5)
        temperature_range = (0.3, 2.5)
        box_size = (20.0, 20.0, 20.0)
    
    # Generate liquid structures
    positions_list, labels_list = generate_multiple_liquid_structures(
        n_structures=n_structures,
        density_range=density_range,
        min_distance_range=min_distance_range,
        temperature_range=temperature_range,
        box_size=box_size
    )
    
    # Process data to match the format expected by the main pipeline
    print("Processing data for neighbor analysis...")
    
    x_data_new = []
    y_data_new = []
    
    for positions, labels in zip(positions_list, labels_list):
        if len(positions) < NN_COUNT + 1:
            print(f"Skipping structure with only {len(positions)} atoms (need at least {NN_COUNT + 1})")
            continue
            
        # Build KD-tree for nearest neighbor search
        tree = cKDTree(positions)
        neighbors = tree.query(positions, k=NN_COUNT + 1)[1]
        
        # Create neighbor configurations for each atom
        for i, neigh_indices in enumerate(neighbors):
            neighbor_positions = positions[neigh_indices]
            x_data_new.append(neighbor_positions)
            y_data_new.append(labels[i])
    
    x_data = np.array(x_data_new)
    label_map = {"liquid": 0}  # Simple mapping for liquid structures
    
    print(f"Generated {len(x_data)} atomic environments from {len(positions_list)} structures")
    
    return x_data, y_data_new, label_map
