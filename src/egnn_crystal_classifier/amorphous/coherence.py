from scipy.spatial import cKDTree
import numpy as np

def compute_coherence(positions, embeddings, num_neighbors=16):
    """
    Computes the coherence of ML embeddings for atoms in a crystal structure.
    
    Args:
        positions (np.ndarray): Array of atomic positions in the crystal structure.
        embeddings (np.ndarray): Array of ML embeddings for each atom.
        cutoff (float): The distance threshold for coherence.
        
    Returns:
        np.ndarray: Array of coherence values for each atom, where lower values indicate higher coherence.
    """

    tree = cKDTree(positions)
    neighbors = tree.query(positions, k=num_neighbors + 1)[1][:, 1:]
    embedding_similarity = np.zeros(len(positions))
    for i in range(len(positions)):
        for j, neigh in enumerate(neighbors[i]):
            embedding_similarity[i] += np.dot(
                embeddings[i].conjugate(), embeddings[neigh]
            ).real
    return embedding_similarity / num_neighbors

def get_amorphous_mask(positions, embeddings, cutoff=-1):
    """
    Determines which atoms are amorphous based on coherence.
    
    Args:
        positions (np.ndarray): Array of atomic positions in the crystal structure.
        embeddings (np.ndarray): Array of ML embeddings for each atom.
        cutoff (float): The distance threshold for coherence.
            If cutoff is -1, then we attempt to compute it automatically
            by taking the histogram and finding value that minimizes inter-class variance.
        
    Returns:
        np.ndarray: Boolean array indicating whether each atom is amorphous (True) or not (False).
    """
    
    embedding_similarity = compute_coherence(positions, embeddings, cutoff)
    if cutoff == -1:
        # automatically determine the cutoff using Otsu's method
        hist, bin_edges = np.histogram(embedding_similarity, bins=100)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

        # class probabilities over all possible thresholds
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]

        # class means over all possible thresholds
        mean1 = np.cumsum(hist * bin_centers) / weight1
        mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

        # calculate interclass variance
        variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

        idx = np.argmax(variance12)
        cutoff = bin_centers[idx]
    return embedding_similarity < cutoff