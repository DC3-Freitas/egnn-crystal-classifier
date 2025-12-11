"""
Configs for the entire pipeline.
"""

from dataclasses import dataclass, field


@dataclass
class Config:
    """
    Configs controlling the entire pipeline.

    Attributes:
        num_neighbors: Number of nearest neighbors to consider.
        num_species: Number of species of atoms the model should be able to handle.

        label_map: Mapping from species name to integer.
        crystals: All non-outlier structures (e.g., bcc).

        output_classes: Number of output classes.
        num_convs: Number of times to run interaction block in model.

        node_embedding_init: Size of the initial embedding that will be continously updated.
        node_embedding_frozen: Size of the frozen embedding that will be used in the SC path.

        irreps_edge_sph: Irreps for edge spherical harmonics.
                         Must all have even parity and multiplicity 1.
        irreps_hidden: Irreps for the hidden equivariant embeddings.
                       Must all have even parity.

        edge_dist_n: Size of the radial bessel embedding for the edge distance.
        r_c: Cutoff for radial bessel embedding.
        radial_hidden: Hidden layer size for the edge distance embedding to weights MLP.
                       Note that there will only be one hidden layer for this MLP.

        invariant_embedding_size: Size of the final invariant embedding of each graph.

        node_dropout: Probability of removing each non-center node during training.
        edge_dropout: Probability of removing each edge during training.

        batch_size: Batch size to use for both training and inference.
        train_split_frac: Fraction of total data to use for training.
        train_eval_sample_frac: Fraction of training data to use to calculate loss.

        epochs: Number of epochs for training.
        lr: Initial learning rate.
        eta_min: What the learning rate decays to at the final epoch.
        weight_decay: Weight decay parameter.

        label_smoothing: Label smoothing parameter.
            *label smoothing makes representations more spread out and
             decreases performance. This makes representations cluster more tightly
             which may not always be a good thing as two different but same-structured
             atoms will have closer representations.

        checkpoint_freq: Frequency of saving our model.

        coherence_range_l: Start of range (inclusive) of largest dot product similarities
                           to take the average of for our coherence calculations.
        coherence_range_r: End of range (exclusive) of largest dot product similarities
                           to take the average of for our coherence calculations.
    """

    # Model (general settings)
    num_neighbors: int = 16
    num_species: int = 1

    label_map: dict[str, int] = field(
        default_factory=lambda: {
            "bcc": 0,
            "cd": 1,
            "fcc": 2,
            "hcp": 3,
            "hd": 4,
            "sc": 5,
            "unknown_crystal": 6,
            "amorphous": 7,
        }
    )
    crystals: list[str] = field(
        default_factory=lambda: ["bcc", "cd", "fcc", "hcp", "hd", "sc"]
    )

    output_classes = 6
    num_convs = 3

    # Model (intiial embeddings)
    node_embedding_init: int = 8
    node_embedding_frozen: int = 8

    # Model (irreps)
    irreps_edge_sph: str = "1x0e + 1x1e + 1x2e"  # Must be even and mul=1
    irreps_hidden: str = "8x0e + 8x1e + 8x2e"  # Must be even

    # Model (edge length projection)
    edge_dist_n: int = 8
    r_c: float = 2.0
    radial_hidden: int = 32

    # Model (final invariant embedding)
    invariant_embedding_size = 32

    # Model (dropout)
    node_dropout: float = 0.05
    edge_dropout: float = 0.15

    # Training (data)
    batch_size: int = 512
    train_split_frac: float = 0.8
    train_eval_sample_frac: float = 0.2

    # Training (hyperparameters)
    epochs: int = 20
    lr: float = 0.01
    eta_min: float = 1e-4
    weight_decay: float = 1e-4

    label_smoothing: float = 0.1

    # Training (misc)
    checkpoint_freq: int = 5

    # Coherence
    coherence_range_l = 2  # Inclusive
    coherence_range_r = 5  # Exclusive
