# pylint: disable=too-many-instance-attributes

"""
Configs for the entire pipeline.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DisorderModelConfig:
    """
    Configs controlling everything related to the disorder model (the one that
    predicts how disordered a local structure is which can be used for things
    like amorphous classification).

    Note that batch_size is applied to all batched operations relating
    to the disorder model.

    Attributes:
        num_neighbors: Number of nearest neighbors to consider.

        irreps_edge_sph: Irreps for initial embedding spherical harmonics.
        irreps_hidden: Irreps for the hidden equivariant embeddings.

        hidden_size: Size of hidden state.
        mlp_hidden: Size of hidden layer in MLP.

        batch_size: Batch size to use for training, inference, and all related batched ops.
        train_split_frac: Fraction of total data to use for training.

        epochs: Number of epochs for training.
        lr: Initial learning rate.
        eta_min: What the learning rate decays to at the final epoch.

        checkpoint_freq: Frequency of saving our model during training.
    """

    num_neighbors: int = 16

    irreps_in: str = "1x2e + 1x4e + 1x6e + 1x8e + 1x10e"
    irreps_mid: str = "6x4e + 6x6e + 6x8e"

    hidden_size: int = 16
    mlp_hidden: int = 64

    batch_size: int = 512
    train_split_frac: float = 0.8

    epochs: int = 20
    lr: float = 5e-3
    eta_min: float = 5e-5

    checkpoint_freq: int = 5


@dataclass(frozen=True)
class NequIPConfig:
    """
    Configs controlling everything related to NequIP (the core model)
    including more general things like outlier detection and labels.

    Note that batch_size is applied to all batched operations relating
    to NequIP.

    Attributes:
        num_neighbors: Number of nearest neighbors to consider.
        num_species: Number of species of atoms the model should be able to handle.

        label_map: Mapping from species name to integer.
        crystals: All non-outlier structures (e.g. bcc).

        output_classes: Number of output classes.
        num_convs: Number of times to run interaction block in model.

        node_embedding_init: Size of the initial embedding that will be continously updated.
        node_embedding_frozen: Size of the frozen embedding that will be used in the SC path.

        irreps_edge_sph: Irreps for edge spherical harmonics.
                         Must exist and all have even parity and multiplicity 1.
        irreps_hidden: Irreps for the hidden equivariant embeddings.
                       Must exist and all have even parity.

        edge_dist_n: Size of the radial bessel embedding for the edge distance.
        r_c: Cutoff for radial bessel embedding.
        radial_hidden: Hidden layer size for the edge distance embedding to weights MLP.
                       Note that there will only be one hidden layer for this MLP.

        invariant_embedding_size: Size of the final invariant embedding of each graph.

        node_dropout: Probability of removing each non-center node during training.
        edge_dropout: Probability of removing each edge during training.

        batch_size: Batch size to use for training, inference, and all related batched ops.
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

        checkpoint_freq: Frequency of saving our model during training.

        cutoff: Cutoff (in terms of percentile) for classification unknown structure.
                For example a cutoff of 1% corresponds to determining if a structure is
                unknown if its distance is greater than 99% of sample data distances.
    """

    # Broad information
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

    output_classes: int = 6
    num_convs: int = 3

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
    invariant_embedding_size: int = 32

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

    # Outlier
    cutoff = 1.0  # Percentile


@dataclass(frozen=True)
class ConfigAll:
    """
    Central config that includes both configurations for disorder
    model and for NequIP.

    Attributes:
        disorder_model_config: Configs for disorder model.
        nequip_config: Configs for NequIP model.
    """

    disorder_model_config: DisorderModelConfig = field(
        default_factory=DisorderModelConfig
    )
    nequip_config: NequIPConfig = field(default_factory=NequIPConfig)
