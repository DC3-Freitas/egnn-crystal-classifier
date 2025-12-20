"""
Computes the embedding for edge distances based
on the formula outlined in the NequIP and DimeNet paper.
"""

from typing import cast

import torch


def bessel(r: torch.Tensor, n: int, r_c: float) -> torch.Tensor:
    """
    Embeds a batch of edge-distances based on the formula
    outlined in the NequIP and DimeNet paper. Does not reduce to 0
    past the cutoff nor apply a polynomial envelope.

    Args:
        r (B, 1): Edge distances.
        n: Length of the embedding.
        r_c: Theoretical cutoff.

    Returns:
        A tensor of the embeddings with shape (B, n).
    """
    sin_cmp = (
        ((2 / r_c) ** 0.5)
        * torch.sin(r * torch.pi * torch.arange(1, n + 1, device=r.device) / r_c)
        / (r + 1e-8)
    )
    return cast(torch.Tensor, sin_cmp)
