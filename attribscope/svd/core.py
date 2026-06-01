import torch
from typing import Callable, Literal

DistMetric = Literal["l1", "l2", "cosine"]



# ═══════════════════════════════════════════════════════════════════
# Family 2: Spectral / subspace-based
#   Score = projection onto (or residual from) an SVD subspace.
# ═══════════════════════════════════════════════════════════════════

# def _run_svd(G: torch.Tensor, k: int) -> torch.Tensor:
#     """Top-k right singular vectors of G. Shape: (d, k)."""
#     torch.manual_seed(100)
#     _, _, V = torch.svd_lowrank(G.float(), q=k, niter=10)
#     return V.contiguous()

def _run_svd(G: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k right singular vectors of G. Shape: (d, k)."""
    _, _, Vh = torch.linalg.svd(G.float(), full_matrices=False)
    return Vh[:k].T.contiguous()

def projection_svd(
    R: torch.Tensor, # (T, d) matrix of row reps to score
    V: torch.Tensor, # (d, c) matrix of top-c right singular vectors from G
    c: int = 1,
    ref: torch.Tensor | None = None, # (d,) mean gradient for centering, if desired
) -> torch.Tensor:
    """Mean squared projection of each row onto the top-c right singular vectors.

        τᵢ = (1/c) Σⱼ ⟨g̃ᵢ, vⱼ⟩²

    Reference : top-c singular subspace of G (optionally mean-centered).
    SAL interpretation (Du et al. 2024): the leading singular direction
    aligns with the outlier direction, so a large projection flags OOD rows.
    Higher score = more anomalous.
    """
    assert c <= V.shape[1], f"Requested c={c} exceeds n_components={V.shape[1]}"
    R_f = R.float()
    if ref is not None:
        R_f = R_f - ref
    scores =  (R_f @ V[:, :c]).square().mean(dim=1).to(R.dtype)
    return scores

def ranged_projection_svd(
    R: torch.Tensor, # (T, d) matrix of row reps to score
    V: torch.Tensor, # (d, c) matrix of top-c right singular vectors from G
    c_begin: int = 0,
    c_end:   int = 20,
    ref: torch.Tensor | None = None, # (d,) mean gradient for centering, if desired
) -> torch.Tensor:
    assert 0 <= c_begin < c_end <= V.shape[1], \
        f"Invalid range [{c_begin}:{c_end}] for V with {V.shape[1]} components"
    R_f = R.float()
    if ref is not None:
        R_f = R_f - ref
    scores = (R_f @ V[:, c_begin:c_end]).square().mean(dim=1).to(R.dtype)
    return scores

def reconstruction_svd(
    R: torch.Tensor,
    V: torch.Tensor,
    c: int = 5,
    ref: torch.Tensor | None = None,
) -> torch.Tensor:
    """Residual L2 norm after projecting each row onto the top-c SVD subspace.

    Reference : top-c singular subspace of G (optionally mean-centered).
    Rows well-explained by the dominant subspace have small residuals;
    rows off this subspace (structural outliers) have large residuals.
    Higher score = more anomalous.
    """
    assert c <= V.shape[1], f"Requested c={c} exceeds n_components={V.shape[1]}"
    R_f = R.float()
    if ref is not None:
        R_f = R_f - ref
    G_rec = (R_f @ V[:, :c]) @ V[:, :c].T
    return torch.norm(R_f - G_rec, dim=1).to(R.dtype)