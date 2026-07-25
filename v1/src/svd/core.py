import torch
from typing import Callable, Literal

DistMetric = Literal["l1", "l2", "cosine"]



# ═══════════════════════════════════════════════════════════════════
# Family 2: Spectral / subspace-based
#   Score = projection onto (or residual from) an SVD subspace.
# ═══════════════════════════════════════════════════════════════════

def _run_svd(G: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k right singular vectors and singular values of G.

    Returns
    -------
    V : Tensor of shape (d, k)
        Right singular vectors, columns ordered by decreasing singular value.
    S : Tensor of shape (k,)
        Corresponding singular values, decreasing.
    """
    _, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
    return Vh[:k].T.contiguous(), S[:k].contiguous()

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
    *,
    singular_values: torch.Tensor | None = None, # (c,) top-c singular values, required if weighted
    weighted: bool = False,
) -> torch.Tensor:
    """Mean squared projection of each row onto right singular vectors in [c_begin:c_end].

    With `weighted=True`, each squared projection on V[:, c] is scaled by the
    corresponding singular value sigma_c before averaging (HaloScope-style
    scoring; Du et al. 2024):

        tau_i = (1/|C|) sum_{c in C} sigma_c * <r_tilde_i, v_c>^2

    With `weighted=False` (default), sigma_c is treated as 1, recovering the
    unweighted mean squared projection.
    """
    assert 0 <= c_begin < c_end <= V.shape[1], \
        f"Invalid range [{c_begin}:{c_end}] for V with {V.shape[1]} components"
    if weighted:
        assert singular_values is not None, \
            "ranged_projection_svd: weighted=True requires `singular_values`"
        assert singular_values.shape[0] >= c_end, (
            f"ranged_projection_svd: singular_values has {singular_values.shape[0]} "
            f"entries; need at least {c_end} to cover c_end={c_end}"
        )
    R_f = R.float()
    if ref is not None:
        R_f = R_f - ref
    proj_sq = (R_f @ V[:, c_begin:c_end]).square()
    if weighted:
        sigma = singular_values[c_begin:c_end].to(proj_sq)
        proj_sq = proj_sq * sigma  # broadcast (T, |C|) * (|C|,)
    scores = proj_sq.mean(dim=1).to(R.dtype)
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