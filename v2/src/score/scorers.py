"""Per-step base scorers over SVD geometry — one registry, sign convention per method.

THE GEOMETRY THESE EXPLOIT
--------------------------
Transformer hidden states are extremely anisotropic: almost all of the energy of a set
of step vectors lies along a few shared directions ("the narrow cone"). Fitting an
SVD to the TRAIN step matrix and taking the top right-singular vectors therefore
yields a basis for what a *typical* step looks like. Empirically (see
``src/analysis/geometry.py``) the top uncentered singular vector is the mean direction
itself — ``cos(u_0, mean) = 1.00`` — and it alone carries 65-90% of all energy. So the
band subspace ``span(u_c : c in [c_begin, c_end))`` is best read as "the common mode".

Decisive-error steps sit slightly OFF that common mode. That single fact admits two
dual formulations, and the choice between them is what the registry encodes:

  * **similarity** — how much of the step lies INSIDE the common mode. Error steps have
    LESS, so the score is "lower = error" (``asc``) and must be flipped before it can
    be combined with anything. This is ``proj``.
  * **distance** — how much lies OUTSIDE it. Error steps have MORE, so the score is
    natively "higher = error" (``desc``) and needs NO orientation. This is ``resid`` /
    ``angres`` / ``maha``.

They are two sides of one identity, ``||v||^2 = ||P_band v||^2 + r^2``: the band energy
and the residual sum to the squared norm. Which side you score changes only the sign
convention — except that the residual side can be normalised by ``||v||``, and that
matters (below).

WHY ``angres`` IS THE PRINCIPLED CHOICE
---------------------------------------
``||P v||^2 ~ ||v||^2 cos^2(theta)`` conflates two things: how strongly the model
activates (norm) and where it points (alignment). Measured separately, only alignment
carries signal — per-step norm ranks the gold error step at rank-AUC ~0.25-0.54 (chance
or worse), while ``sin^2(theta)`` reaches ~0.7-0.84. ``angres`` is exactly that
norm-free quantity, bounded in [0,1], natively oriented. ``resid``/``maha`` retain the
norm term and score measurably worse.

SIGNATURE
---------
All scorers share ONE signature so the grid can treat them uniformly:

    fn(R, V, c_begin, c_end, ref=None, *, singular_values=None, weighted=False) -> (T,)

``R`` (T,d) step reps; ``V`` (d,k) top-k right singular vectors of the TRAIN matrix;
``singular_values`` the FULL train spectrum (length ``min(T_train, d)``, not just the
top k — ``maha`` needs the discarded tail); ``ref`` the train mean when centering.
Band is the half-open ``[c_begin, c_end)``.

``METHOD_SUPPORTS_WEIGHTED = {proj}`` (sigma-scaling of each squared projection).
Configs never enable it — unweighted wins empirically — but it is kept so the
weighted variant remains reproducible for reference.

    from src.score.scorers import SCORERS, METHOD_DIRECTION, native_directions
"""
from __future__ import annotations

import torch
from torch import Tensor

EPS = 1e-12


# ── projection family ───────────────────────────────────────────────────────
def proj(R, V, c_begin=0, c_end=20, ref=None, *, singular_values=None, weighted=False):
    """Mean squared projection onto the band singular vectors.

    tau = (1/|band|) sum_{c in band} <v~, u_c>^2   (x sigma_c if weighted). Lower = error.
    """
    R_f = R.float()
    if ref is not None:
        R_f = R_f - ref
    proj_sq = (R_f @ V[:, c_begin:c_end]).square()          # (T, |band|)
    if weighted:
        assert singular_values is not None, "proj weighted=True needs singular_values"
        proj_sq = proj_sq * singular_values[c_begin:c_end].to(proj_sq)
    return proj_sq.mean(dim=1).to(R.dtype)


# ── distance family (natively higher = error) ───────────────────────────────
def _band_and_norm(R, V, c_begin, c_end, ref) -> tuple[Tensor, Tensor, Tensor]:
    """Return (band_energy_sum, sq_norm, coeffs_sq) for the centered/raw reps."""
    R_f = R.float()
    if ref is not None:
        R_f = R_f - ref
    coeffs_sq = (R_f @ V[:, c_begin:c_end]).square()        # (T, |band|)
    return coeffs_sq.sum(dim=1), R_f.square().sum(dim=1), coeffs_sq


def resid(R, V, c_begin=0, c_end=20, ref=None, *, singular_values=None, weighted=False):
    """Residual energy off the band subspace: r^2 = ||v~||^2 - sum_band <v~,u_c>^2. Higher = error."""
    assert not weighted, "resid does not support weighted"
    band, sq_norm, _ = _band_and_norm(R, V, c_begin, c_end, ref)
    return (sq_norm - band).clamp_min(0.0).to(R.dtype)


def angres(R, V, c_begin=0, c_end=20, ref=None, *, singular_values=None, weighted=False):
    """Angular residual sin^2(theta) = r^2 / ||v~||^2 in [0,1], norm-free. Higher = error."""
    assert not weighted, "angres does not support weighted"
    band, sq_norm, _ = _band_and_norm(R, V, c_begin, c_end, ref)
    return ((sq_norm - band).clamp_min(0.0) / (sq_norm + EPS)).to(R.dtype)


def maha(R, V, c_begin=0, c_end=20, ref=None, *, singular_values=None, weighted=False):
    """PPCA anomaly: sum_band <v~,u_c>^2/sigma_c^2 + r^2/sigma_bar_resid^2. Higher = error.

    This is the Gaussian/probabilistic reading of the same geometry: model the step
    distribution as isotropic noise around the band subspace, and score a step by its
    negative log-density. Two terms, matching the two parts of the norm identity:

      * inside the band, each direction gets WHITENED by its own train variance
        ``sigma_c^2`` — a deviation along a low-variance direction is more surprising
        than the same deviation along a high-variance one;
      * outside the band, the residual is divided by ``sigma_bar_resid^2``, the mean of
        the OFF-BAND ``sigma^2`` over the FULL train spectrum. That is the classic
        PPCA "average of the discarded eigenvalues" estimate of the leftover variance,
        and it is why the full spectrum (not just the top k) has to be threaded in.

    The global ``1/(n-1)`` that would turn each ``sigma^2`` into a variance is a single
    positive constant shared by both terms, so it cancels under per-trajectory ranking
    and is omitted. Whitening makes this scale-aware where ``angres`` is scale-free —
    in practice it tracks ``resid`` and both trail ``angres``.
    """
    assert not weighted, "maha does not support weighted"
    assert singular_values is not None, "maha needs the full-spectrum singular_values"
    band, sq_norm, coeffs_sq = _band_and_norm(R, V, c_begin, c_end, ref)
    sv2 = singular_values.float().square()                  # (full,)
    whitened = (coeffs_sq / (sv2[c_begin:c_end] + EPS)).sum(dim=1)
    mask = torch.ones(sv2.shape[0], dtype=torch.bool, device=sv2.device)
    mask[c_begin:c_end] = False
    offband = sv2[mask]
    sigma_resid = offband.mean() if offband.numel() > 0 else sv2[-1]
    r2 = (sq_norm - band).clamp_min(0.0)
    return (whitened + r2 / (sigma_resid + EPS)).to(R.dtype)


# ── norm baselines (no SVD; band ignored) ───────────────────────────────────
def _norm(R, ref, p):
    R_f = R.float()
    if ref is not None:
        R_f = R_f - ref
    return R_f.norm(p=p, dim=1)


def norm_l2(R, V, c_begin=0, c_end=0, ref=None, *, singular_values=None, weighted=False):
    """L2 norm of the (optionally centered) rep. Both directions kept."""
    return _norm(R, ref, 2).to(R.dtype)


def norm_l1(R, V, c_begin=0, c_end=0, ref=None, *, singular_values=None, weighted=False):
    """L1 norm of the (optionally centered) rep. Both directions kept."""
    return _norm(R, ref, 1).to(R.dtype)


# ── registry + metadata ─────────────────────────────────────────────────────
SCORERS = {
    "proj": proj, "resid": resid, "angres": angres, "maha": maha,
    "norm_l2": norm_l2, "norm_l1": norm_l1,
}
METHOD_DIRECTION = {
    "proj": "asc", "resid": "desc", "angres": "desc", "maha": "desc",
    "norm_l2": "both", "norm_l1": "both",
}
METHOD_SUPPORTS_WEIGHTED = {"proj"}
DISTANCE_METHODS = {"resid", "angres", "maha"}   # natively desc, need no orientation


def native_directions(method: str) -> list[str]:
    """Directions to EMIT for a method (both for norms, else its single native one)."""
    d = METHOD_DIRECTION[method]
    return ["asc", "desc"] if d == "both" else [d]


def native_direction(method: str, direction: str | None = None) -> str:
    """Resolve a single native direction; for 'both' methods fall back to a row's
    ``direction`` column (NaN/None -> 'asc')."""
    d = METHOD_DIRECTION[method]
    if d != "both":
        return d
    if isinstance(direction, str) and direction in ("asc", "desc"):
        return direction
    return "asc"


def weighted_options(method: str, configured: list[bool]) -> list[bool]:
    """Weighted arms to sweep for a method: only proj honours weighted; others -> [False]."""
    if method in METHOD_SUPPORTS_WEIGHTED:
        return list(configured)
    return [False]
