"""The base score: uncentered SVD geometry, with the inverse orientation folded in.

THE GEOMETRY
Transformer hidden states are extremely anisotropic: almost all of the energy of a set
of step vectors lies along a few shared directions. Fitting an SVD to the TRAIN step
matrix and taking the top right-singular vectors therefore yields a basis for what a
TYPICAL step looks like — empirically the top uncentered singular vector *is* the mean
direction (``cos(u_0, mean) ~ 0.99``) and alone carries 65-90% of all energy. So the band
``span(V[:, c] : c in [c_begin, c_end))`` is best read as "the common mode".

Decisive-error steps sit slightly OFF that common mode, i.e. they project SMALLER. The
score therefore inverts the projection:

    pi(v)  = mean_{c in band} <v, V[:,c]>^2
    S(v)   = 1 / (pi(v) + eps)                          higher = error

Folding the inverse in is what removes the orientation axis from this package entirely.
It is free for a single position: pi >= 0 and ``x -> 1/(x+eps)`` is strictly decreasing
and injective on [0, inf), so it maps ties to ties and the within-trajectory ranking is
IDENTICAL to ranking pi ascending. It is not free downstream — rescoring ADDS scores —
which is exactly why the protocol chose ``inverse`` over ``negate``.

    from main.score import fit_svd, score_steps, ens_score_steps, base_positions
"""
from __future__ import annotations

import torch

EPS = 1e-12          # the fold-in epsilon
Z_EPS = 1e-8         # ensemble z-score epsilon
ENSEMBLE_POSITION = "ens-mid3"


def fit_svd(R_train: torch.Tensor, n_components: int = 20) -> torch.Tensor:
    """Uncentered SVD of the TRAIN matrix; return the top-k right singular vectors (d,k).

    Only V is kept: the full singular spectrum existed for the whitened/weighted
    variants, which this package does not have.
    """
    _, _, Vh = torch.linalg.svd(R_train.float(), full_matrices=False)
    return Vh[:n_components].T.contiguous()


def band_bounds(n: int = 20) -> list[tuple[int, int]]:
    """All (c_begin, c_end) bands with 0 <= c_begin < c_end <= n."""
    return [(a, b) for a in range(n) for b in range(a + 1, n + 1)]


def score_steps(R: torch.Tensor, V: torch.Tensor, c_begin: int, c_end: int) -> torch.Tensor:
    """S(v) = 1 / (mean band projection + EPS). Higher = error.

    ``R`` is floated FIRST: reps are fp16 on disk and scoring in fp16 rounds the values
    enough to flip near-ties, which changes rankings. Do not "optimise" this away.
    """
    pi = (R.float() @ V[:, c_begin:c_end]).square().mean(dim=1)
    return 1.0 / (pi + EPS)


# ── the layer-band ensemble ─────────────────────────────────────────────────
def member_positions(positions: list[str]) -> list[str]:
    """Middle third of ``act/N`` positions (excludes embed, *_normed, ens-*).

    Which layer to score is the noisiest hyperparameter in the pipeline: adjacent layers
    give similar-but-not-identical rankings, and on small validation splits the argmax
    over ~10-34 positions is mostly sampling noise. Averaging over a BAND removes that
    axis instead of tuning it. The middle third is a fixed rule on the standard depth
    argument — early layers are largely lexical, the last specialise toward next-token
    prediction — and is deliberately NOT tuned, since tuning it would reintroduce the
    selection noise the ensemble exists to remove.
    """
    acts = sorted((p for p in positions
                   if p.startswith("act/") and not p.endswith("_normed")),
                  key=lambda s: int(s.split("/")[1]))
    P = len(acts)
    return acts[P // 3: P - P // 3]


def ens_score_steps(c_begin: int, c_end: int, members: list[str],
                    V_by_pos: dict[str, torch.Tensor],
                    train_R: dict[str, torch.Tensor],
                    eval_R: dict[str, torch.Tensor]) -> torch.Tensor:
    """z-averaged ensemble score over the middle-third band. Higher = error.

    Different layers produce scores on wildly different scales, so a raw mean would be
    dominated by whichever layer happens to have the largest magnitude; each member is
    z-scored first. The z-statistics come from the TRAIN split, never from the split
    being scored — using eval statistics would leak the evaluation distribution into the
    score, the same discipline as fitting the SVD on train only.

    KNOWN DIVERGENCE FROM ``src/``. There, ``orient`` was a separate downstream stage
    that ``ens-mid3`` bypassed, so members were oriented by NEGATION (``-pi``) before
    z-scoring. Here there is one base-score function, so members are the inverse-oriented
    ``1/(pi+eps)``. Every single-position number is bit-identical to ``src/``; only
    ``ens-mid3`` differs. To restore exact parity, use ``-pi`` as the member score below
    — but that reintroduces a second orientation, which the fold-in exists to avoid.
    """
    zs = []
    for pos in members:
        V = V_by_pos[pos]
        s_tr = score_steps(train_R[pos], V, c_begin, c_end)
        s_ev = score_steps(eval_R[pos], V, c_begin, c_end)
        mu, sd = s_tr.mean(), s_tr.std(unbiased=False)
        zs.append((s_ev - mu) / (sd + Z_EPS))
    return torch.stack(zs).mean(dim=0)


def base_positions(available: list[str], want="all", ensemble: bool = True) -> list[str]:
    """The swept ``position`` axis: real layer positions, plus ens-mid3 when enabled."""
    if want in ("all", None):
        positions = list(available)
    else:
        missing = [p for p in want if p not in available]
        if missing:
            raise SystemExit(f"positions {missing} not in reps (have {available})")
        positions = list(want)
    if ensemble and len(member_positions(available)) >= 2:
        positions = positions + [ENSEMBLE_POSITION]
    return positions
