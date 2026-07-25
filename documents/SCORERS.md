# Base scorers

The base score `S(s_t)` rates each step's anomaly from a single pooled hidden-state
vector `v_t`. This document explains every scorer in `src/score/scorers.py`, the geometry
they all exploit, and what the computed results say about how they differ.

> **Note on `maha`.** The `maha` scorer is documented below for completeness but is
> **excluded from the default sweep** (dropped from `methods`/`headline` in the configs);
> it tracked `resid` closely and is no longer of interest. Its function and invariant test
> remain in the code. Add it back to a `methods:` list to sweep it again.

Convention throughout: a base score ranks steps **within one trajectory**; the predicted
decisive-error step is the argmax after orientation. Each scorer declares a *native
direction* (`METHOD_DIRECTION`) — `asc` = "lower is error", `desc` = "higher is error".

---

## 1. The geometry all scorers exploit

Fit an SVD to the **train**-split step matrix `G` (rows = per-step vectors `g_i`). Two
theorems make the top right singular vectors `u_1, u_2, …` a basis for "what a normal
step looks like":

- **Variational (Courant–Fischer).** `u_1` maximises the population energy
  `Σ_i ⟨g_i, u⟩²` over unit directions, and each `u_k` maximises it among directions
  orthogonal to `u_1 … u_{k−1}` — by construction, the directions in which the step
  population's energy concentrates.
- **Best-fit subspace (Eckart–Young).** `span(u_1 … u_k)` minimises
  `Σ_i ‖g_i − P g_i‖²` over all k-dimensional subspaces: the best rank-k summary of the
  train step cloud in least squares. `resid` (§3) is exactly a new step's squared
  reconstruction error under this train-optimal subspace.

"Typical" is carried by the data, not by the factorization: the SVD is a least-squares
object (each step contributes `∝ ‖g_i‖²`, nothing downweights outliers), so the basis
describes typical steps only because the train split is overwhelmingly normal steps and
no small subset dominates the total energy.

Two measured facts (from `src/analysis/geometry.py`,
`outputs/<ds>/analysis/325/geometry/`) hold on every dataset and both proxy models:

- **`cos(u_1, mean-direction) = 0.986–0.999`** on the *uncentered* fit — the top
  singular vector is, to within ~10°, the mean direction. So the leading component is
  the population's **common mode** (its location), and only `u_2, u_3, …` encode
  variation about it. This is a consequence of anisotropy, *not* an SVD identity: `u_1`
  is the top eigenvector of the second moment `GᵀG = n(μμᵀ + C)`, which coincides with
  the mean direction `μ̂` only when `Cμ ∥ μ`; in general
  `cos²(u_1, μ̂) ≥ (n‖μ‖² − σ_2²) / (σ_1² − σ_2²)`, so near-alignment follows whenever
  the mean's energy dominates the covariance — which is exactly the next fact.
- **`energy_top1_frac = σ_1² / Σ_c σ_c² ≈ 0.65–0.90`** at the mid-to-late positions the
  sweeps favour (DeepSeek's middle layers dip to a median ≈ 0.55; the isotropic value
  would be `1/d ≈ 2e-4`). A single direction holds most of the total energy — extreme
  anisotropy (the "narrow cone"). Adjacent layers/positions give similar-but-not
  identical bases, which is why *which layer* is the noisiest hyperparameter.

**Why centering is usually unnecessary.** If `u_1` were exactly `μ̂`, then for any
`u ⊥ μ̂` the uncentered second moment equals the covariance
(`Σ_i ⟨g_i, u⟩² = Σ_i ⟨g_i − μ, u⟩²`), so `u_2, u_3, …` would be exactly the centered
principal directions. With `cos ≥ 0.986` they are nearly so: the uncentered basis is
approximately `{mean direction} ∪ {centered PCs}`, and the uncentered band grid can
express what the centered one can.

Decisive-error steps sit slightly **off** this common mode. That one fact can be scored
from two sides of the Pythagorean identity (exact, since the `u_c` are orthonormal —
it holds for any band `[c_begin, c_end)` and for out-of-sample steps):

```
‖ṽ‖²  =  ‖P_band ṽ‖²  +  r²          (P_band = projection onto span(u_c : c∈[c_begin,c_end)))
        └─ band energy ─┘   └ residual ┘
```

- **similarity** — how much of the step lies *inside* the common mode (`proj`). Error steps
  have *less*, so this is "lower = error" (`asc`) and must be flipped before use.
- **distance** — how much lies *outside* it (`resid`, `angres`, `maha`). Error steps have
  *more*, so this is natively "higher = error" (`desc`) and needs **no orientation**.

The identity ties the two *energies*, not the two *rankings*: `resid = ‖ṽ‖² − |band|·proj`,
so `proj`-asc and `resid`-desc agree only where step norms are constant. The exactly
complementary pair is `(cos²θ, sin²θ)`, which sum to 1 — `angres`-desc *is* norm-normalized
`proj`-asc. That the unnormalized forms still land within a few points of each other (§7)
is an empirical fact, explained by the norm carrying no signal: per-step `‖v‖` on its own
ranks the gold step at chance or worse. So the distance form buys orientation-freedom at
~no cost, and the normalized distance (`angres`) is the principled member.

The anisotropy number is also the scorer's baseline, via an exact identity: summing the
Pythagorean identity over train gives `Σ_train resid(0:k) = Σ_{c>k} σ_c²`, i.e. the
norm-weighted train mean of `angres` equals `1 − energy_topk_frac`. A typical step
already has ~10–35% of its energy off the common mode; the gold step is found because
its `sin²θ` exceeds that baseline *within its trajectory*.

Common config knobs (swept per scorer): the band `[c_begin, c_end)` over the top-20
components; `centered` (subtract the train mean before scoring); `pooling` (mean/last);
`position` (which layer). `weighted` (σ-scaling) exists only for `proj`. Note the scores
depend on the band only through `span(u_c : c∈band)` — band energy is invariant to
rotations within the band — but a band with `c_begin > 0` also depends on where the
spectrum splits at `c_begin`, which is gap-sensitive: one reason band choice, like layer
choice, behaves like a noisy hyperparameter.

---

## 2. `proj` — mean squared projection (similarity, asc)

```
proj(v) = (1/|band|) · Σ_{c∈band} ⟨ṽ, u_c⟩²          native: asc  (lower = error)
```

Mean squared projection of the (optionally centered) step onto the band singular vectors —
the step's energy inside the common mode, per band component (unnormalized: it retains the
step's magnitude). Error steps project *smaller*, so `proj` is "lower = error" and the
rescoring stage must orient it first (negate/inverse/sigmoid). This is the legacy base
scorer; the faithful `results_table.tsv` uses it.

`weighted=True` scales each squared projection by its singular value σ_c before averaging
(HaloScope-style), up-weighting high-variance directions. Configs never enable it
(unweighted wins); it is kept for reference and reproducibility only.

## 3. `resid` — residual energy (distance, desc)

```
resid(v) = ‖ṽ‖² − Σ_{c∈band} ⟨ṽ, u_c⟩²  = r²          native: desc (higher = error)
```

The energy left **off** the band subspace — the exact complement of `proj`'s band energy,
and the step's squared reconstruction error under the train-optimal subspace
(Eckart–Young, §1). Natively "higher = error", so no orientation. Because it keeps the raw
magnitude, it mixes "points off-manifold" with "is simply large", which (§7) costs it
against the normalized form.

## 4. `angres` — angular residual / sin²θ (distance, desc, norm-free)

```
angres(v) = r² / (‖ṽ‖² + ε)  = sin²θ ∈ [0,1]          native: desc (higher = error)
```

The residual as a *fraction* of the step's norm — the squared sine of the angle between the
step and the common-mode subspace. This removes the norm entirely and measures pure
alignment; equivalently, it is the exact norm-free dual of `proj` (`cos²θ + sin²θ = 1`, §1).
It is the principled distance scorer: bounded, orientation-free, and (§7) the one that
matches or beats `proj`.

## 5. `maha` — PPCA whitened anomaly (distance, desc)

```
maha(v) = Σ_{c∈band} ⟨ṽ,u_c⟩²/σ_c²  +  r²/σ̄²_resid    native: desc (higher = error)
```

The probabilistic reading: model steps as isotropic noise around the band subspace and score
a step's negative log-density. Inside the band each direction is **whitened** by its own
train variance σ_c² (a deviation along a low-variance direction is more surprising); outside,
the residual is divided by σ̄²_resid, the mean of the **off-band** σ² over the full train
spectrum (PPCA's "average discarded eigenvalue" — this is why `fit_one` keeps the full
spectrum, not just the top-k). The global 1/(n−1) cancels under per-trajectory ranking and
is omitted. (Strictly, the Gaussian log-density reading applies to the *centered* fit; on
the uncentered fit the same formula whitens the second moment rather than the covariance.)
In practice it tracks `resid` and both trail `angres` (§7) — whitening does not help here,
because the signal is angular, not variance-scaled.

## 6. `norm_l1` / `norm_l2` — magnitude baselines (both directions)

```
norm_p(v) = ‖ṽ‖_p                                     native: both (asc + desc emitted)
```

The raw L1/L2 norm of the (optionally centered) step; the band is ignored (`c_begin=c_end=0`).
Emitted in both directions since neither is theoretically preferred. These are baselines to
show that **magnitude alone is not the signal** (§7).

---

## 7. What the computed results say

### 7.1 Accuracy — best-config-per-seed, mean step@1 over seeds

`val` = leak-free (config chosen on val); `test` = optimistic (chosen on test). Source:
`outputs/<ds>/reduced/325/<model>/<subset>/base_by_method_test.tsv`. Two numbers = qwen3.5-9b
/ deepseek-8b, **test** convention (the val numbers are noisy on the small subsets):

| dataset / subset | proj | angres | resid | maha | norm_l2 | norm_l1 |
|---|---|---|---|---|---|---|
| correct-full / magentic | **.562/.563** | .556/.558 | .541/.554 | .541/.554 | .504/.434 | .394/.435 |
| ww / algorithm-generated | **.419/.443** | .403/.405 | .330/.321 | .324/.319 | .330/.392 | .321/.371 |
| ww / hand-crafted | **.303/.283** | .259/.269 | .259/.290 | .262/.286 | .255/.228 | .217/.234 |
| traceelephant / magentic | .237/**.302** | **.257**/.261 | .254/.239 | .252/.241 | .196/.213 | .196/.198 |
| traceelephant / captain | .330/**.372** | **.342**/.367 | .298/.291 | .298/.291 | .244/.291 | .230/.260 |
| correct-error / arc | .827/**.873** | .851/.851 | **.857**/.844 | — | .809/.818 | — |
| correct-error / gaia | .707/.707 | **.773/.720** | .693/.680 | — | .560/.533 | — |
| correct-error / hotpot | .518/**.537** | **.548**/.528 | .531/.527 | — | .450/.391 | — |
| correct-error / math500 | **.675**/.654 | .662/**.671** | .624/.658 | — | .641/.603 | — |
| correct-error / mmlu_pro | .783/.790 | .783/**.797** | **.790**/.790 | — | .754/.768 | — |
| correct-error / musique | .479/.483 | **.504**/.481 | .494/**.500** | — | .400/.321 | — |
| correct-error / wikimqa | **.513**/.505 | .507/**.508** | .469/.470 | — | .439/.363 | — |

(maha is dropped from the default sweep — see note below; correct-error numbers are over
its 3 seeds, matching the legacy experiment.)

Reading:
- **`proj` ≈ `angres`, and `angres` pulls ahead on the higher-signal datasets.** They are
  within ~1–4 points everywhere, `angres` *wins* on both traceelephant subsets (qwen), and
  on **correct-error** it beats `proj` outright on gaia (.773 vs .707), hotpot, musique, and
  ties/edges it on math500 / mmlu_pro / wikimqa. Since `angres` needs no orientation, it is a
  free, cleaner replacement for `proj` — and where the geometry is strong, a better one.
- **`resid` is competitive on correct-error** (wins arc/deepseek, mmlu_pro/qwen) but still
  trails `angres` on the low-signal ww/traceelephant subsets; `maha` tracked `resid` closely
  before it was dropped. Both keep a magnitude/variance scale that `angres` divides out.
- **`norm_l1/l2` are the weakest** on the geometry-rich datasets, and only look competitive on
  ww-algorithm-generated, the weakest-signal subset (below), where everything is near noise.

### 7.2 Which configuration each scorer selects

Modal winning `(c_begin, c_end)` band and centering fraction, pooled over all reduced tables:

| scorer | top bands | centered-frac |
|---|---|---|
| proj | (0,2), (0,5), (0,6) | 0.23 |
| angres | (0,2), (2,4), (1,4) | 0.37 |
| resid | (0,10), (0,11), (0,4) | 0.51 |
| maha | (1,6), (2,8), (0,9) | 0.49 |
| norm_l1/l2 | (0,0) — band ignored | 0.31 |

`proj`/`angres` favour a **small top band** (2–5 components — the common mode) and mostly
**uncentered** fits (consistent with `u_1 ≈ mean`, so the raw top components already encode
the mean plus the leading centered PCs — §1). `resid`/`maha` prefer **wider** bands and
center more often — they need to "explain away" more of the vector before the residual is
informative, which is another symptom of their norm-sensitivity.

### 7.3 Geometry — why alignment, not magnitude, is the signal

Per-step rank-AUC of each raw quantity for ranking the gold step (uncentered, mean pooling,
band 0:5; 0.5 = chance, higher = more discriminative):

| dataset / subset | ‖v‖ (norm) | band energy (proj) | resid | sin² (angres) |
|---|---|---|---|---|
| correct-error / arc (qwen) | 0.36 | 0.23 | 0.92 | **0.92** |
| correct-error / gaia (qwen) | 0.44 | 0.36 | 0.78 | **0.74** |
| correct-full / magentic (qwen) | 0.46 | 0.38 | 0.79 | **0.79** |
| correct-full / magentic (deepseek) | 0.43 | 0.28 | 0.78 | **0.81** |
| traceelephant / captain (deepseek) | 0.25 | 0.23 | 0.65 | **0.73** |
| ww / hand-crafted (qwen) | 0.42 | 0.40 | 0.63 | **0.68** |
| ww / algorithm-generated (qwen) | 0.47 | 0.48 | 0.49 | 0.51 |

- **`norm` ≈ 0.25–0.54** — at or below chance. Magnitude alone does not locate the error;
  often it is mildly *anti*-correlated. This is the empirical death of the `norm_*` scorers.
- **band energy < 0.5** — because it is the *similarity*: error steps project *less*, so as a
  raw "higher = error" quantity it looks anti-correlated. Negating it (what `proj`+orient
  does) recovers the signal; equivalently, its complement `resid`/`sin²` is directly
  discriminative.
- **`sin²` (angres) is the strongest single quantity** (up to 0.81), edging `resid`
  everywhere — normalizing by the norm helps exactly because norm is noise.
- **ww/algorithm-generated is the exception**: every quantity sits near 0.5. These
  trajectories are short (~9 steps) and few (126), and the mid-band geometry barely separates
  the gold step — which is why `proj`'s extra band/centering freedom edges `angres` there,
  and why absolute accuracy on that subset is low for all scorers.

### 7.4 Synthesis

`proj` and the distance family are two readings of one geometry, related by
`‖v‖² = band energy + residual`. The decisive-error step is distinguished by pointing *away*
from the common-mode subspace (high angular residual), **not** by its magnitude:

- `proj` scores the similarity side and needs orientation; `angres` scores the normalized
  distance side and does not. They are empirically interchangeable in accuracy, so `angres`
  is the recommended default — it removes the orientation hack (and the sigmoid-saturation
  pitfall) at ~zero cost, sometimes with a small gain.
- `resid`/`maha` are inferior to `angres` because they retain a magnitude/variance scale that
  is not where the signal lives.
- `norm_l1/l2` confirm the negative control: magnitude alone is uninformative.

Practical default: **`angres`** (orientation-free, best-or-tied); keep `proj` for faithful
comparison to the legacy tables; treat `norm_*` as baselines only. All six run through the
identical grid, so this comparison is reproducible from the per-method reduced tables and the
geometry TSVs cited above.
