# Testing

Two layers guard the numbers: a CPU test suite that pins the algorithms, and a set of
parity checkers that pin the artifacts and the end-to-end results. Run the suite before
any commit that touches `src/` or `main/`; run the relevant checker before trusting a
regenerated artifact.

## The test suite

`pytest tests/ -q` runs standalone on CPU.

**`tests/test_invariants.py` is `src/`'s correctness story:** scorer identities, `γ=0`
identity for all three strategies, `orient=none` identity, the backprop transpose by
hand, vec-vs-reference-loop for the succ variants, `w="all"` coincidence of all
strategies, sink pass-through, strongest≠nearest divergence, column-mask correctness,
batched-vs-loop metrics on tie-saturated inputs, and the with-GT context block (pinned
GT prefix, truncation, `build_W` sentinel drop).

**`tests/test_main.py` is `main/`'s**, in two groups. Group A pins it to `src/` where
they must agree: the seed→partition mapping (also frozen as a golden literal), keeper
row order, the metric quirks, the base score bit-for-bit, `fit_svd`, `build_W`
including the GT-sentinel slot consumption, the column masks, the context spans, and
the selection rule. Group B covers the invariants that survive `src/` being retired,
plus an `ast` walk asserting `main/` never imports `src/`, a check that every
`configs-main/*.yaml` declares a 3-seed triple for each of its subsets, and
`test_selection_tiebreak_survives_float_noise` (see the float-fragility entry in
[CONVENTIONS.md](CONVENTIONS.md)).

## Parity and verification checkers

- **`scripts/check_extract_parity.py`** (GPU) proves `main/extract.py` reproduces the
  reference extractions bit-for-bit — needed because the bulk tensors are COPIED into
  the results trees by `scripts/seed_results.sh` rather than recomputed.
- **`scripts/check_gt_parity.py`** (read-only) asserts the with-GT extraction mirrors
  the without-GT one file-for-file and step-for-step. The splits are derived from the
  file SET, so a mismatch silently shifts every split. Run after GT extraction, before
  scoring.
- **`scripts/check_main_parity.py`** proves `main/` selects the same configs and
  reports the same numbers as `src/` for every cell (the two grids are identical by
  construction). The one known divergence is `ens-mid3`, so it re-selects with those
  rows excluded and reports ens-mid3 wins separately: a finding, not a failure.
- **`scripts/main/check_sweep_repro.py`** compares the triple sweep against known
  results before you trust a 6-hour run: the new sweep must reproduce all 44 cells of
  the five pre-existing triples exactly.
- **`scripts/prompting/verify.py`** runs four checks on the prompting evaluation, each
  guarding a mistake that would produce wrong numbers in silence — above all V2, the
  cross-repo GT folder swap (see [CONVENTIONS.md](CONVENTIONS.md)).
- **`scripts/tables/sync_seeds.py --check`** fails loud if the seed blocks in
  `configs-main/` drift from the recorded triple picks + manual overrides.
