# Conventions that bite

Every rule here guards a bug that fails silently. None of them can be recovered from
the code alone — that is why they are written down. Read the ones that touch the code
you are changing.

- **Two sign conventions.** `proj` means "lower = error"; the rescoring math assumes
  "higher = error". Route `proj` through `orient` first; distance scorers need nothing.
  `METHOD_DIRECTION` in `src/score/scorers.py` is the source of truth, and
  `allowed_orients` auto-restricts native-desc methods to `orient=none`. (`main/` has no
  orient axis at all: it folds the inverse into the base score, `S = 1/(pi+eps)`.)

- **`sigmoid` orientation saturates.** On large-magnitude scores `sigmoid(-s) ≈ 0` for
  every step, which collapses the ranking to a tie. The undiscounted reference is
  therefore taken from the base row's own metric, never recomputed from an oriented
  score.

- **Float before scoring.** Scorers cast back to `R.dtype`; passing fp16 `R` rounds the
  scores and flips near-ties. `score_config` floats `R` first — do not "optimise" this.

- **backprop's slot-consumption order is semantics.** `build_W` applies top-w selection
  BEFORE dropping unscored buckets (turn-0 human, GT block), so those buckets can claim
  a slot and then vanish. Deriving the row trim from the already-filtered full W would
  silently change every backprop number. The succ masks, by contrast, are DEFINED on
  the filtered full W.

- **Layer indexing differs by stage.** Activations store `embed` (tuple index 0),
  `act/k` (tuple index k+1), and `act/{N-1}_normed`. Attention stores one row per
  attention block, so `layer_range` labels index ATTENTION blocks (8 for Qwen3.5, which
  is hybrid and only exposes full-attention layers; 32 for DeepSeek), not positions.

- **Split ratios have one home.** They live in the manifest and derive the split-tag
  (`0.3/0.2/0.5 → "325"`) that names every split-tagged root. Extraction is NOT tagged,
  so overriding `split_tag` reruns analysis while reusing the expensive forward passes.

- **Protocol seeds ≠ manifest seeds** (in `src/`). The window universe is
  `triples.seeds` in the protocol config; the manifest's `seeds` list may be narrower
  (it caps score-stage sweeps). `triples.py` overrides the seed restriction when
  loading scores — keep it.

- **Metric quirks are intentional.** `agent@k` lowercases the gold role but
  `standardize_role()`s the candidates; trajectories with no gold mistake are skipped
  yet still counted in the divisor; ties resolve to the earliest step. These quirks
  define the metric — changing them silently changes every number. `main/` reproduces
  all of them; only the key naming differs (`step@1`, since there is no direction
  axis).

- **Test-selected by design.** The protocol selects per window/triple on test metrics —
  optimistic by construction; that is the protocol. The score files still carry val
  columns, and `scripts/main/pick_triple.py` has `--rule val`, if a leak-free
  convention is ever needed.

- **The selection tiebreak is float-fragile in `src/`.** Accuracies are rationals with
  small denominators, so two configs that tie mathematically can land 1-2 ulps apart
  once averaged over seeds — the addends differ even though their sum does not
  (26+24+27 vs 25+24+28, over 63). `src/`'s `select_shared` compares raw floats, so
  SUMMATION ORDER — i.e. row order in `sweep.tsv` — can decide a tie before the
  documented agent tiebreak is consulted. `main.sweep.select_config` and
  `manuscript.pick_shared_window` round the comparison key to 12 dp, which restores the
  intended rule. Two ww cells differ between the packages for exactly this reason; the
  step accuracy is identical either way and `main/`'s pick has the higher agent
  accuracy. Pinned by `test_selection_tiebreak_survives_float_noise`.

- **The GT folder names look backwards across repos.** In `../attrib-prompting`,
  `outputs/` is the WITH-GT run and `outputs-nogt/` the without-GT one. In soap,
  `outputs/` is the frozen without-GT `src/` tree. `scripts/prompting/evaluate.py`
  carries the mapping table; `scripts/prompting/verify.py` check V2 exists because this
  is the easiest thing in the repo to get backwards. Relatedly, the six
  `configs-main/*.yaml` seed blocks were once corrupted by a hand-written regex edit
  that put without-GT seeds into the `-gt` files — `scripts/tables/sync_seeds.py`
  now owns those blocks (`--check` fails loud on drift).

- **`outputs/` and `outputs-gt/` are frozen reference.** `main/` writes to
  `results-nogt/` / `results-gt/` instead, seeded from them by
  `scripts/seed_results.sh`. The two extractors produce byte-compatible artifacts, so
  seeding is a hardlink copy, not a conversion (`scripts/check_extract_parity.py`
  proves it).

- **Legacy artifacts on disk.** `outputs/<ds>/reduced/crr_*.tsv`, `sweep_v2.tsv`,
  `outputs/*/xfit-*` and `results_synthfit*.tsv` are archived pre-protocol runs whose
  only consumer (the `xfit` strand) was deleted in the 2026-08 cleanup. They have no
  reader — left in place as a record; do not regenerate. `reduced_root` itself is
  still live: the rescore stage writes `base_triples.tsv` there.
