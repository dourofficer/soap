# exp-soap — successor-side top-w for SOAP (backprop)

Self-contained experiment: **`src/` is untouched**; everything here rides the existing
pipeline through module-attribute patches (same style as exp-august's path patch).

## The change

Main-pipeline top-w trims each step's **predecessors** (rows of W) and backprop merely
transposes those matrices — selection and aggregation sit on different axes, so a step
that is nobody's top-w predecessor gets a zero column and backprop silently passes it
through. Here the weights stay full and top-w is applied per **column**: each step i
selects which w successors it collects blame from,

    strongest:  K_i = argtop-w_{t>i} w_{i,t}
    nearest:    K_i = the w smallest t>i with w_{i,t} > 0

    S~_i = s_i + gamma * ( sum_{t in K_i} w_{i,t} s_t ) / ( sum_{t in K_i} w_{i,t} )

i.e. `S~ = s + gamma (M^T s)/(M^T 1)` with `M = colmask_w(build_W(..., "all"))` —
unchanged backprop arithmetic on column-masked full matrices. At `w = "all"` both
variants coincide exactly with the recorded SOAP (the parity anchor). Full math and
caveats: docstring of `succ.py`.

## Protocol

Evaluation is exp-august's **protocol 2** (seed-window / "triples"): per consecutive
3-seed window over seeds 1..20, the base config is the window's recorded SVD (proj)
choice (read from exp-august's `triples_selection.tsv`), and the rescore hyperparameters
(`layer_range, gamma, w`) are selected over the same window with `orient=inverse` and
`score_norm=none` fixed — exactly how SOAP (full) was evaluated, so rows are directly
comparable. Unlike exp-august, the sweep grid is trimmed to those fixed values
(selection never reads the other orient/score_norm combinations).

## Running (from repo root)

```
pytest exp-soap/test_succ.py                                        # invariants (CPU)
python exp-soap/run_rescore_triples.py --config exp-soap/configs/<ds>.yaml   # GPU sweeps
python exp-soap/check_parity.py        --config exp-soap/configs/<ds>.yaml   # w=all == SOAP
python exp-soap/triples_table.py       --config exp-soap/configs/<ds>.yaml   # tables
```

Datasets: `ww`, `traceelephant`, `correct-error` (with subset macro-average rows),
`correct-full`. Both variants run in ONE `run_pair` pass per cell (the per-seed
representation loads and SVD refits dominate; masking is cheap), so sweeps land in a
single `outputs/<ds>/rescore/<tag>/<model>/<subset>/sweep_triples-succ.tsv` with both
names in the `strategy` column; tables in `outputs/<ds>/tables/<tag>/
triples_succ_{selection,summary}.tsv` (summary: SVD / SOAP / succ-strong / succ-near
side by side with diffs vs SOAP).

## Files

- `succ.py` — column masks, `SuccWCache` (drop-in for `src.rescore.weights.WCache`),
  and the per-step reference loop the vectorized path is tested against.
- `run_rescore_triples.py` — builds the same union base tables as
  exp-august/run_rescore_triples.py, then runs `src.rescore.run.run_pair` once for
  both variants with `WCache`/`STRATEGIES` patched.
- `triples_table.py` — copies SVD (proj) / SOAP (full) from exp-august's selection,
  selects the succ variants the same way, writes the comparison.
- `check_parity.py` — asserts every `w = all` succ row equals exp-august's frozen
  `strategy = backprop, w = all` row.
- `test_succ.py` — invariants: vec == reference loop, gamma=0 identity, w=all parity,
  sink pass-through, strongest != nearest, mask correctness, GT-sentinel transparency.
