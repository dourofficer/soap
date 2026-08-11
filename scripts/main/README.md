# Sweeping seed triples

`main/` runs one seed triple at a time. These scripts run many.

The seeds choose the train/val/test split. They change nothing else. So a new triple needs
no new GPU extraction. It only re-scores the activations already on disk.

## The seed selection protocol

A triple is three consecutive seeds: (1,2,3), then (2,3,4), and so on.

We sweep 48 triples for `ww` and `traceelephant`, from seed 1 to seed 50. We sweep 18 for
`correct-error`, from seed 1 to seed 20. One `correct-error` triple costs five times more
than the others, and 18 triples already cover the span the older protocol used.

For each triple, `main/` picks the best config. Then we pick the best triple.

The default rule is `sum`:

1. Add the SOAP step accuracy of both backbones.
2. The highest sum wins.
3. If two triples tie, the higher agent sum wins.
4. If they still tie, the earlier triple wins.

Two other rules are available:

- `sum-diff` prefers a triple where SOAP beats the base scorer on **both** backbones.
- `val` reads the validation accuracy instead of the test accuracy.

`ww` and `traceelephant` choose one triple per subset. `correct-error` chooses one triple
for all seven subsets, because we report the average over those subsets.

**Read the margin before you trust a pick.** The gap between the best and second-best
triple is usually below 0.02. The gap between the best and the worst is much larger — up
to 0.15. So the seeds matter a lot, but the exact winner does not.

## How to run the code

Run these from the repo root.

```bash
# 1. Sweep every triple. Writes results-sweep/. Takes about 6 hours on 8 GPUs.
python scripts/main/sweep_triples.py --workers 24 --plan 'correct-error:1-20'

# 2. Gather the results into two files. Takes a few minutes.
python scripts/main/collect.py

# 3. Choose the best triple. Takes seconds.
python scripts/main/pick_triple.py --rule sum
```

You can stop step 1 and start it again. It skips the work it already finished.

Two commands help you before and after:

```bash
python scripts/main/sweep_triples.py --dry-run     # show the plan and the time estimate
python scripts/main/check_sweep_repro.py           # compare the sweep to known results
```

Run `check_sweep_repro.py` before you trust a long sweep. Five triples already exist from
an earlier run. The new sweep must reproduce all 44 of their cells exactly. Start with two
triples, check them, and only then sweep the rest.

Change the rule and run step 3 again. It reads one small file, so it costs seconds. You do
not need to sweep again.

## How to read the result

Three files land in `results-sweep/`.

### `best_triples_<rule>_<metric>_<strategy>.tsv` — the answer

One row per cell. A cell is one dataset, one subset, and one GT setting.

| column | meaning |
|---|---|
| `with_gt` | `True` if the run saw the gold answer |
| `dataset`, `key` | the cell. `key` is the subset, or the dataset for `correct-error` |
| `triple`, `seeds` | the winning triple, and its three seeds |
| `score` | the objective: SOAP step accuracy added across both backbones |
| `margin` | how far ahead of the second-best triple. Small means the pick is fragile |
| `runners_up` | the next three triples, best first |
| `n_beat` | how many backbones beat the base scorer, out of 2 |
| `diff_sum` | how much SOAP beats the base scorer, added across backbones |

`pick_triple.py` also prints a `seeds:` block. Paste it into `configs-main/<ds>.yaml`.

### `selections_all.tsv` — the chosen config for every triple

5,088 rows. One row per triple, backbone, subset, and method.

`row` names the method: `svd` is SOAP without rescoring; `backprop`, `succ-strong` and
`succ-near` are the three rescoring strategies. `position`, `c_begin` and `c_end` describe
the base score. `layer_range`, `gamma` and `w` describe the rescoring. The last four
columns hold the test and validation accuracies.

Use this file to see how the chosen hyperparameters move as the seeds change.

### `grid_all.parquet` — every config for every triple

6.8 million rows in 63 MB. Same columns, but for **all** configs, not just the winners.
Each row averages the three seeds of its triple.

Use this file to change how a config is chosen. You might want a different tiebreak, the
top five instead of the top one, or validation instead of test. This file saves you from
reading the 10 GB of raw sweep tables.

Note two column names. The GT flag is `with_gt`, not `gt`, because `df.gt` is already a
pandas method. The `w` column holds text, because it mixes `1` to `5` with `all`.
