# Qualitative examples: base picks a downstream consequence, SOAP recovers the source

Generated 2026-08-05 16:11 UTC at commit `e5be950` by `python -m src.analysis.qualitative`.

Frozen configs and recorded accuracies come from `exp-august/outputs/manuscript-tables/table1_main_selection.tsv` (the manuscript's
without-GT seed-window selection). Before any example is selected, the
reproduced base/SOAP step accuracies over the manuscript seed window are
asserted equal to the recorded main-table numbers (see `verification.json`).

Seed scope: **all**. Rows marked `window = no` come from a seed
OUTSIDE the manuscript triple: same frozen config, different train/val/test
partition. They are valid illustrations but are not the seeds behind the
reported number, so prefer a `window = yes` row for the paper.
Figures are deduplicated by trajectory (best-ranked seed kept); every occurrence is still listed in `candidates.tsv`.

`combined.pdf` is the manuscript-ready figure (marked ★ below).
Each cell has a paginated `gallery` contact sheet of all its examples.

| | cell | seed | traj | steps | gold | base pred | SOAP pred | offset | window |
|---|---|---|---|---|---|---|---|---|---|
| ★ | WW-AG | 2 | 95 | 10 | 5 | 7 | 5 | 2 | yes |
|  | WW-AG | 2 | 110 | 9 | 5 | 6 | 5 | 1 | yes |
|  | WW-AG | 4 | 32 | 10 | 1 | 3 | 1 | 2 | yes |
|  | WW-AG | 4 | 102 | 10 | 1 | 2 | 1 | 1 | yes |
|  | WW-AG | 3 | 88 | 10 | 0 | 9 | 0 | 9 | yes |
|  | WW-AG | 2 | 119 | 8 | 0 | 6 | 0 | 6 | yes |
|  | WW-AG | 3 | 114 | 7 | 1 | 5 | 1 | 4 | yes |
|  | WW-AG | 2 | 65 | 6 | 1 | 4 | 1 | 3 | yes |
|  | WW-AG | 2 | 58 | 6 | 1 | 3 | 1 | 2 | yes |
|  | WW-AG | 2 | 42 | 6 | 1 | 3 | 1 | 2 | yes |
|  | WW-AG | 4 | 37 | 6 | 1 | 2 | 1 | 1 | yes |
|  | WW-AG | 2 | 46 | 7 | 1 | 2 | 1 | 1 | yes |
|  | WW-AG | 4 | 9 | 5 | 1 | 2 | 1 | 1 | yes |
|  | WW-AG | 7 | 56 | 10 | 5 | 9 | 5 | 4 | no |
|  | WW-AG | 11 | 118 | 10 | 1 | 9 | 1 | 8 | no |
|  | WW-AG | 8 | 19 | 10 | 0 | 7 | 0 | 7 | no |
|  | WW-AG | 17 | 77 | 10 | 0 | 3 | 0 | 3 | no |
|  | WW-AG | 6 | 43 | 9 | 0 | 2 | 0 | 2 | no |
|  | WW-AG | 5 | 80 | 8 | 0 | 1 | 0 | 1 | no |
|  | WW-AG | 19 | 99 | 7 | 2 | 4 | 2 | 2 | no |
|  | WW-AG | 12 | 75 | 7 | 1 | 4 | 1 | 3 | no |
|  | WW-AG | 7 | 121 | 7 | 1 | 3 | 1 | 2 | no |
|  | WW-AG | 6 | 98 | 7 | 1 | 2 | 1 | 1 | no |
|  | WW-AG | 1 | 1 | 6 | 0 | 2 | 0 | 2 | no |
| ★ | WW-HC | 15 | 23 | 73 | 8 | 73 | 8 | 65 | yes |

Per example, `examples/seed-<s>_traj-<t>/` holds the verbatim trajectory JSON,
`steps.tsv` (the signal at every pipeline stage: `base -> oriented -> normalized
-> final`, with both rankings), `meta.json` (frozen config, gold/predictions,
source paths) and the figure.

Figures plot the ORIENTED base score (higher = more error-like, the quantity the
rescoring adds to) against the final SOAP score; the lower panel is the
promotion `delta = final - oriented` on its own scale. Steps where the two
curves coincide exactly are steps that are nobody's top-`w` predecessor and so
receive no correction at all -- a property of the weight construction, not a
plotting artifact.
