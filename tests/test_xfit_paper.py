"""CPU invariants for the xfit paper setting (alignment, tags, config, reductions).

Alignment tests read the in-tree corpora (data/ww, data/synthetic, datagen/pools) and
skip if absent, so the file still passes on a source-only checkout.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.xfit import align, prov
from src.xfit.common import (load_config, source_tag, paper_cfg, paper_jobs,
                             paper_seeds, setting, REAL_SOURCE)
from src.xfit.legacy import best_per_group

REPO = Path(__file__).resolve().parents[1]
HAVE_DATA = (REPO / "data/ww/algorithm-generated").is_dir() \
    and (REPO / "data/synthetic/captain-qwen9b/filename_map.csv").is_file() \
    and (REPO / "datagen/pools/data/gaia.jsonl").is_file()

POOLS = ["gaia", "assistantbench"]


def _cfg(*overrides):
    return load_config(["setting=paper", *overrides])


# ── tag routing ──────────────────────────────────────────────────────────────
def test_source_tag_legacy_unchanged():
    assert source_tag("captain-qwen9b") == "xfit-captain-qwen9b"
    legacy = load_config()
    assert source_tag("captain-qwen9b", legacy) == "xfit-captain-qwen9b"


def test_source_tag_paper_prefix():
    cfg = _cfg()
    assert source_tag("captain-qwen9b", cfg) == "xfitp-captain-qwen9b"
    assert source_tag(REAL_SOURCE, cfg) == "xfitp-real"
    assert source_tag("x", _cfg("paper.tag_prefix=zz")) == "zz-x"


# ── config shape ─────────────────────────────────────────────────────────────
def test_paper_cfg_defaults_and_validation():
    pc = paper_cfg(_cfg())
    assert pc["convention"] == "test" and pc["seed_policy"] == "main-top3"
    assert pc["ensemble"] is True and pc["table_scorers"] == ["proj"]
    with pytest.raises(SystemExit):
        paper_cfg(_cfg("paper.convention=bogus"))
    with pytest.raises(SystemExit):
        setting(load_config(["setting=bogus"]))


def test_paper_seeds_literal():
    cfg = _cfg("paper.seed_policy=literal", "paper.seeds=[4,2]")
    assert paper_seeds(cfg, "ww") == [4, 2]


def test_paper_jobs_real_first_per_target():
    jobs = list(paper_jobs(_cfg()))
    # every target contributes real + q9 + q35, real first.
    targets = [(d, s) for src, d, s in jobs if src == REAL_SOURCE]
    assert len(targets) == len(set(targets)) == 4
    for d, s in targets:
        i = jobs.index((REAL_SOURCE, d, s))
        assert jobs[i + 1][0].endswith("qwen9b") and jobs[i + 2][0].endswith("qwen35b")
    assert not [j for j in jobs if j[1].startswith("correct")]   # CE deferred


def test_config_hash_tracks_knobs():
    a, b = prov.config_hash(_cfg()), prov.config_hash(_cfg("paper.convention=val"))
    assert a == prov.config_hash(_cfg()) and a != b
    assert a != prov.config_hash(_cfg("gammas=[0.5]"))


# ── alignment ────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not HAVE_DATA, reason="in-tree corpora not present")
def test_align_deterministic_count_and_dedup():
    stems = ["1", "2", "3", "5", "8", "13", "21"]
    f1, r1 = align.fit_files("ww", "algorithm-generated", 7, "captain-qwen9b", stems, POOLS)
    f2, r2 = align.fit_files("ww", "algorithm-generated", 7, "captain-qwen9b", stems, POOLS)
    assert f1 == f2 and r1["files"] == r2["files"]
    assert len(f1) == len(stems) == r1["n_matched"] + r1["n_filled"]
    assert len(set(f1)) == len(f1)


@pytest.mark.skipif(not HAVE_DATA, reason="in-tree corpora not present")
def test_align_matched_files_carry_train_questions():
    stems = ["0", "10", "20", "30", "40"]
    _, rep = align.fit_files("ww", "algorithm-generated", 1, "captain-qwen35b", stems, POOLS)
    keys = align.subset_keys("ww", "algorithm-generated", tuple(POOLS))
    smap = align.synth_map("captain-qwen35b", tuple(POOLS))
    inv = {f: k for k, f in smap.items()}
    for stem, f in rep["matched"].items():
        assert inv[f] == keys[stem]        # same question, different generator


@pytest.mark.skipif(not HAVE_DATA, reason="in-tree corpora not present")
def test_align_fill_never_touches_subset_questions():
    # captain-qwen9b has datagen misses on ww-AG, so fills occur for full train splits.
    from src.stores import list_rep_files, split_files
    files = list_rep_files(REPO / "outputs/ww/activations/qwen3.5-9b/algorithm-generated")
    stems = [Path(f).stem for f in
             split_files(files, {"train": .3, "val": .2, "test": .5}, 1)["train"]]
    _, rep = align.fit_files("ww", "algorithm-generated", 1, "captain-qwen9b", stems, POOLS)
    keys = {k for k in align.subset_keys("ww", "algorithm-generated", tuple(POOLS)).values() if k}
    smap = align.synth_map("captain-qwen9b", tuple(POOLS))
    inv = {f: k for k, f in smap.items()}
    for f in rep["filled_files"]:
        assert inv[f] not in keys          # fill excludes ALL subset questions


@pytest.mark.skipif(not HAVE_DATA, reason="in-tree corpora not present")
def test_align_traceelephant_text_join():
    stems = ["0", "1", "2", "3"]
    _, rep = align.fit_files("traceelephant", "magentic", 2, "magentic-qwen9b", stems, POOLS)
    assert rep["n_matched"] + rep["n_filled"] == 4


def test_align_rejects_unregistered_dataset():
    with pytest.raises(SystemExit):
        align.subset_keys("correct-full", "magentic", tuple(POOLS))


# ── reduction winner invariance under grid thinning ──────────────────────────
def test_best_per_group_stable_under_method_and_k_thinning():
    """The paper score grid drops norm/k=3 rows the main grid has. With shared rows
    equal-valued and in the same relative order, per-method winners must not move —
    including under ties, which resolve by (stable) row order."""
    rows, order = [], 0
    for pooling in ("mean", "last"):
        for pos in ("act/3", "act/7", "ens-mid3"):
            for method in ("proj", "resid", "angres", "norm_l2"):
                for k in (1, 3):
                    if pos == "ens-mid3" and method == "norm_l2":
                        continue
                    # deliberate heavy ties: metric depends only on (method, pos)
                    v = (hash((method, pos)) % 7) / 10
                    rows.append({"pooling": pooling, "position": pos, "method": method,
                                 "k": k, "seed": 1, "order": order,
                                 "step_acc_test": v, "agent_acc_test": v,
                                 "step_acc_val": v, "agent_acc_val": v})
                    order += 1
    full = pd.DataFrame(rows)
    full_k1 = full[full["k"] == 1]
    thin = full_k1[full_k1["method"].isin(["proj", "resid", "angres"])]
    for metrics in (["step_acc_test", "agent_acc_test"], ["step_acc_val", "agent_acc_val"]):
        w_full = best_per_group(full_k1, metrics, ["seed", "method"], 1)
        w_thin = best_per_group(thin, metrics, ["seed", "method"], 1)
        for m in ("proj", "resid", "angres"):
            a = w_full[w_full["method"] == m].iloc[0]["order"]
            b = w_thin[w_thin["method"] == m].iloc[0]["order"]
            assert a == b, f"winner moved for {m} under {metrics}"


# ── tag-config guard ─────────────────────────────────────────────────────────
def test_tag_config_guard(tmp_path):
    cfg = _cfg()
    prov.write_tag_config(tmp_path, cfg)
    prov.ensure_tag_config(tmp_path, cfg)                     # same knobs: fine
    changed = _cfg("paper.seeds=[9]", "paper.seed_policy=literal")
    with pytest.raises(SystemExit):
        prov.ensure_tag_config(tmp_path, changed)
    prov.ensure_tag_config(tmp_path, changed, force=True)     # explicit override
    prov.ensure_tag_config(tmp_path / "missing", cfg)         # grandfathered: warn only
