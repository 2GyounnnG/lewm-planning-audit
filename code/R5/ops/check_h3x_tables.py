"""Independent H3X statistics from all observed old/new open and closed rows."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

SEEDS = (103201, 103202, 103203)
BASE = ("H0",) + tuple(f"REFIT_{s}" for s in SEEDS)
NEW = tuple(f"H3X_{s}" for s in SEEDS)
R3_MEAN = "FIXED3_R3_MEAN_NOT_ENSEMBLE"
H3X_MEAN = "FIXED3_H3X_MEAN_NOT_ENSEMBLE"
STREAMS = ("R3_ORIGINAL", "R4_ALT_CEM_1", "R4_ALT_CEM_2")
LABEL = "H3X_EXPLORATORY_AFTER_TRIGGER_DEFECT"
STATS_SHA = "e110a936b19622a659812f6b1b1d01a82245fc83ac2381434762dd3a5ae3320b"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    with Path(p).open(newline="") as f:
        return list(csv.DictReader(f))


def check(folder, old_open, source):
    assert sha(source) == STATS_SHA
    spec = importlib.util.spec_from_file_location("r5_h3x_independent_statistics", source)
    stats = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stats)
    complete = json.loads((folder / "COMPLETE.json").read_text())
    assert complete["status"] == "COMPLETE" and complete["label"] == LABEL
    assert complete["preset_H3_status"] == "NOT_TRIGGERED"
    for item in complete["files"]:
        p = folder / Path(item["path"]).name
        assert p.stat().st_size == item["bytes"] and sha(p) == item["sha256"]
    old_rows = [r for r in read(old_open) if r["task"] == "reacher"]
    new_rows = read(folder / "H3X_OPEN_LOOP_RAW.csv")
    assert len(old_rows) == 2400 and len(new_rows) == 1800
    open_values = {}
    for r in old_rows:
        key = r["case_id"], r["model"], r["history_kind"], int(r["horizon_raw"]) // 5
        assert key not in open_values
        open_values[key] = float(r["latent_mse"])
    for r in new_rows:
        assert r["label"] == LABEL and r["preset_H3_status"] == "NOT_TRIGGERED"
        assert int(r["checkpoint_step"]) == 30000 and r["valid"] == "True"
        key = r["case_id"], r["arm"], r["history_kind"], int(r["horizon_macro"])
        assert key not in open_values
        open_values[key] = float(r["latent_mse"])
    ids = sorted({r["case_id"] for r in old_rows})
    assert len(ids) == 100
    assert set(open_values) == {(c, a, h, k) for c in ids for a in BASE + NEW for h in ("H_POLICY", "H_REAL3") for k in (1, 2, 5)}
    old_closed = read(folder / "PAIRED_R4_CONTROL_RAW.csv")
    new_closed = read(folder / "H3X_CLOSED_LOOP_RAW.csv")
    assert len(old_closed) == 1200 and len(new_closed) == 900
    closed_values = {}
    for r in old_closed + new_closed:
        assert r["label"] == LABEL and r["preset_H3_status"] == "NOT_TRIGGERED"
        key = r["case_id"], r["arm"], r["stream"]
        assert key not in closed_values
        assert float(r["success"]) in (0, 1)
        closed_values[key] = r
    assert set(closed_values) == {(c, a, s) for c in ids for a in BASE + NEW for s in STREAMS}
    def with_means(values):
        values[R3_MEAN] = np.stack([values[a] for a in BASE[1:]]).mean(0)
        values[H3X_MEAN] = np.stack([values[a] for a in NEW]).mean(0)
        return values
    vectors = {}
    for hist in ("H_POLICY", "H_REAL3"):
        for h in (1, 2, 5):
            vectors["OPEN_LOOP", "LATENT_MSE_FP32", hist, str(h)] = with_means({a: np.array([open_values[c, a, hist, h] for c in ids]) for a in BASE + NEW})
    for metric, key in (("SUCCESS_RATE", "success"), ("EXECUTED_RAW_STEPS", "executed_raw_steps"), ("REPLAN_CALLS", "replan_calls"), ("WALL_SECONDS_DESCRIPTIVE_ONLY", "wall_seconds")):
        vectors["CLOSED_LOOP", metric, "OFFICIAL_DEFAULT", ""] = with_means({a: np.array([np.mean([float(closed_values[c, a, s][key]) for s in STREAMS]) for c in ids]) for a in BASE + NEW})
    ix = stats.case_indices("reacher", ids)
    maximum = 0.0
    def equal(actual, expected):
        nonlocal maximum
        error = abs(float(actual) - float(expected))
        maximum = max(maximum, error)
        assert error <= 1e-12, (actual, expected, error)
    rows = read(folder / "H3X_EXPLORATORY_CONTEXT_MATCHED_REFIT_TABLE.csv")
    assert len(rows) == 90
    keys = set()
    for r in rows:
        key = tuple(r[k] for k in ("scope", "metric", "history_kind", "horizon_macro"))
        arms = vectors[key]
        assert (key, r["arm"]) not in keys
        keys.add((key, r["arm"]))
        values = arms[r["arm"]]
        assert np.isfinite(values).all()
        equal(r["mean"], values.mean())
        lo, hi = np.quantile(values[ix].mean(1), [.025, .975])
        equal(r["conditional95_low"], lo)
        equal(r["conditional95_high"], hi)
        for name, reference in (("H0", "H0"), ("R3_FIXED3", R3_MEAN)):
            diff = values - arms[reference]
            equal(r["difference_vs_" + name], diff.mean())
            lo, hi = np.quantile(diff[ix].mean(1), [.025, .975])
            equal(r["difference_vs_" + name + "_low"], lo)
            equal(r["difference_vs_" + name + "_high"], hi)
        assert r["label"] == LABEL and r["preset_H3_status"] == "NOT_TRIGGERED"
        assert int(r["independent_cases"]) == 100 and int(r["bootstrap_replicates"]) == 5000
    ledger = json.loads((folder / "H3X_UPDATE_LEDGER.json").read_text())
    assert ledger["formal_updates"] == 90000 and ledger["technical_updates"] == 101
    assert ledger["formal_closed_trajectories"] == 900
    return {"status": "PASS", "label": LABEL, "preset_H3_status": "NOT_TRIGGERED", "summary_rows": 90,
            "new_open_rows": 1800, "old_open_rows": 2400, "new_closed_rows": 900, "old_closed_rows": 1200,
            "case_count": 100, "bootstrap_replicates": 5000, "max_absolute_numeric_difference": maximum,
            "statistics_sha256": STATS_SHA, "audit_code_sha256": sha(__file__), "module_complete_sha256": sha(folder / "COMPLETE.json"),
            "new_simulation": 0, "new_optimizer_updates": 0}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reports", type=Path, required=True)
    p.add_argument("--old-open", type=Path, required=True)
    p.add_argument("--statistics", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = check(args.reports, args.old_open, args.statistics)
    with args.output.open("x") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(result))
