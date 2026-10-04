"""Independent reconstruction of Cube C2 paired statistics from raw rows."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ARMS = ("H0", "REFIT_103201", "REFIT_103202", "REFIT_103203")
MEAN = "FIXED_THREE_REFIT_MEAN_NOT_ENSEMBLE"
STREAMS = ("R3_ORIGINAL", "R4_ALT_CEM_1", "R4_ALT_CEM_2")
STATS_SHA = "e110a936b19622a659812f6b1b1d01a82245fc83ac2381434762dd3a5ae3320b"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def check(folder, source):
    assert sha(source) == STATS_SHA
    spec = importlib.util.spec_from_file_location("r5_c2_independent_statistics", source)
    stats = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stats)
    status = json.loads((folder / "MODULE_STATUS.json").read_text())
    assert status["status"] == "COMPLETE" and status["new_trajectories"] == 800 and status["reused_C1"] == 400
    for name, item in status["files"].items():
        p = folder / name
        assert p.stat().st_size == item["bytes"] and sha(p) == item["sha256"]
    raw = read(folder / "ALL_RAW_VALUES.csv")
    assert len(raw) == 1500
    observed = {}
    for row in raw:
        key = row["case_id"], row["arm"], row["stream"]
        assert key not in observed
        observed[key] = float(row["success"])
        assert row["reset_label"] == "CUBE_OFFICIAL_RESET_SYMMETRIC_NONEXACT"
        assert row["derived_within_case_refit_mean"] == str(row["arm"] == MEAN)
        assert row["reused_C1"] == str(row["stream"] == "R3_ORIGINAL")
        if row["arm"] in ARMS:
            assert observed[key] in (0, 1)
    ids = sorted({r["case_id"] for r in raw})
    assert len(ids) == 100
    assert set(observed) == {(c, a, s) for c in ids for a in (*ARMS, MEAN) for s in STREAMS}
    for c in ids:
        for s in STREAMS:
            assert observed[c, MEAN, s] == np.mean([observed[c, a, s] for a in ARMS[1:]])
    vectors = {(a, s): np.array([observed[c, a, s] for c in ids]) for a in (*ARMS, MEAN) for s in STREAMS}
    ix = stats.case_indices("cube", ids)
    with np.load(folder / "BOOTSTRAP_INDICES.npz", allow_pickle=False) as f:
        assert f["case_ids"].tolist() == ids and np.array_equal(f["indices"], ix)
    max_error = 0.0
    def equal(actual, expected):
        nonlocal max_error
        error = abs(float(actual) - float(expected))
        max_error = max(error, max_error)
        assert error <= 1e-12, (actual, expected)
    rows = read(folder / "CUBE_PLANNER_RANDOMNESS_TABLE.csv")
    assert len(rows) == 35
    keys = set()
    for row in rows:
        kind, arm, stream, reference = (row[k] for k in ("comparison", "arm", "stream", "reference"))
        key = kind, arm, stream, reference
        assert key not in keys
        keys.add(key)
        if kind == "SAME_STREAM_MODEL":
            assert reference == "H0"
            values, base = vectors[arm, stream], vectors["H0", stream]
        elif kind == "SAME_MODEL_STREAM":
            values, base = vectors[arm, stream], vectors[arm, reference]
        else:
            assert kind == "THREE_STREAM_CASE_MEAN"
            values = np.stack([vectors[arm, s] for s in STREAMS]).mean(0)
            base = np.stack([vectors["H0", s] for s in STREAMS]).mean(0)
        difference = (values - base) * 100
        equal(row["success_percent"], values.mean() * 100)
        equal(row["delta_success_pp"], difference.mean())
        for prefix, samples in (("success_ci95", values[ix].mean(1) * 100), ("delta_ci95", difference[ix].mean(1))):
            lo, hi = np.quantile(samples, [.025, .975])
            equal(row[prefix + "_low"], lo)
            equal(row[prefix + "_high"], hi)
        binary = arm != MEAN and kind != "THREE_STREAM_CASE_MEAN"
        for column, b, n in (("failure_to_success", 0, 1), ("success_to_failure", 1, 0)):
            if binary:
                assert int(row[column]) == ((base == b) & (values == n)).sum()
            else:
                assert row[column] == ""
        assert int(row["cases"]) == 100 and row["independent_unit"] == "CASE"
    return {"status": "PASS", "raw_rows": 1500, "observed_trajectories": 1200, "new_trajectories": 800,
            "derived_rows": 300, "summary_rows": 35, "cases": 100, "bootstrap_replicates": 5000,
            "all_saved_indices_equal_frozen_R4": True, "max_absolute_numeric_difference": max_error,
            "module_status_sha256": sha(folder / "MODULE_STATUS.json"), "statistics_sha256": STATS_SHA,
            "audit_code_sha256": sha(__file__), "new_simulation": 0, "new_optimizer_updates": 0}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reports", type=Path, required=True)
    p.add_argument("--statistics", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = check(args.reports, args.statistics)
    with args.output.open("x") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(result))
