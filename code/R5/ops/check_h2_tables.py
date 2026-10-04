"""Independent read-only H2 table check from paired raw observations.

Does not import H2 report code or run any model/physics. Frozen R4 bootstrap
draws are shared with the scientific protocol, while row reconstruction uses
independent dictionary grouping rather than the report's padded tensors.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ARMS = ("H0", "REFIT_103201", "REFIT_103202", "REFIT_103203")
STREAMS = ("R3_ORIGINAL", "R4_ALT_CEM_1", "R4_ALT_CEM_2")
MEAN_ARM = "FIXED3_REFIT_MEAN_NOT_ENSEMBLE"
MEAN_STREAM = "MEAN_OF_THREE_STREAMS_WITHIN_CASE"
STATS_SHA = "e110a936b19622a659812f6b1b1d01a82245fc83ac2381434762dd3a5ae3320b"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def boolean(value):
    assert value in ("True", "False")
    return float(value == "True")


def check(folder, statistics):
    status = json.loads((folder / "MODULE_STATUS.json").read_text())
    task = status["task"]
    assert task in ("reacher", "pusht") and status["status"] == "COMPLETE"
    for name, record in status["outputs"].items():
        path = folder / name
        assert sha(path) == record["sha256"]
        assert path.stat().st_size == record["bytes"]
    assert sha(statistics) == STATS_SHA
    spec = importlib.util.spec_from_file_location("r5_independent_h2_statistics", statistics)
    stats = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stats)
    raw = read(folder / "ALL_RAW_VALUES.csv")
    assert len(raw) == 1200 and {r["task"] for r in raw} == {task}
    ids = sorted({r["case_id"] for r in raw})
    assert len(ids) == 100
    observed, families = {}, {}
    for row in raw:
        key = row["case_id"], row["arm"], row["stream"]
        assert key not in observed
        assert boolean(row["first_plan_bitwise_equal"])
        assert row["control_history"] == "H_POLICY" and row["intervention_history"] == "H_REAL3_REPLAN"
        assert int(row["new_optimizer_updates"]) == 0
        b, n = float(row["control_success"]), float(row["intervention_success"])
        assert b in (0, 1) and n in (0, 1)
        assert n - b == float(row["difference_REAL3_minus_POLICY"])
        br, nr = boolean(row["control_entered_replan"]), boolean(row["intervention_entered_replan"])
        assert br == nr
        if not nr:
            assert b == n
        if row["case_id"] in families:
            assert families[row["case_id"]] == row["family_id"]
        families[row["case_id"]] = row["family_id"]
        observed[key] = np.array([b, n, br, nr])
    assert set(observed) == {(c, a, s) for c in ids for a in ARMS for s in STREAMS}
    vectors = {}
    for arm in (*ARMS, MEAN_ARM):
        for stream in (*STREAMS, MEAN_STREAM):
            use_arms = ARMS[1:] if arm == MEAN_ARM else (arm,)
            use_streams = STREAMS if stream == MEAN_STREAM else (stream,)
            # Equal model means first, then equal stream means, within each case.
            vectors[arm, stream] = np.stack([
                np.stack([np.stack([observed[c, a, s] for a in use_arms]).mean(0)
                          for s in use_streams]).mean(0) for c in ids
            ])
    indices = stats.case_indices(task, ids)
    weights = stats.family_weights(task, ids, [families[c] for c in ids]) if task == "pusht" else None
    max_error = 0.0
    def equal(actual, expected):
        nonlocal max_error
        error = abs(float(actual) - float(expected))
        max_error = max(max_error, error)
        assert error <= 1e-12, (actual, expected, error)
    table = read(folder / "REAL_HISTORY_REPLANNING_TABLE.csv")
    assert len(table) == 132
    summary_keys = set()
    raw_sha = sha(folder / "ALL_RAW_VALUES.csv")
    for row in table:
        arm, stream, metric, history = (row[k] for k in ("arm", "stream", "metric", "history_kind"))
        key = arm, stream, metric, history
        assert key not in summary_keys
        summary_keys.add(key)
        v = vectors[arm, stream]
        if metric == "success_difference_REAL3_minus_POLICY":
            assert history == "PAIRED_HISTORY_CONTRAST"
            values = v[:, 1] - v[:, 0]
        else:
            q = {"H_POLICY": 0, "H_REAL3_REPLAN": 1}[history]
            if metric == "success_fraction":
                values = v[:, q]
            elif metric == "entered_replanning_fraction":
                values = v[:, q + 2]
            else:
                assert metric == "success_difference_vs_H0" and arm != "H0"
                values = v[:, q] - vectors["H0", stream][:, q]
        equal(row["estimate"], values.mean())
        low, high = np.quantile(values[indices].mean(1), [.025, .975])
        equal(row["conditional95_low"], low)
        equal(row["conditional95_high"], high)
        if weights is not None:
            low, high = np.quantile(weights @ values / weights.sum(1), [.025, .975])
            equal(row["family95_low"], low)
            equal(row["family95_high"], high)
        else:
            assert row["family95_low"] == row["family95_high"] == ""
        assert int(row["cases"]) == 100 and int(row["bootstrap_replicates"]) == 5000
        assert row["independent_unit"] == "CASE"
        assert row["source_table_sha256"] == raw_sha
        assert int(row["planning_streams_averaged_within_case"]) == (3 if stream == MEAN_STREAM else 1)
        assert int(row["models_averaged_within_case"]) == (3 if arm == MEAN_ARM else 1)
    means = read(folder / "CASE_MEANS.csv")
    assert len(means) == 2000
    positions = {c: i for i, c in enumerate(ids)}
    mean_keys = set()
    for row in means:
        key = row["case_id"], row["arm"], row["stream"]
        assert key not in mean_keys
        mean_keys.add(key)
        v = vectors[row["arm"], row["stream"]][positions[row["case_id"]]]
        for column, expected in zip(("control_success_fraction", "intervention_success_fraction", "control_replan_fraction", "intervention_replan_fraction"), v):
            equal(row[column], expected)
        equal(row["paired_difference"], v[1] - v[0])
    transitions = read(folder / "SUCCESS_TRANSITIONS.csv")
    assert len(transitions) == 20
    for row in transitions:
        arm, stream = row["arm"], row["stream"]
        v = vectors[arm, stream]
        diff = v[:, 1] - v[:, 0]
        for column, expected in (("improved_cases", (diff > 0).sum()), ("worsened_cases", (diff < 0).sum()), ("unchanged_cases", (diff == 0).sum())):
            assert int(row[column]) == expected
        if arm in ARMS and stream in STREAMS:
            for column, before, after in (("failure_to_success", 0, 1), ("success_to_failure", 1, 0), ("both_success", 1, 1), ("both_failure", 0, 0)):
                assert int(row[column]) == ((v[:, 0] == before) & (v[:, 1] == after)).sum()
    return {"status": "PASS", "task": task, "paired_raw_rows": 1200, "summary_rows": 132,
            "case_mean_rows": 2000, "transition_rows": 20, "max_absolute_numeric_difference": max_error,
            "module_status_sha256": sha(folder / "MODULE_STATUS.json"), "statistics_sha256": STATS_SHA,
            "audit_code_sha256": sha(__file__), "new_physics_trajectories": 0, "new_optimizer_updates": 0,
            "scope": "All delivered paired raw rows, model/stream means within case, case/family bootstrap intervals and transitions; no new simulation or model inference"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.reports, args.statistics)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False))
