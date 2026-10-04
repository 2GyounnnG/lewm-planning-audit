"""Audit C2 coverage and explicit missingness after its unchanged gate failed."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ARMS = ("H0", "REFIT_103201", "REFIT_103202", "REFIT_103203")
MEAN = "FIXED_THREE_REFIT_MEAN_NOT_ENSEMBLE"
STREAMS = ("R4_ALT_CEM_1", "R4_ALT_CEM_2")
ALL_STREAMS = ("R3_ORIGINAL",) + STREAMS
PARTIAL = "PARTIAL_TECHNICAL_DIAGNOSTIC_NOT_INFERENTIAL"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def check(folder):
    folder = Path(folder)
    status = json.loads((folder / "MODULE_STATUS.json").read_text())
    assert status["status"] == "TECHNICALLY_UNEVALUABLE"
    assert status["completed_observed_trajectories"] == 280
    assert status["expected_new_trajectories"] == 800
    assert status["failed_design_cells_before_action_and_CEM"] == 1
    assert status["failed_attempts_before_action_and_CEM"] == 2
    assert status["not_run_design_cells"] == 519
    assert status["prior_280_complete_and_all_files_unchanged"]
    assert status["C1_sources_unchanged"] and status["new_training_updates"] == 0
    for name, expected in status["files"].items():
        path = folder / name
        assert path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"]
    contract = json.loads((folder / "C2_CONTRACT.json").read_text())
    ids = {r["case_id"] for r in contract["cases"]["EVAL"]}
    assert len(ids) == 100
    observed = rows(folder / "C2_PARTIAL_RAW_VALUES.csv")
    coverage = rows(folder / "C2_EXPECTED_CELL_STATUS.csv")
    key = lambda r: (r["case_id"], r["arm"], r["stream"])
    actual = {key(r): r for r in observed}
    complete = {key(r): r for r in coverage}
    assert len(observed) == len(actual) == 280
    assert len(coverage) == len(complete) == 800
    assert set(complete) == {(c, a, s) for c in ids for a in ARMS for s in STREAMS}
    assert {key(r) for r in coverage if r["status"] == "OBSERVED_COMPLETE"} == set(actual)
    partial_ids = {r["case_id"] for r in observed}
    assert len(partial_ids) == 70
    assert set(actual) == {(c, a, "R4_ALT_CEM_1") for c in partial_ids for a in ARMS}
    source_shas = {r["path"]: r["sha256"] for r in status["sources"]}
    for k, r in actual.items():
        assert r["evaluation_status"] == PARTIAL
        assert float(r["success"]) in (0, 1) and 1 <= int(r["executed_raw_steps"]) <= 50
        assert 1 <= int(r["replan_calls"]) <= 2 and int(r["optimizer_updates"]) == 0
        assert r["source_sha256"] == source_shas[r["source_result_path"]]
        for dest, source in (("success", "success"), ("executed_raw_steps", "executed_raw_steps"),
                             ("replan_calls", "replan_calls"), ("source_result_sha256", "source_sha256"),
                             ("source_complete_sha256", "source_complete_sha256")):
            assert complete[k][dest] == r[source]
    failed = [r for r in coverage if r["status"] == "FAILED_BEFORE_ACTION_AND_CEM"]
    unrun = [r for r in coverage if r["status"] == "NOT_RUN"]
    assert len(failed) == 1 and len(unrun) == 519
    assert all(r["success"] == "NA" for r in failed + unrun)
    assert failed[0]["executed_raw_steps"] == failed[0]["replan_calls"] == "0"
    assert all(r["executed_raw_steps"] == r["replan_calls"] == "NA" for r in unrun)
    failures = json.loads((folder / "TWO_FAILED_ATTEMPTS_RAW.json").read_text())
    assert len(failures) == 2 and {r["attempt"] for r in failures} == {"ORIGINAL", "SINGLE_NATIVE_RESUME"}
    for r in failures:
        assert key(r) == key(failed[0])
        assert r["fields_compared"] == 76 and r["nonrender_fields_exact"] == 75
        assert r["render_different_channels"] == 13 and r["render_max_abs"] == 1
        assert r["evaluated_raw_actions"] == r["CEM_calls"] == 0
    table = rows(folder / "CUBE_PLANNER_RANDOMNESS_TABLE.csv")
    expected_keys = {("SAME_STREAM_MODEL", a, s, "H0") for a in ARMS + (MEAN,) for s in ALL_STREAMS}
    expected_keys |= {("SAME_MODEL_STREAM", a, ALL_STREAMS[b], ALL_STREAMS[c])
                      for a in ARMS + (MEAN,) for c, b in ((0, 1), (0, 2), (1, 2))}
    expected_keys |= {("THREE_STREAM_CASE_MEAN", a, "MEAN_OF_FIXED_THREE_STREAMS", "H0") for a in ARMS + (MEAN,)}
    assert len(table) == 35
    assert {tuple(r[k] for k in ("comparison", "arm", "stream", "reference")) for r in table} == expected_keys
    c1 = folder.parents[1] / "c1/reports"
    old = {r["arm"]: r for r in rows(c1 / "MAIN_TABLE.csv")}
    flips = {r["arm"]: r for r in rows(c1 / "PAIRED_FLIPS.csv")}
    map_fields = {"success_percent": "success_percent", "success_ci95_low": "success_case_ci95_low",
                  "success_ci95_high": "success_case_ci95_high", "delta_success_pp": "delta_success_pp",
                  "delta_ci95_low": "delta_case_ci95_low", "delta_ci95_high": "delta_case_ci95_high"}
    baseline_count = na_count = 0
    for r in table:
        assert int(r["cases"]) == 100
        if r["comparison"] == "SAME_STREAM_MODEL" and r["stream"] == "R3_ORIGINAL":
            baseline_count += 1
            assert r["estimation_status"] == "C1_SOURCE_ONLY_UNCHANGED"
            assert r["source_sha256"] == sha(c1 / "MAIN_TABLE.csv")
            for dest, source in map_fields.items():
                assert r[dest] == (old[r["arm"]][source] or "NA")
            if r["arm"] in flips:
                assert r["failure_to_success"] == flips[r["arm"]]["s01_H0_failure_REFIT_success"]
                assert r["success_to_failure"] == flips[r["arm"]]["s10_H0_success_REFIT_failure"]
        else:
            na_count += 1
            assert r["estimation_status"] == "TECHNICALLY_UNEVALUABLE"
            assert all(r[k] == "NA" for k in list(map_fields) + ["failure_to_success", "success_to_failure"])
            assert int(r["bootstrap_replicates"]) == 0
    assert (baseline_count, na_count) == (5, 30)
    return {"status": "PASS_TECHNICAL_DEGRADATION_ACCOUNTING", "C2_scientific_status": "TECHNICALLY_UNEVALUABLE",
            "actual_raw_rows": 280, "expected_cells": 800, "failed_before_action_cells": 1, "not_run_cells": 519,
            "same_cell_failed_attempts": 2, "C1_unchanged_reference_rows": 5, "C2_NA_estimate_rows": 30,
            "missing_success_values_filled_with_zero": False, "partial_sample_inference_performed": False,
            "module_status_sha256": sha(folder / "MODULE_STATUS.json"), "audit_code_sha256": sha(__file__),
            "C1_main_sha256": sha(c1 / "MAIN_TABLE.csv"), "C1_flips_sha256": sha(c1 / "PAIRED_FLIPS.csv"),
            "new_simulation": 0, "new_optimizer_updates": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.reports)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result))
