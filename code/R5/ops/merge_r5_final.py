"""Merge delivered R5 tables without recomputing or changing source statistics.

Each publication goes to a new directory. R4 and module evidence is read-only.
H3X is always explicitly exploratory and never inserted into preset H3 labels.
"""
from __future__ import annotations
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path

ARMS = {"H0", "REFIT_103201", "REFIT_103202", "REFIT_103203"}
MEAN = "FIXED_THREE_REFIT_MEAN_NOT_ENSEMBLE"


def record(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def read_table(path):
    source = record(path)
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("Missing source rows: " + str(path))
    for row in rows:
        if "merge_source_sha256" in row or "merge_source_table" in row:
            raise RuntimeError("Use original module tables, not an earlier merge")
        row.update(merge_source_sha256=source["sha256"], merge_source_table=source["path"])
    return rows, source


def write_table(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def merge(args):
    sources, tables = {}, {}
    for key in ("r4_main", "r4_raw", "c1_main", "c1_raw", "h1a", "h1b"):
        tables[key], sources[key] = read_table(getattr(args, key))
    for key in ("h2_table", "h2_raw"):
        tables[key], sources[key] = [], []
        for path in getattr(args, key):
            rows, source = read_table(path)
            tables[key].extend(rows)
            sources[key].append(source)
    cube_main = [r for r in tables["c1_main"] if r.get("task") == "cube" and r.get("evaluation_kind") == "CLOSED_LOOP_CEM"]
    if len(cube_main) != 5 or {r["arm"] for r in cube_main} != ARMS | {MEAN}:
        raise RuntimeError("C1 main must contain four arms and the fixed-three mean")
    if any(int(r["cases"]) != 100 or r.get("success") in (None, "") for r in cube_main):
        raise RuntimeError("C1 completed 100-case estimates are required")
    cube_raw = [r for r in tables["c1_raw"] if r.get("task") == "cube" and r.get("evaluation_kind") == "CLOSED_LOOP_CEM"]
    for arm in ARMS | {MEAN}:
        part = [r for r in cube_raw if r.get("arm") == arm]
        if len(part) != 100 or len({r["case_id"] for r in part}) != 100:
            raise RuntimeError("C1 raw must preserve100 cases for each observed/derived arm")
    if len(tables["h1a"]) != 276 or len(tables["h1b"]) != 445:
        raise RuntimeError("Unexpected fixed H1a/H1b delivered table coverage")
    for key in ("h2_table", "h2_raw"):
        if {r["task"] for r in tables[key]} != {"pusht", "reacher"}:
            raise RuntimeError("H2 must contain both authorized tasks, without TwoRoom")
    for task in ("pusht", "reacher"):
        summary = [r for r in tables["h2_table"] if r["task"] == task]
        raw = [r for r in tables["h2_raw"] if r["task"] == task]
        if len(summary) != 132 or len(raw) != 1200:
            raise RuntimeError("Unexpected H2 delivered table coverage for " + task)
        keys = {(r["case_id"], r["arm"], r["stream"]) for r in raw}
        case_ids = {r["case_id"] for r in raw}
        arms = {r["arm"] for r in raw}
        streams = {r["stream"] for r in raw}
        if len(keys) != 1200 or len(case_ids) != 100 or len(arms) != 4 or len(streams) != 3:
            raise RuntimeError("H2 must preserve every case, arm and stream exactly once")
        if keys != {(c, a, s) for c in case_ids for a in arms for s in streams}:
            raise RuntimeError("H2 is missing a case-arm-stream pair")
        if any(r["first_plan_bitwise_equal"].lower() != "true" for r in raw):
            raise RuntimeError("H2 first-plan equality verification is required")
    for row in cube_main + cube_raw:
        row.update(source_study="R5_C1", protocol_label="CUBE_OFFICIAL_RESET_SYMMETRIC_NONEXACT")
    is_old_cube_closed = lambda r: r.get("task") == "cube" and r.get("evaluation_kind") == "CLOSED_LOOP_CEM"
    old_cube_main = [r for r in tables["r4_main"] if is_old_cube_closed(r)]
    old_cube_raw = [r for r in tables["r4_raw"] if is_old_cube_closed(r)]
    if len(tables["r4_main"]) != 80 or len(old_cube_main) != 5 or len(old_cube_raw) != 500:
        raise RuntimeError("Unexpected R4 source table coverage")
    output_tables = {
        "FOUR_TASK_MAIN_TABLE_V2.csv": [r for r in tables["r4_main"] if not is_old_cube_closed(r)] + cube_main,
        "FOUR_TASK_ALL_RAW_VALUES_V2.csv": [r for r in tables["r4_raw"] if not is_old_cube_closed(r)] + cube_raw,
        "R4_CUBE_PREVIOUSLY_UNEVALUABLE_ROWS.csv": old_cube_main,
        "HISTORY_CONDITION_4TASK_TABLE.csv": tables["h1a"],
        "SINGLE_FRAME_INFORMATION_TABLE.csv": tables["h1b"],
        "REAL_HISTORY_REPLANNING_TABLE.csv": tables["h2_table"],
        "REAL_HISTORY_REPLANNING_RAW.csv": tables["h2_raw"],
    }
    for key, filename in (("h3x", "H3X_CONTEXT_MATCHED_REFIT_EXPLORATORY_TABLE.csv"), ("c2", "CUBE_PLANNER_RANDOMNESS_TABLE.csv")):
        path = getattr(args, key)
        if path:
            rows, sources[key] = read_table(path)
            if key == "h3x":
                for row in rows:
                    row["exploratory_label"] = "H3X_EXPLORATORY_AFTER_TRIGGER_DEFECT"
                    row["preset_H3_triggered"] = "False"
            output_tables[filename] = rows
    if not args.h3x or not args.c2:
        raise RuntimeError("Final closeout requires H3X and an explicit C2 result/status table")
    c2_status_path = args.c2.parent / "MODULE_STATUS.json"
    c2_status = json.loads(c2_status_path.read_text())
    sources["c2_module_status"] = record(c2_status_path)
    c2_complete = c2_status["status"] == "COMPLETE"
    if not c2_complete:
        if c2_status["status"] != "TECHNICALLY_UNEVALUABLE":
            raise RuntimeError("C2 must be complete or explicitly closed as technically unevaluable")
        for name in ("C2_PARTIAL_RAW_VALUES.csv", "C2_EXPECTED_CELL_STATUS.csv"):
            rows, source = read_table(args.c2.parent / name)
            sources[name] = source
            output_tables[name] = rows
    # Validate everything before creating a publication; never replace an old one.
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in output_tables.items():
        write_table(args.output / name, rows)
    manifest = {
        "version": "R5_DELIVERED_TABLE_MERGE_V1", "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "evidence_label": "POST_R4_SUPPLEMENT_ON_KNOWN_EVALUATION_CASES",
        "main_modules_complete": ["C1", "H1a", "H1b", "H2"],
        "preset_H3": "NOT_TRIGGERED_TRIGGER_QUANTITY_NOT_DISCRIMINATIVE",
        "H1b_interpretation": "Small-probe baselines, not single-frame information-theoretic lower bounds",
        "C2_included": bool(args.c2), "H3X_exploratory_included": bool(args.h3x),
        "all_R5_and_authorized_H3X_tables_complete": c2_complete,
        "all_authorized_modules_accounted_for": True,
        "C2_status": c2_status["status"],
        "C2_missing_results_are_zero": False,
        "source_tables": sources, "source_files_modified": False, "statistics_recomputed": False,
        "outputs": {name: {**record(args.output / name), "rows": len(rows)} for name, rows in output_tables.items()},
        "source_code": record(__file__),
    }
    with (args.output / "MERGE_STATUS.json").open("x") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "tables": len(output_tables), "rows_main": 80}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("r4-main", "r4-raw", "c1-main", "c1-raw", "h1a", "h1b"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("h2-table", "h2-raw"):
        parser.add_argument("--" + name, type=Path, required=True, nargs="+")
    parser.add_argument("--h3x", type=Path)
    parser.add_argument("--c2", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    merge(parser.parse_args())
