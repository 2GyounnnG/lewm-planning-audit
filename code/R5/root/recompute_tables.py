"""One local entry point: recheck paired statistics and rebuild R5 tables.

Reads recovered original tables only. Runs no model, simulator, optimizer or
network request. The output must be a new directory; sealed files are untouched.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def load(name):
    path = ROOT / "ops" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(args):
    if args.output.exists():
        raise ValueError("Choose a new output directory; no existing publication is replaced")
    statistics = ROOT / "h1a/inherited_statistics.py"
    h2 = [ROOT / "h2/reports" / (task + "_MAIN") for task in ("reacher", "pusht")]
    checks = {task: load("check_h2_tables").check(folder, statistics)
              for task, folder in zip(("H2_reacher", "H2_pusht"), h2)}
    h3x = ROOT / "h3x/reports/main"
    c2 = ROOT / "c2/reports"
    if not args.main_only:
        checks["H3X"] = load("check_h3x_tables").check(h3x, ROOT / "h1a/reports/HISTORY_CONDITION_4TASK_RAW.csv", statistics)
        checks["C2"] = load("check_c2_tables").check(c2, statistics)
    # H1a/H1b sources must still match their already sealed original bytes.
    sealed = json.loads((ROOT / "ops/INITIAL_STAGE_RECOVERY_MANIFEST.json").read_text())
    original = {row["path"]: row for row in sealed["verified_local_files"]}
    for relative in ("h1a/reports/HISTORY_CONDITION_4TASK_TABLE.csv", "h1b/reports/main/SINGLE_FRAME_INFORMATION_TABLE.csv"):
        path = ROOT / relative
        expected = original.get(str(path))
        if expected is None:
            matches = [v for k, v in original.items() if k.endswith("/r5_execution/" + relative)]
            if len(matches) != 1:
                raise ValueError("Cannot uniquely map the sealed source: " + relative)
            expected = matches[0]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"]
    load("merge_r5").merge(SimpleNamespace(
        r4_main=ROOT / "source_tables/R4_MAIN_SEALED/FOUR_TASK_MAIN_TABLE.csv",
        r4_raw=ROOT / "source_tables/R4_MAIN_SEALED/FOUR_TASK_ALL_RAW_VALUES.csv",
        c1_main=ROOT / "c1/reports/MAIN_TABLE.csv", c1_raw=ROOT / "c1/reports/ALL_RAW_VALUES.csv",
        h1a=ROOT / "h1a/reports/HISTORY_CONDITION_4TASK_TABLE.csv",
        h1b=ROOT / "h1b/reports/main/SINGLE_FRAME_INFORMATION_TABLE.csv",
        h2_table=[p / "REAL_HISTORY_REPLANNING_TABLE.csv" for p in h2],
        h2_raw=[p / "ALL_RAW_VALUES.csv" for p in h2],
        h3x=None if args.main_only else h3x / "H3X_EXPLORATORY_CONTEXT_MATCHED_REFIT_TABLE.csv",
        c2=None if args.main_only else c2 / "CUBE_PLANNER_RANDOMNESS_TABLE.csv",
        output=args.output))
    receipt = {"status": "PASS", "statistical_checks": checks,
               "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "scope": "All H2 paired statistics; H3X and C2 also checked when included. H1a/H1b unchanged sealed table SHA verified; C1 coverage/source values preserved by the merge.",
               "new_optimizer_updates": 0, "new_simulation": 0, "source_files_changed": False}
    with (args.output / "RECOMPUTATION_CHECKS.json").open("x") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--main-only", action="store_true", help="Rebuild the delivered C1/H1a/H1b/H2 main stage without later supplements")
    main(parser.parse_args())
