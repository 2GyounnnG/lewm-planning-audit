#!/usr/bin/env python3
"""Recompute delivered R4 statistics on CPU into a NEW, independent directory.

Example (Python with NumPy; no torch, simulator, model or optimizer required):
  python ops/recompute_delivery.py --r3-root /path/to/r3/recovery --dry-run
  python ops/recompute_delivery.py --r3-root /path/to/r3/recovery

Default output: ROOT/recomputed/<UTC>-<pid>. Existing outputs are never reused.
R4 S1/S2/S3 use the SHA-verified statistics_full_v3 source and its fixed 5000
bootstrap draws. R3/X1/X2 original tables and CI columns are preserved after
coverage/arithmetic validation, not model re-evaluation or new CI estimation.
MAIN_MERGE_SPEC is remapped exclusively into the new directory. A full run
requires all main inputs, including explicitly allowed technical limitations.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

FROZEN = {
    "aggregate.py": "f17eb75fc036986963b675c4281f19bc938728fb583773277ccd640f19b98d8a",
    "statistics.py": "e110a936b19622a659812f6b1b1d01a82245fc83ac2381434762dd3a5ae3320b",
    "test_aggregate.py": "8734bbb55543db263cfab3b7f32fe03755c4595d822ec204cf040bf3b8e0a362",
    "test_statistics.py": "3274c7d6e082298b0bd197ad1e52729e4d81c8e19496bc5be49fc5757d1004a0",
}
S1_INPUTS = {
    "pusht": {
        "s1_offline": "r4/reports/remote_pusht_s1_offline_fast/S1_offline_raw.csv",
        "s1_cross": "r4/reports/remote_small_001/r4_pusht/s1/S1_cross_raw.csv",
    },
    "reacher": {
        "s1_offline": "r4/reports/remote_reacher_s1_offline/S1_offline_raw.csv",
        "s1_cross": "r4/reports/remote_reacher_s1_complete/s1/S1_cross_raw.csv",
    },
}


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_helper(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def same_csv(left, right):
    """Compare every original field/value exactly, ignoring row/column order."""
    canonical = lambda path: sorted(json.dumps(row, sort_keys=True) for row in read_csv(path))
    check(canonical(left) == canonical(right), f"Frozen table reconstruction differs: {right}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--r3-root", type=Path, required=True, help="Contains manifests/{pusht,reacher}_data_roles.json")
    parser.add_argument("--out", type=Path, help="New directory; inside ROOT only ROOT/recomputed/ is allowed")
    parser.add_argument("--dry-run", action="store_true", help="Verify frozen source/paths and write plan only; do not recompute")
    args = parser.parse_args()
    root, r3 = args.root.resolve(), args.r3_root.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = (args.out or root / "recomputed" / f"{stamp}-{os.getpid()}").resolve()
    check(not out.exists(), f"Output already exists; choose a new directory: {out}")
    check(out != root and root not in out.parents or out.is_relative_to(root / "recomputed"), "Refusing output inside a sealed source directory")
    check(not out.is_relative_to(r3), "Refusing output inside R3 inputs")
    check(not root.is_relative_to(out), "Output cannot be an ancestor of inputs")
    out.mkdir(parents=True)
    manifest = {
        "version": "R4_V23_CPU_DELIVERY_RECOMPUTE_V1", "status": "STARTED",
        "started_utc": datetime.now(timezone.utc).isoformat(), "root": str(root),
        "r3_root": str(r3), "output_root": str(out), "dry_run": args.dry_run,
        "model_inference_calls": 0, "simulator_calls": 0, "optimizer_updates": 0,
        "scope": {
            "R4": "S1/S2/S3 reconstructed from delivered raw observations using unchanged frozen v3 definitions and bootstrap indices.",
            "R3_X1_X2": "Original raw/summary/CI columns retained, with coverage and arithmetic validation; no model rerun and no new confidence intervals.",
            "merge": "Lossless concatenation using delivered MAIN_MERGE_SPEC; tasks/units are not pooled.",
            "secondary": "Main batch only. Mid-fork availability and optional second-batch analyses are not recomputed.",
            "recovery": "S3 uses explicitly derived physical-state subsets with original result.json; full pixels/trajectories are not required or claimed.",
        }, "operations": [], "inputs": [], "comparisons": [], "pending": [],
    }
    tracked = {}

    def track(path, expected=None):
        value = record(path)
        if expected:
            check(all(value[k] == expected[k] for k in ("bytes", "sha256")), f"Source digest mismatch: {path}")
        tracked[value["path"]] = value
        return value

    def copy(relative):
        source, destination = root / relative, out / relative
        track(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return destination

    def operation(name, scope, **extra):
        manifest["operations"].append(dict(module=name, scope=scope, **extra))
        print(json.dumps({"module": name, "scope": scope}), flush=True)

    try:
        track(Path(__file__))
        frozen_dir = root / "manifests/statistics_full_v3"
        freeze = read_json(frozen_dir / "STATISTICS_FREEZE.json")
        check(freeze["files"] == FROZEN and freeze["replicates"] == 5000, "Not the delivered frozen statistics v3 contract")
        track(frozen_dir / "STATISTICS_FREEZE.json")
        for name, digest in FROZEN.items():
            check(track(frozen_dir / "source" / name)["sha256"] == digest, "Frozen source changed: " + name)
        for task in S1_INPUTS:
            role_file = r3 / f"manifests/{task}_data_roles.json"
            track(role_file)
            cases = read_json(role_file)["cases"]["EVAL"]
            check(len(cases) == len({x["case_id"] for x in cases}) == 100, "R3 fixed100 universe differs")
            if task == "pusht":
                check(all(x.get("family_id") for x in cases), "Missing original PushT family labels")

        spec_path = root / "manifests/MAIN_MERGE_SPEC.json"
        track(spec_path)
        merge_spec = read_json(spec_path)
        original_root = Path(merge_spec["requirements"][0]["status_file"]).parents[2]

        def relative_source(path):
            return Path(path).relative_to(original_root)

        required = set()
        for requirement in merge_spec["requirements"]:
            relative = relative_source(requirement["status_file"])
            required.add(relative)
            path = root / relative
            if path.exists():
                gate = read_json(path)
                permitted = gate.get("status") == "COMPLETE" or (gate.get("status") in requirement.get("allowed_technical_statuses", []) and gate.get("reason"))
                if not permitted:
                    manifest["pending"].append({"module": requirement["name"], "status": gate.get("status")})
        for table in merge_spec["tables"]:
            check(Path(table["name"]).name == table["name"], "Unsafe merge output filename")
            required.update(relative_source(source["path"]) for source in table["sources"])
        for module in ("r3", "tworoom", "cube"):
            folder = Path("tables/r3_reference" if module == "r3" else f"x1/reports/{module}_MAIN")
            required.update(folder / name for name in ("MODULE_STATUS.json", "ALL_RAW_VALUES.csv", "MAIN_TABLE.csv"))
        required.update(map(Path, ["x2/inputs/X2_INPUT_MANIFEST.json", "x2/reports/X2_STATUS.json"]))
        for task, variants in S1_INPUTS.items():
            required.update(map(Path, variants.values()))
            required.add(Path(f"r4/reports/remote_{task}_statistics/statistics_recovery/STATISTICS_RECOVERY_MAP.json"))
        for relative in sorted(required):
            if (root / relative).exists():
                track(root / relative)
            else:
                manifest["pending"].append({"path": str(relative), "reason": "NOT_RECOVERED"})
        manifest["planned_required_files"] = [str(x) for x in sorted(required)]
        if args.dry_run:
            manifest["status"] = "DRY_RUN_WAITING_MAIN_INPUTS" if manifest["pending"] else "DRY_RUN_READY"
            return 0
        if manifest["pending"]:
            manifest["status"] = "WAITING_MAIN_INPUTS"
            return 2

        import numpy as np
        manifest["runtime"] = {"python": sys.version, "numpy": np.__version__, "threads": 1, "device": "CPU"}
        for name in FROZEN:
            copy(Path("manifests/statistics_full_v3/source") / name)
        copy(Path("manifests/statistics_full_v3/STATISTICS_FREEZE.json"))
        frozen_script = out / "manifests/statistics_full_v3/source/aggregate.py"

        def recompute(module, source, task, relative):
            destination = out / relative
            destination.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-B", str(frozen_script), module, "--input", str(source), "--r3-root", str(r3), "--task", task, "--out", str(destination)]
            log = out / "logs" / f"{task}_{relative.name}.log"
            log.parent.mkdir(exist_ok=True)
            with log.open("w") as handle:
                subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True, env=os.environ.copy())
            for generated in sorted(destination.glob("*.csv")):
                reference = root / relative / generated.name
                if reference.exists():
                    track(reference)
                    same_csv(generated, reference)
                    manifest["comparisons"].append({"table": str(relative / generated.name), "all_fields_equal_order_independent": True})
            operation(f"R4_{task}_{relative.name}", "RECOMPUTED_FROZEN_V3", command=command)

        for task, variants in S1_INPUTS.items():
            for variant, relative in variants.items():
                recompute("s1", root / relative, task, Path(f"tables/r4_{task}/{variant}"))
            combined = root / f"tables/r4_{task}/s2/S2_ALL_CANDIDATE_RAW.csv"
            track(combined)
            groups = defaultdict(list)
            for row in read_csv(combined):
                check(row["fork"] == "initial", "This entry point handles the initial main S2 batch only")
                groups[row["case_id"]].append(row)
            check(len(groups) == 100 and all(len(rows) == 64 for rows in groups.values()), "S2 fixed100x64 coverage differs")
            staged = out / "work" / task / "s2"
            for case, rows in groups.items():
                check(Path(case).name == case and case not in ("", ".", ".."), "Unsafe case directory name")
                destination = staged / case / "candidate_raw.csv"
                destination.parent.mkdir(parents=True)
                with destination.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            recompute("s2", staged, task, Path(f"tables/r4_{task}/s2"))
            copy(Path(f"tables/r4_{task}/s2/S2_ALL_CANDIDATE_RAW.csv"))
            source = root / f"r4/reports/remote_{task}_statistics/statistics_recovery"
            metadata = read_json(source / "STATISTICS_RECOVERY_MAP.json")
            logical = []
            for entry in metadata["records"]:
                folder = (source / entry["relative_path"]).resolve()
                check(folder.is_relative_to(source.resolve()), "Recovery path escapes its directory")
                for name, key in (("result.json", "source_result"), ("trajectory.npz", "derived_state_npz")):
                    track(folder / name, entry[key])
                track(folder / "DERIVED_STATISTICS_INPUT.json")
                check(read_json(folder / "DERIVED_STATISTICS_INPUT.json") == entry, "Derived state lineage differs")
                result = read_json(folder / "result.json")
                check(result["status"] == "COMPLETE" and result["optimizer_updates"] == 0, "R4 result status or update count differs")
                logical.append((result["case_id"], result["arm"], result["stream"]))
            check(len(logical) == len(set(logical)) == (1380 if task == "pusht" else 1260), "S3 main policy coverage differs")
            recompute("s3", source, task, Path(f"tables/r4_{task}/s3"))

        deliver = load_helper(root / "ops/deliver.py", "deliver")
        validators = load_helper(root / "ops/validate_main_tables.py", "recompute_main_validators")
        track(root / "ops/deliver.py")
        track(root / "ops/validate_main_tables.py")
        for module in ("r3", "tworoom", "cube"):
            folder = Path("tables/r3_reference" if module == "r3" else f"x1/reports/{module}_MAIN")
            for name in ("MODULE_STATUS.json", "ALL_RAW_VALUES.csv", "MAIN_TABLE.csv"):
                copy(folder / name)
            validation = validators.canonical(out / folder, ["pusht", "reacher"] if module == "r3" else [module])
            atomic(out / f"manifests/main_validation/{module}.json", validation)
            operation(module, "ORIGINAL_TABLE_AND_CI_PRESERVED_ARITHMETIC_VERIFIED", validation=validation)
        for relative in sorted(required):
            if str(relative).startswith("x2/"):
                copy(relative)
        validation = validators.x2(out)
        atomic(out / "manifests/main_validation/x2.json", validation)
        operation("x2", "ORIGINAL_TABLE_AND_SCOPE_DIFFERENCES_PRESERVED_RISK_SUBTRACTION_AND_THREE_SEED_MEANS_VERIFIED", validation=validation)

        # Preserve technical gates and non-statistical source tables. Recomputed
        # paths above are never replaced by original tables during this staging.
        for relative in sorted(required):
            if not (out / relative).exists():
                copy(relative)
        rewritten = json.loads(json.dumps(merge_spec))
        for requirement in rewritten["requirements"]:
            requirement["status_file"] = str(out / relative_source(requirement["status_file"]))
        for table in rewritten["tables"]:
            for source in table["sources"]:
                source["path"] = str(out / relative_source(source["path"]))
        atomic(out / "manifests/MAIN_MERGE_SPEC.json", rewritten)
        result = deliver.merge(out / "manifests/MAIN_MERGE_SPEC.json", out / "merged")
        check(result["status"] in ("COMPLETE", "COMPLETE_WITH_TECHNICAL_LIMITATIONS"), "Merge inputs not complete")
        operation("main_merge", "LOSSLESS_MERGE_OF_RECOMPUTED_R4_AND_PRESERVED_VALIDATED_R3_X1_X2", merge_status=result["status"])
        for original in tracked.values():
            now = record(original["path"])
            check(now == original, "Input changed during run: " + original["path"])
        manifest["source_files_unchanged_during_run"] = True
        manifest["outputs"] = [record(path) for path in sorted(out.rglob("*")) if path.is_file()]
        manifest["status"] = result["status"]
        return 0
    except Exception as error:
        manifest["status"] = "FAILED"
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        print(json.dumps(manifest["error"]), file=sys.stderr, flush=True)
        return 1
    finally:
        manifest["inputs"] = list(tracked.values())
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        atomic(out / "RECOMPUTE_MANIFEST.json", manifest)
        (out / "README.txt").write_text(
            "R4-v2.3 CPU delivery reconstruction\n"
            "Inspect RECOMPUTE_MANIFEST.json status before using any output.\n"
            "R4 S1/S2/S3: unchanged SHA-verified frozen v3 statistics; initial main batch only.\n"
            "R3/X1/X2: original raw and summary/CI columns preserved after coverage/arithmetic checks.\n"
            "No models, encoders, simulators, optimizers, new seeds or new statistical definitions were run.\n"
            "Original root files are read-only inputs. Derived S3 state subsets retain explicit original SHA lineage.\n"
            "Independent outputs are under tables/, x1/, x2/; merged/ exists only after all main gates pass.\n"
        )
        print(json.dumps({"status": manifest["status"], "out": str(out), "pending": len(manifest["pending"])}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
