"""Seal a finished R5 continuation into a new directory; never edit old seals.

The input config names completed module receipts and explicitly preserves any
CPU acceptance failures and recovery scope limits. It is assembled only after
the scientific modules and their recovery checks have finished.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
from validate_component_seals import validate_component_seals


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main(args):
    source_root = args.source_root.resolve()
    config = read(args.config)
    assert config["all_scientific_jobs_finished"] is True
    assert config["all_module_recovery_checks_finished"] is True
    assert config["automatic_destroy"] is False and config["automatic_rental"] is False
    assert config["preset_H3_status"] == "NOT_TRIGGERED_TRIGGER_QUANTITY_NOT_DISCRIMINATIVE"
    assert config["H3X_label"] == "H3X_EXPLORATORY_AFTER_TRIGGER_DEFECT"
    counts = config["formal_new_closed_loop_trajectories"]
    assert counts == {"C1": 400, "H2_reacher": 1200, "H2_pusht": 1200, "H3X": 900, "C2": 280}
    actual_trajectories = sum(counts.values())
    assert actual_trajectories == 3980
    assert config["technical_degradations"]["C2"]["status"] == "TECHNICALLY_UNEVALUABLE"
    assert config["technical_degradations"]["C2"]["missing_cells"] == 520
    assert config["technical_degradations"]["C2"]["missing_values_filled_with_zero"] is False
    assert config["formal_world_model_updates"] == {"H3X": 90000}
    assert config["H3X_technical_world_model_updates"] == 101
    for item in config["completion_receipts"].values():
        receipt = read(item["local_path"])
        for key, expected in item["required_values"].items():
            assert receipt[key] == expected, (item["local_path"], key, receipt[key], expected)
    merge = read(config["final_merge_status"])
    assert merge["all_R5_and_authorized_H3X_tables_complete"] is False
    assert merge["all_authorized_modules_accounted_for"] is True
    assert merge["C2_status"] == "TECHNICALLY_UNEVALUABLE"
    assert set(merge["main_modules_complete"]) == {"C1", "H1a", "H1b", "H2"}
    for name, item in merge["outputs"].items():
        local = Path(config["final_merge_status"]).parent / name
        actual = record(local)
        assert (actual["sha256"], actual["bytes"]) == (item["sha256"], item["bytes"])
    boundary = read(config["final_boundary_check"])
    assert boundary["status"] == "PASS"
    assert not any(row["differences"] for row in boundary["checks"].values())
    primary_boundary = read(config["primary_initial_boundary_check"])
    assert primary_boundary["status"] == "PASS" and not primary_boundary["differences"]
    assert primary_boundary["checked_files"] == 2880
    required = {"INITIAL_STAGE", "C1", "H2_reacher", "H2_pusht", "H3X", "C2"}
    assert set(config["module_recovery_seals"]) == required
    module_records = {}
    for key, item in config["module_recovery_seals"].items():
        data = read(item["local_path"])
        module_records[key] = {**item, "record": record(item["local_path"]), "sealed_status": data["status"]}
    # Both observed failures are required, rather than trusting a hand-written
    # configuration list that could accidentally omit the later H2 failure.
    failures = config["strict_CPU_acceptance_failures"]
    by_module = {row["module"]: row for row in failures}
    assert len(by_module) == len(failures)
    required_failures = {
        "H1a": {"sha256": "dbf72b47cae90ace985d666a6ef73032ac2d522e43f5e446483f8bb8fad6a070",
                "status": "FAILED_ORIGINAL_CPU_TOLERANCE", "outliers": 12, "coordinates": 1536000},
        "H2_reacher": {"sha256": "61318c27cd81704517006515c5da10ec64b501a994264e92917066911ab9abf5",
                       "status": "FAIL_FROZEN_CPU_COST_TOLERANCE", "outliers": 3, "candidate_costs": 1200},
    }
    for module, expected_failure in required_failures.items():
        item = by_module[module]
        path = item["local_receipt"]
        assert record(path)["sha256"] == expected_failure["sha256"]
        actual_failure = read(path)
        for key, expected_value in expected_failure.items():
            if key != "sha256":
                assert actual_failure[key] == expected_value
        assert item["outliers"] == actual_failure["outliers"] and item["release_gate"] == "HOLD"
        if module == "H1a":
            assert actual_failure["tolerance"] == {"atol": 1e-5, "rtol": 1e-5}
        else:
            assert actual_failure["release_gate_for_this_narrow_check"] == "HOLD"
            assert actual_failure["atol"] == actual_failure["rtol"] == 1e-5
    assert config["all_original_pixels_recovered_locally"] is False
    component_expected = validate_component_seals(config)
    inherited = read(config["initial_stage_recovery_manifest"])["verified_local_files"]
    expected = {row["path"]: row for row in inherited}
    for row in component_expected:
        prior = expected.get(row["path"])
        if prior is not None:
            assert prior == row, "Conflicting component seal: " + row["path"]
        expected[row["path"]] = row
    for path in source_root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            resolved = str(path.resolve())
            expected.setdefault(resolved, None)
    for value in config.get("additional_local_dependencies", []):
        expected.setdefault(str(Path(value).resolve()), None)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(record, sorted(expected)))
    for row in actual:
        old = expected[row["path"]]
        if old is not None:
            assert row == old, "Previously sealed local file changed: " + row["path"]
    # Create only after every input and inherited byte comparison succeeds.
    args.output.mkdir(parents=True, exist_ok=False)
    utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest = {"version": "R5_CONTINUATION_RECOVERY_MANIFEST_V1", "status": "VERIFIED_LOCAL_FILES_AND_COMPONENT_RECEIPTS",
                "created_utc": utc, "primary_remote_root": "/workspace/r5", "verified_local_files": actual,
                "file_count": len(actual), "local_bytes": sum(r["bytes"] for r in actual),
                "module_recovery_seals": module_records, "scope_limits": config["recovery_scope_limits"],
                "component_sealed_local_files_verified": len(component_expected),
                "strict_CPU_acceptance_failures": failures, "original_remote_files_retained": True,
                "technical_degradations": config["technical_degradations"],
                "original_R4_and_initial_R5_boundaries": record(config["final_boundary_check"]),
                "primary_initial_stage_and_referenced_dependencies": record(config["primary_initial_boundary_check"]),
                "source_config": record(args.config), "source_code": record(__file__),
                "all_training_caches_and_optimizer_states_local": False,
                "all_original_pixels_local": False, "R4_evidence_modified": False}
    manifest_path = args.output / "RECOVERY_MANIFEST.json"
    write(manifest_path, manifest)
    gate = {"version": "R5_RELEASE_GATE_V1", "status": "HOLD", "utc": utc,
            "all_scientific_jobs_finished": True, "all_scientific_results_complete": False,
            "all_authorized_modules_accounted_for": True,
            "all_formal_new_closed_loop_trajectories": actual_trajectories,
            "completed_trajectories_by_module": counts,
            "technical_degradations": config["technical_degradations"],
            "formal_world_model_updates": 90000, "technical_world_model_updates": 101,
            "H1b_measurement_probe_updates": config["H1b_measurement_probe_updates"],
            "strict_CPU_acceptance_failures": failures, "recovery_scope_limits": config["recovery_scope_limits"],
            "manual_instance_release_recommended": False, "automatic_destroy": False,
            "reason": "Main and H3X scientific results are delivered. C2 is technically unevaluable after its unchanged render gate failed twice, with280 complete partial records retained and520 missing. Inherited and any new CPU tolerance failures remain unresolved; tolerances are not relaxed.",
            "module_recovery_seals": module_records, "manifest": record(manifest_path)}
    gate_path = args.output / "RELEASE_GATE.json"
    write(gate_path, gate)
    seal = {"version": "R5_FINAL_CONTINUATION_SEAL_V2", "status": "SEALED_WITH_C2_TECHNICAL_DEGRADATION_AND_CPU_RECOVERY_FAILURE",
            "sealed_utc": utc, "all_scientific_jobs_finished": True,
            "all_scientific_results_complete": False, "preset_H3": config["preset_H3_status"],
            "H3X_label": config["H3X_label"], "all_authorized_modules_accounted_for": True,
            "completed_trajectories_by_module": counts, "technical_degradations": config["technical_degradations"],
            "final_merged_tables": record(config["final_merge_status"]), "recovery_manifest": record(manifest_path),
            "release_gate": record(gate_path), "module_recovery_seals": module_records,
            "original_R4_and_initial_R5_seals_unchanged": True, "automatic_destroy": False,
            "automatic_rental": False, "original_remote_files_retained": True}
    write(args.output / "SEAL.json", seal)
    print(json.dumps({"status": seal["status"], "output": str(args.output.resolve()), "file_count": len(actual), "release_gate": "HOLD"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
