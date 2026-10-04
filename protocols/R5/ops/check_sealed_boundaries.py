"""Read-only verification of original R4 and R5 first-stage sealed bytes."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(row):
    path = Path(row["path"])
    if not path.is_file():
        return {"path": str(path), "difference": "MISSING"}
    if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
        return {"path": str(path), "difference": "SEALED_BYTES_DIFFER"}
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = {
        "r4_v23_execution/SEAL.json": "c86fd600da69ee5b735563527e9456cc2bbd3e998fcda13597628fea87999f29",
        "r4_v23_execution/RECOVERY_MANIFEST.json": "28f4020a1c4b2433ef7ea5efb5376a115a2b9d8e4542a5518b583e48cd3a96a6",
        "r4_v23_execution/RELEASE_GATE.json": "572e250ef249766ead4d1d215b5e9b6262a8ae639af2dbf757c9822fc5388622",
        "r5_execution/ops/INITIAL_STAGE_SEAL.json": "a982abb1fd8d4472b909158f96106bcfec89d1f88756dbd6d946a785db68937e",
        "r5_execution/ops/INITIAL_STAGE_RECOVERY_MANIFEST.json": "6b86762eb2d260d07a8866b7929711d811d1bccd6269e0c69784db9256830897",
        "r5_execution/ops/INITIAL_STAGE_DELIVERY_INDEX.json": "3f5c638ba390796b8dc3a2f2729ae2b7ab276a94a3bf03c1aab1bb1aa398f47a",
    }
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    boundaries = {name: sha(args.workspace / name) for name in expected}
    if boundaries != expected:
        raise RuntimeError("An original seal/manifest boundary differs; no automatic repair")
    checks = {}
    for name in ("r4_v23_execution/RECOVERY_MANIFEST.json", "r5_execution/ops/INITIAL_STAGE_RECOVERY_MANIFEST.json"):
        rows = json.loads((args.workspace / name).read_text())["verified_local_files"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            differences = [r for r in pool.map(verify, rows) if r is not None]
        checks[name] = {"checked_files": len(rows), "recorded_bytes": sum(r["bytes"] for r in rows), "differences": differences}
    assert {name: sha(args.workspace / name) for name in expected} == expected
    result = {"status": "PASS" if not any(r["differences"] for r in checks.values()) else "FAIL",
              "started_utc": started, "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "original_boundary_sha256": expected, "checks": checks, "audit_code_sha256": sha(__file__),
              "source_files_written": 0, "automatic_repairs": 0,
              "scope": "All current bytes match original sealed local records; this is not a claim that no historical transient modification ever occurred"}
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False))
