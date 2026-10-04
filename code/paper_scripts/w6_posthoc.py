"""W6 post hoc analyses from sealed raw tables (no trajectories rerun, no training).

1. Predictor x history interaction (Reacher original, Reacher fresh history set,
   PushT original): I_i = (refit - official | real history) - (refit - official | single frame),
   equivalently the refit's history gain minus the official predictor's.
2. Single-frame refit - official success on each Reacher/PushT case set and pooled
   over case sets (stratified case bootstrap: cases resampled within each set).

Unit: case. Within a case the three refit seeds and the three planner streams are
averaged first. 5,000 replicates; percentile 95% intervals; one shared index draw per
sample for all quantities of that sample. Seed = first 8 bytes of SHA-256(label),
little endian. Labels: W6_CASE_BOOTSTRAP/<sample>, W6_POOLED/<task>.
"""
import csv, hashlib, json
from pathlib import Path
import numpy as np

B = 5000
REFITS = ["REFIT_103201", "REFIT_103202", "REFIT_103203"]
STREAMS = ["R3_ORIGINAL", "R4_ALT_CEM_1", "R4_ALT_CEM_2"]
OUT = Path("evidence/w6"); OUT.mkdir(parents=True, exist_ok=True)
seed = lambda s: int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "little")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()

def load_r5(path, task):
    S = {}
    for r in csv.DictReader(open(path)):
        if r["task"] == task:
            S[(r["case_id"], r["arm"], r["stream"], 0)] = float(r["control_success"])
            S[(r["case_id"], r["arm"], r["stream"], 1)] = float(r["intervention_success"])
    return S

def load_hist(path):
    hmap = {"H_POLICY": 0, "H_REAL3_REPLAN": 1}
    return {(r["case_id"], r["arm"], r["stream"], hmap[r["history"]]): float(r["success"])
            for r in csv.DictReader(open(path)) if r.get("status", "COMPLETE") == "COMPLETE"}

def per_case(S, histories):
    cases = sorted({k[0] for k in S})
    d = {}
    for h in histories:
        d[f"official_{h}"] = np.array([np.mean([S[(c, "H0", s, h)] for s in STREAMS]) for c in cases])
        d[f"refit_{h}"] = np.array([np.mean([S[(c, a, s, h)] for a in REFITS for s in STREAMS]) for c in cases])
        d[f"contrast_{h}"] = d[f"refit_{h}"] - d[f"official_{h}"]
    if 1 in histories:
        d["gain_official"] = d["official_1"] - d["official_0"]
        d["gain_refit"] = d["refit_1"] - d["refit_0"]
        d["interaction"] = d["contrast_1"] - d["contrast_0"]
    return cases, d

SAMPLES = [
    ("reacher_original", "evidence/r5/REAL_HISTORY_REPLANNING_RAW.csv", lambda p: load_r5(p, "reacher"), (0, 1)),
    ("reacher_fresh_history_set", "evidence/r6/R6_RAW_VALUES.csv", load_hist, (0, 1)),
    ("pusht_original", "evidence/r5/REAL_HISTORY_REPLANNING_RAW.csv", lambda p: load_r5(p, "pusht"), (0, 1)),
    ("reacher_fresh_refit_set", "evidence/r8/m2/R8_M2_REACHER_RAW_VALUES.csv", load_hist, (0,)),
    ("pusht_fresh_refit_set", "evidence/r8/m2/R8_M2_PUSHT_RAW_VALUES.csv", load_hist, (0,)),
]
rows, meta, single = [], {}, {}
for name, path, loader, hs in SAMPLES:
    S = loader(path); cases, d = per_case(S, hs); n = len(cases)
    label = f"W6_CASE_BOOTSTRAP/{name}"
    idx = np.random.default_rng(seed(label)).integers(0, n, size=(B, n))
    meta[name] = {"source": path, "source_sha256": sha(path), "cases": n, "cells": len(S), "bootstrap_label": label}
    for k, v in d.items():
        bs = v[idx].mean(axis=1)
        rows.append({"sample": name, "quantity": k, "cases": n, "estimate_pp": round(100 * v.mean(), 2),
                     "ci95_low_pp": round(100 * np.percentile(bs, 2.5), 2),
                     "ci95_high_pp": round(100 * np.percentile(bs, 97.5), 2), "evidence": "post hoc"})
    single[name] = d["contrast_0"]
for task, names in [("reacher", ["reacher_original", "reacher_fresh_history_set", "reacher_fresh_refit_set"]),
                    ("pusht", ["pusht_original", "pusht_fresh_refit_set"])]:
    rng = np.random.default_rng(seed(f"W6_POOLED/{task}"))
    vecs = [single[n] for n in names]; total = sum(len(v) for v in vecs)
    reps = sum(v[rng.integers(0, len(v), size=(B, len(v)))].sum(axis=1) for v in vecs) / total
    allv = np.concatenate(vecs)
    rows.append({"sample": f"{task}_pooled_" + "+".join(names), "quantity": "contrast_0", "cases": total,
                 "estimate_pp": round(100 * allv.mean(), 2), "ci95_low_pp": round(100 * np.percentile(reps, 2.5), 2),
                 "ci95_high_pp": round(100 * np.percentile(reps, 97.5), 2), "evidence": "post hoc"})
    meta[f"{task}_pooled"] = {"bootstrap_label": f"W6_POOLED/{task}", "strata": names}
with open(OUT / "W6_POSTHOC_TABLE.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
meta["script_sha256"] = sha(__file__); meta["replicates"] = B
meta["quantity_key"] = {"official_0/refit_0": "single-frame success", "official_1/refit_1": "real-history success",
                        "contrast_h": "refit minus official at history h", "gain_*": "real minus single for that predictor",
                        "interaction": "contrast_1 - contrast_0 = gain_refit - gain_official"}
(OUT / "W6_POSTHOC_META.json").write_text(json.dumps(meta, indent=2) + "\n")
for r in rows: print(r["sample"], r["quantity"], r["estimate_pp"], [r["ci95_low_pp"], r["ci95_high_pp"]])
