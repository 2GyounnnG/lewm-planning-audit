"""W6 post hoc: predictor x history interaction at the other operating points.

Same estimator as scripts/w6_posthoc.py (case unit; three refit seeds and three
planner streams averaged within case; 5,000 case-bootstrap replicates; seed =
first 8 bytes of SHA-256("W6_CASE_BOOTSTRAP/<sample>"), little endian).
Samples: U1 (budget 100, offset 25; history set), U2 (budget 50, offset 50;
stress set), R7 stress (budget 100, offset 50; stress sets, Reacher and PushT).
"""
import csv, hashlib, json
from pathlib import Path
import numpy as np
B = 5000
REFITS = ["REFIT_103201", "REFIT_103202", "REFIT_103203"]; ST = ["R3_ORIGINAL", "R4_ALT_CEM_1", "R4_ALT_CEM_2"]
seed = lambda s: int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "little")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
HM = {"H_POLICY": 0, "H_REAL3_REPLAN": 1}
def load(path, task=None):
    S = {}
    for r in csv.DictReader(open(path)):
        if task and r.get("task") != task: continue
        if r.get("status", "COMPLETE") != "COMPLETE": continue
        S[(r["case_id"], r["arm"], r["stream"], HM[r["history"]])] = float(r["success"])
    return S
SAMPLES = [
    ("reacher_U1_budget100_offset25", "evidence/r8/reports/R8_M1_U1_RAW_VALUES.csv", None),
    ("reacher_U2_budget50_offset50", "evidence/r8/reports/R8_M1_U2_RAW_VALUES.csv", None),
    ("reacher_stress_budget100_offset50", "evidence/r7/reports/A_RAW_VALUES.csv", "reacher"),
    ("pusht_stress_budget100_offset50", "evidence/r7/reports/A_RAW_VALUES.csv", "pusht"),
]
rows, meta = [], {}
for name, path, task in SAMPLES:
    S = load(path, task); cases = sorted({k[0] for k in S}); n = len(cases)
    assert len(S) == n * 4 * 3 * 2, (name, len(S))
    d = {}
    for h in (0, 1):
        d[f"official_{h}"] = np.array([np.mean([S[(c, "H0", s, h)] for s in ST]) for c in cases])
        d[f"refit_{h}"] = np.array([np.mean([S[(c, a, s, h)] for a in REFITS for s in ST]) for c in cases])
        d[f"contrast_{h}"] = d[f"refit_{h}"] - d[f"official_{h}"]
    d["gain_official"] = d["official_1"] - d["official_0"]; d["gain_refit"] = d["refit_1"] - d["refit_0"]
    d["interaction"] = d["contrast_1"] - d["contrast_0"]
    idx = np.random.default_rng(seed(f"W6_CASE_BOOTSTRAP/{name}")).integers(0, n, size=(B, n))
    meta[name] = {"source": path, "task_filter": task, "source_sha256": sha(path), "cases": n}
    for k, v in d.items():
        bs = v[idx].mean(axis=1)
        rows.append({"sample": name, "quantity": k, "cases": n, "estimate_pp": round(100 * v.mean(), 2),
                     "ci95_low_pp": round(100 * np.percentile(bs, 2.5), 2), "ci95_high_pp": round(100 * np.percentile(bs, 97.5), 2), "evidence": "post hoc"})
out = Path("evidence/w6"); out.mkdir(exist_ok=True)
with open(out / "W6_INTERACTION_OPERATING_POINTS.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
meta["script_sha256"] = sha(__file__); meta["replicates"] = B
(out / "W6_INTERACTION_OPERATING_POINTS_META.json").write_text(json.dumps(meta, indent=2) + "\n")
for r in rows:
    if r["quantity"] in ("official_0", "official_1", "refit_0", "refit_1", "gain_official", "gain_refit", "contrast_0", "contrast_1", "interaction"):
        print(f'{r["sample"]:38s} {r["quantity"]:14s} {r["estimate_pp"]:7.2f} [{r["ci95_low_pp"]:6.2f}, {r["ci95_high_pp"]:6.2f}]')
