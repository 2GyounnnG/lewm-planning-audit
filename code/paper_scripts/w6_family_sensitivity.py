"""W6: family-cluster bootstrap sensitivity for PushT contrasts (post hoc check).

Families are resampled with replacement; the estimate is the sum of case-level
contrasts in the sampled families divided by the number of sampled cases
(the R4 family rule). 5,000 replicates; seed = first 8 bytes of
SHA-256("W6_FAMILY/<name>"), little endian. Case-bootstrap intervals use the
label "W6_CASE_BOOTSTRAP/<name>" for comparison.
"""
import csv, json, hashlib, collections, sys
from pathlib import Path
import numpy as np
B = 5000
REFITS = ["REFIT_103201", "REFIT_103202", "REFIT_103203"]; ST = ["R3_ORIGINAL", "R4_ALT_CEM_1", "R4_ALT_CEM_2"]
seed = lambda s: int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "little")
sel = Path(sys.argv[1])  # M2_PUSHT_CASE_SELECTION.json
fam_m2 = {c["case_id"]: c["family_id"] for c in json.load(open(sel))["cases"]}
m2 = {(r["case_id"], r["arm"], r["stream"]): float(r["success"]) for r in csv.DictReader(open("evidence/r8/m2/R8_M2_PUSHT_RAW_VALUES.csv"))}
cases = sorted({k[0] for k in m2})
v_m2 = {c: np.mean([m2[(c, a, s)] for a in REFITS for s in ST]) - np.mean([m2[(c, "H0", s)] for s in ST]) for c in cases}
m4 = list(csv.DictReader(open("evidence/r8/m4/M4_ALL_RAW_VALUES.csv")))
fam_m4 = {r["case_id"]: r["family_id"] for r in m4}
S4 = collections.defaultdict(list)
for r in m4: S4[(r["case_id"], r["arm"])].append(float(r["success"]))
c4 = sorted(fam_m4)
v_m4 = {c: np.mean(S4[(c, "SIM_LAT_RERANK")]) - np.mean(S4[(c, "H0_MENU_RERANK")]) for c in c4}
def run(name, vals, fam):
    cs = sorted(vals); x = np.array([vals[c] for c in cs]); n = len(x)
    rng = np.random.default_rng(seed(f"W6_CASE_BOOTSTRAP/{name}")); bs = x[rng.integers(0, n, size=(B, n))].mean(1)
    groups = collections.defaultdict(list)
    for i, c in enumerate(cs): groups[fam[c]].append(i)
    G = list(groups.values()); sums = np.array([x[g].sum() for g in G]); cnt = np.array([len(g) for g in G])
    rng = np.random.default_rng(seed(f"W6_FAMILY/{name}")); idx = rng.integers(0, len(G), size=(B, len(G)))
    fb = sums[idx].sum(1) / cnt[idx].sum(1)
    out = dict(sample=name, cases=n, families=len(G), estimate_pp=round(100 * x.mean(), 2),
               case_ci=[round(100 * np.percentile(bs, q), 2) for q in (2.5, 97.5)],
               family_ci=[round(100 * np.percentile(fb, q), 2) for q in (2.5, 97.5)])
    print(json.dumps(out)); return out
res = [run("pusht_fresh_refit_set_family", v_m2, fam_m2), run("pusht_m4_oracle_family", v_m4, fam_m4)]
Path("evidence/w6/W6_FAMILY_SENSITIVITY.json").write_text(json.dumps(res, indent=2) + "\n")
