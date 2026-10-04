#!/usr/bin/env python3
"""Audit numeric tokens in the manuscript against CLAIM_EVIDENCE.

This is a traceability screen, not a semantic proof: protocol literals and
bibliographic metadata can be unmatched. It records every unmatched token with
its source file and line so an author can inspect it.
"""
from __future__ import annotations
import csv, json, re
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUM = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", re.I)

def norm(s: str) -> str:
    if len(re.sub(r'\D', '', s)) > 8:
        return '__long__'
    try:
        d = Decimal(s)
        if d == 0: return '0'
        return format(d.normalize(), 'f')
    except InvalidOperation:
        return s

def strip_comments(text):
    return '\n'.join(line.split('%',1)[0] for line in text.splitlines())

files = [ROOT/'main.tex'] + sorted((ROOT/'sections').glob('*.tex')) + sorted((ROOT/'tables').glob('*.tex'))
entries=[]
for f in files:
    for ln, raw in enumerate(f.read_text(errors='replace').splitlines(), 1):
        line=raw.split('%',1)[0]
        # Ignore numeric metadata embedded in citation keys, labels, ORCID links,
        # graphics options and cross-reference commands.  These are not
        # manuscript claims and should not be mistaken for reported results.
        line = line.replace('--', ' to ')
        line = re.sub(r'\\(?:cite|citet|citep|citeauthor|orcidlink|label|ref|eqref)\{[^}]*\}', '', line)
        line = re.sub(r'\\includegraphics(?:\[[^]]*\])?\{[^}]*\}', '', line)
        for m in NUM.finditer(line):
            entries.append({'file':str(f.relative_to(ROOT)),'line':ln,'token':m.group(),'normalized':norm(m.group())})

claims=list(csv.DictReader((ROOT/'CLAIM_EVIDENCE.csv').open(newline='')))
claim_numbers={row['claim_id']:{norm(m.group()) for m in NUM.finditer(' '.join(str(v) for v in row.values()))} for row in claims}
for e in entries:
    e['matched_claim_ids']=sorted(cid for cid, nums in claim_numbers.items() if e['normalized'] in nums)
matched=[e for e in entries if e['matched_claim_ids']]
unmatched=[e for e in entries if not e['matched_claim_ids']]
# unique unmatched tokens with locations, keeping all occurrences in JSON
out={'manuscript_files':[str(f.relative_to(ROOT)) for f in files],
     'numeric_tokens_total':len(entries),
     'numeric_tokens_matched_to_claim_evidence':len(matched),
     'numeric_tokens_unmatched':len(unmatched),
     'matched_fraction':round(len(matched)/len(entries),4) if entries else 1.0,
     'unmatched_occurrences':unmatched,
     'unique_unmatched_tokens':sorted({e['normalized'] for e in unmatched}, key=lambda x:(float(x) if re.fullmatch(r'-?\d+(?:\.\d+)?',x) else 0,x))}
(ROOT/'number_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ('numeric_tokens_total','numeric_tokens_matched_to_claim_evidence','numeric_tokens_unmatched','matched_fraction')},ensure_ascii=False))
