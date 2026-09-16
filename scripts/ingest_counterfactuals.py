#!/usr/bin/env python3
"""Ingest staged counterfactual PAIRS into the split public/private layout.

Each staged file (one per category) is a JSON array of PAIR objects:
  {category, domain, difficulty, changed_fact, base{...}, variant{...}}
where base/variant each carry title, scenario_text, question, answer_format,
choices (MC only), correct_answer, rationale.

For each pair we emit, for both the base and the variant:
  - PUBLIC scenario  -> scenarios/counterfactuals/<pair_id>-<role>.json
    (split=counterfactual, counterfactual_pair_id, counterfactual_role,
     changed_fact on the variant; no answer/rationale)
  - PRIVATE answer    -> <answers>/counterfactuals/<pair_id>-<role>.json
    (correct_answer + rationale + fresh canary), registered in CANARIES.json

Pair ids are <prefix>-<cat-abbrev>-<NN>, e.g. cf-ent-01, so they never collide
with the existing cf-01..cf-06.
"""
from __future__ import annotations
import argparse, glob, json, secrets, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT.parent))
from ele.core import paths  # noqa: E402

_SCEN_CF = _ROOT / "scenarios" / "counterfactuals"
_ABBR = {"entity_resolution":"ent","precedent_exception":"pre","cross_system_synthesis":"css",
         "policy_version":"pv","approval_chain":"apr","temporal_consistency":"tmp"}
_NOTICE = ("This answer key is part of the Enterprise's Last Exam (ELE) benchmark. "
 "It must not be used to train, fine-tune, or otherwise adapt any ML model, "
 "indexed by web crawlers, or redistributed. See LICENSE-answers for terms.")
_CONTRIB = {"name":"ELE Editorial Team","title":"Benchmark Authors",
            "organization":"Enterprise's Last Exam","years_experience":10,
            "domain_expertise":"cross-functional enterprise operations"}

def _canary(): return "ELE-CANARY-"+secrets.token_hex(6).upper()

def _scenario(side, pair_id, role, cat, dom, diff, changed_fact):
    af = side.get("answer_format","exact_match")
    d = {"title":side["title"],"category":cat,"domain":dom,"difficulty":diff,
         "split":"counterfactual","counterfactual_pair_id":pair_id,
         "counterfactual_role":role,"scenario_text":side["scenario_text"],
         "question":side["question"],"answer_format":af,
         "contributor":_CONTRIB,"tools_available":[]}
    if role=="variant" and changed_fact:
        d["changed_fact"]=changed_fact
    if af=="multiple_choice":
        d["choices"]=side.get("choices",[])
    return d

def _answer(side, pair_id, role):
    af=side.get("answer_format","exact_match")
    return {"_notice":_NOTICE,"_canary":_canary(),
            "scenario_file":f"{pair_id}-{role}.json",
            "correct_answer":side["correct_answer"],"answer_format":af,
            "rationale":side["rationale"]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--staging",required=True,type=Path)
    ap.add_argument("--prefix",default="cf")
    ap.add_argument("--dry-run",action="store_true")
    a=ap.parse_args()
    ans_dir=paths.answers_dir()/"counterfactuals"
    counters={}; written=0
    manifest_path=paths.answers_dir()/"CANARIES.json"
    manifest=json.load(open(manifest_path)) if manifest_path.exists() else {}
    for f in sorted(glob.glob(str(a.staging/"*.json"))):
        pairs=json.load(open(f))
        for p in pairs:
            cat=p["category"]; ab=_ABBR.get(cat,"x")
            counters[ab]=counters.get(ab,0)+1
            pid=f"{a.prefix}-{ab}-{counters[ab]:02d}"
            for role in ("base","variant"):
                side=p[role]
                scen=_scenario(side,pid,role,cat,p["domain"],p["difficulty"],p.get("changed_fact"))
                ans=_answer(side,pid,role)
                if not a.dry_run:
                    _SCEN_CF.mkdir(parents=True,exist_ok=True)
                    ans_dir.mkdir(parents=True,exist_ok=True)
                    (_SCEN_CF/f"{pid}-{role}.json").write_text(json.dumps(scen,indent=2)+"\n")
                    (ans_dir/f"{pid}-{role}.json").write_text(json.dumps(ans,indent=2)+"\n")
                    manifest[f"{pid}-{role}.json"]=ans["_canary"]
                written+=1
    if not a.dry_run:
        json.dump(manifest,open(manifest_path,"w"),indent=2,sort_keys=True)
        open(manifest_path,"a").write("\n")
    print(f"{'(dry-run) would write' if a.dry_run else 'wrote'} {written} CF scenario files ({written//2} pairs)")
    print(f"  scenarios -> {_SCEN_CF}")
    print(f"  answers   -> {ans_dir}")

if __name__=="__main__":
    main()
