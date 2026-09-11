#!/usr/bin/env python3
"""Robustness probes (§5.3).

Currently implements the fully-deterministic **answer-order permutation**
probe for multiple-choice scenarios: each scenario is evaluated twice — once
as authored and once with its answer choices shuffled (seeded) — and we
measure whether the model's chosen option (by text, not letter) stays the
same. A position-insensitive model chooses the same option regardless of
where it appears; flips indicate option-position bias.

The correct answer is stored as the full choice TEXT, so permuting the choice
order does not change the gold answer and scoring is unaffected — only the
letter position changes.

Other §5.3 probes (semantics-preserving paraphrase, entity-name substitution,
critical-evidence ablation) are documented extension points: paraphrase and
entity substitution require an LLM/curated transform, and evidence ablation
requires identifying the governing sentence; each would plug in as an
additional ``--probe`` that produces a perturbed Scenario the same way.

Reuses App (mandatory Bedrock judge), the engine, and the scorer. Scoped to a
manifest subset to keep cost bounded.

Usage:
  PYTHONPATH=.. python scripts/robustness.py --probe answer_order \
      --manifest validation/manifests/rq3_challenge.json \
      --models nova-micro gpt-5.6-terra --limit 15 --seed 3
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT.parent))  # make `ele` importable

from ele.core.cli import App, AppConfig  # noqa: E402
from ele.core.engine import EvaluationConfig  # noqa: E402
from ele.core.models import AnswerFormatEnum  # noqa: E402

# Reuse run.py's adapter/config helpers.
sys.path.insert(0, str(_ROOT))
import run as _run  # noqa: E402


def _chosen_text(scenario, scored) -> Optional[str]:
    """Map the model's extracted letter to the choice TEXT in this scenario's ordering."""
    if scored is None:
        return None
    letter = (scored.extracted_answer or "").strip().upper()
    if len(letter) == 1 and "A" <= letter <= "Z" and scenario.choices:
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(scenario.choices):
            return scenario.choices[idx]
    return None


def _permute_choices(scenario, seed: int):
    """Return a deep copy of the scenario with its choices shuffled (seeded)."""
    perm = copy.deepcopy(scenario)
    rng = random.Random(f"{seed}:{scenario.id}")
    order = list(range(len(perm.choices)))
    # Reshuffle until the order actually changes (when >1 choice).
    for _ in range(10):
        rng.shuffle(order)
        if order != list(range(len(perm.choices))):
            break
    perm.choices = [scenario.choices[i] for i in order]
    return perm


def main() -> int:
    parser = argparse.ArgumentParser(description="ELE robustness probes (§5.3)")
    parser.add_argument("--probe", choices=["answer_order"], default="answer_order")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Max MC scenarios (0=all)")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--eval-config", default=str(_ROOT / "config" / "eval_config.json"))
    parser.add_argument("--models-config", default=str(_ROOT / "config" / "models.json"))
    args = parser.parse_args()

    app_config = (AppConfig.from_file(args.eval_config)
                  if Path(args.eval_config).exists() else AppConfig.from_env())
    app = App(app_config)

    # Register requested models.
    models_cfg = {m["id"]: m for m in _run.load_models_config(Path(args.models_config))}
    registered = {}
    for mid in args.models:
        if mid not in models_cfg:
            print(f"  ⚠ model '{mid}' not in config; skipping")
            continue
        adapter, err = _run.create_adapter(models_cfg[mid])
        if err:
            print(f"  ⚠ {mid}: {err}")
            continue
        app.register_model_instance(mid, adapter)
        registered[mid] = adapter
    if not registered:
        print("No models registered.")
        return 1

    # Load + submit the manifest subset (submit merges answer keys).
    manifest = json.loads(Path(args.manifest).read_text())
    wanted = {it["scenario_key"] for it in manifest.get("items", [])}
    scenarios = []
    for s in _run.load_scenarios(_ROOT / "scenarios"):
        if s.get("_source_file") not in wanted:
            continue
        res = app.submit_scenario(s, source_file=s.get("_source_file", ""))
        if res.get("success"):
            scn = app.repository.get_scenario(res["scenario_id"])
            if (scn and scn.answer_format == AnswerFormatEnum.MULTIPLE_CHOICE
                    and len(scn.choices or []) > 1 and not scn.tools_available):
                scenarios.append(scn)
    if args.limit:
        scenarios = scenarios[:args.limit]
    if not scenarios:
        print("No eligible multiple-choice scenarios in manifest.")
        return 1

    eval_config = EvaluationConfig(
        timeout_seconds=app_config.eval_timeout_seconds,
        max_tokens=app_config.eval_max_tokens,
        temperature=app_config.eval_temperature,
        enable_tools=False,
    )

    print(f"\n=== §5.3 answer-order robustness "
          f"({len(scenarios)} MC scenarios, {len(registered)} models) ===\n")
    report: Dict[str, Any] = {"probe": "answer_order", "n_scenarios": len(scenarios),
                              "seed": args.seed, "models": {}}

    for mid, model in registered.items():
        base_correct = perm_correct = flips = both_scored = 0
        for scn in scenarios:
            base = app.engine.execute_scenario(scn, model, eval_config)
            perm_scn = _permute_choices(scn, args.seed)
            perm = app.engine.execute_scenario(perm_scn, model, eval_config)
            bs = base.scored_result
            ps = perm.scored_result
            if bs and bs.is_correct:
                base_correct += 1
            if ps and ps.is_correct:
                perm_correct += 1
            bt = _chosen_text(scn, bs)
            pt = _chosen_text(perm_scn, ps)
            if bt is not None and pt is not None:
                both_scored += 1
                if bt != pt:
                    flips += 1
        n = len(scenarios)
        stability = (1 - flips / both_scored) * 100 if both_scored else None
        report["models"][mid] = {
            "base_accuracy": base_correct / n * 100,
            "permuted_accuracy": perm_correct / n * 100,
            "decision_flip_rate": (flips / both_scored * 100) if both_scored else None,
            "position_stability": stability,
            "n": n, "n_both_scored": both_scored,
        }
        print(f"{mid:<18} base={base_correct/n*100:5.1f}%  permuted={perm_correct/n*100:5.1f}%  "
              f"flip={ (flips/both_scored*100) if both_scored else float('nan'):5.1f}%  "
              f"stability={stability if stability is not None else float('nan'):5.1f}%")

    reports = _ROOT / "validation" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "robustness_answer_order.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nReport written: {reports / 'robustness_answer_order.json'}")
    print("flip = chosen option changed when choices were reordered (position sensitivity).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
