#!/usr/bin/env python3
"""Repeated-trial stability (§5.5).

Re-runs a stratified subset K times per model under identical settings and
quantifies stochastic decision instability. At temperature 0 many providers
are near-deterministic, but residual nondeterminism (sampling, backend
routing, reasoning models) can still flip decisions; this measures how often.

Per model it reports:
  - unanimous_rate: fraction of scenarios where all K trials chose the same
    option (perfectly stable items).
  - mean_agreement: average within-scenario agreement = for each scenario the
    modal-choice share across the K trials, averaged over scenarios.
  - majority_accuracy: accuracy using each scenario's majority-vote decision.

Reuses App (mandatory Bedrock judge), the engine, and the scorer. Scoped to a
manifest subset and a small K to keep cost bounded.

Usage:
  PYTHONPATH=.. python scripts/stability.py \
      --manifest validation/manifests/rq3_challenge.json \
      --models nova-micro gpt-5.6-terra --trials 3 --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT.parent))  # make `ele` importable

from ele.core.cli import App, AppConfig  # noqa: E402
from ele.core.engine import EvaluationConfig  # noqa: E402
from ele.core.models import AnswerFormatEnum  # noqa: E402

sys.path.insert(0, str(_ROOT))
import run as _run  # noqa: E402


def _decision_key(scenario, scored) -> Optional[str]:
    """A stable key for the model's decision: the chosen choice text (MC) or extracted text."""
    if scored is None:
        return None
    if scenario.answer_format == AnswerFormatEnum.MULTIPLE_CHOICE and scenario.choices:
        letter = (scored.extracted_answer or "").strip().upper()
        if len(letter) == 1 and "A" <= letter <= "Z":
            idx = ord(letter) - ord("A")
            if 0 <= idx < len(scenario.choices):
                return scenario.choices[idx]
        return None
    return (scored.extracted_answer or "").strip().lower() or None


def main() -> int:
    parser = argparse.ArgumentParser(description="ELE repeated-trial stability (§5.5)")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--eval-config", default=str(_ROOT / "config" / "eval_config.json"))
    parser.add_argument("--models-config", default=str(_ROOT / "config" / "models.json"))
    args = parser.parse_args()

    app_config = (AppConfig.from_file(args.eval_config)
                  if Path(args.eval_config).exists() else AppConfig.from_env())
    app = App(app_config)

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

    manifest = json.loads(Path(args.manifest).read_text())
    wanted = {it["scenario_key"] for it in manifest.get("items", [])}
    scenarios = []
    for s in _run.load_scenarios(_ROOT / "scenarios"):
        if s.get("_source_file") not in wanted:
            continue
        res = app.submit_scenario(s, source_file=s.get("_source_file", ""))
        if res.get("success"):
            scn = app.repository.get_scenario(res["scenario_id"])
            if scn and not scn.tools_available:
                scenarios.append(scn)
    if args.limit:
        scenarios = scenarios[:args.limit]
    if not scenarios:
        print("No eligible scenarios in manifest.")
        return 1

    eval_config = EvaluationConfig(
        timeout_seconds=app_config.eval_timeout_seconds,
        max_tokens=app_config.eval_max_tokens,
        temperature=app_config.eval_temperature,
        enable_tools=False,
    )

    K = args.trials
    print(f"\n=== §5.5 repeated-trial stability "
          f"({len(scenarios)} scenarios × {K} trials, {len(registered)} models) ===\n")
    report: Dict[str, Any] = {"n_scenarios": len(scenarios), "trials": K, "models": {}}

    for mid, model in registered.items():
        unanimous = 0
        agreements: List[float] = []
        majority_correct = 0
        scored_scenarios = 0
        for scn in scenarios:
            decisions: List[str] = []
            correctness: List[bool] = []
            for _ in range(K):
                r = app.engine.execute_scenario(scn, model, eval_config)
                key = _decision_key(scn, r.scored_result)
                if key is not None:
                    decisions.append(key)
                if r.scored_result is not None:
                    correctness.append(bool(r.scored_result.is_correct))
            if not decisions:
                continue
            scored_scenarios += 1
            counts = Counter(decisions)
            modal, modal_n = counts.most_common(1)[0]
            agreements.append(modal_n / len(decisions))
            if len(counts) == 1 and len(decisions) == K:
                unanimous += 1
            # majority-vote correctness: correct if the modal decision was a correct one
            # (approximated by majority of the per-trial correctness flags)
            if correctness and sum(correctness) >= (len(correctness) / 2):
                majority_correct += 1
        n = scored_scenarios or 1
        report["models"][mid] = {
            "n_scenarios_scored": scored_scenarios,
            "unanimous_rate": unanimous / n * 100,
            "mean_agreement": (sum(agreements) / len(agreements) * 100) if agreements else None,
            "majority_accuracy": majority_correct / n * 100,
        }
        m = report["models"][mid]
        print(f"{mid:<18} unanimous={m['unanimous_rate']:5.1f}%  "
              f"mean_agreement={m['mean_agreement']:5.1f}%  "
              f"majority_acc={m['majority_accuracy']:5.1f}%  (n={scored_scenarios})")

    reports = _ROOT / "validation" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "stability.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nReport written: {reports / 'stability.json'}")
    print("unanimous = all K trials chose the same option; mean_agreement = modal-choice share.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
