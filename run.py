#!/usr/bin/env python3
"""
AI Model Evaluation Workflow — Main Runner

Usage:
  python -m evaluation_workflow.run                          # evaluate all models against all scenarios
  python -m evaluation_workflow.run --model gpt-4o-mini      # evaluate one model
  python -m evaluation_workflow.run --scenario 005_*.json    # run specific scenario(s)
  python -m evaluation_workflow.run --category cross_system_synthesis  # filter scenarios
  python -m evaluation_workflow.run --difficulty expert       # filter by difficulty
  python -m evaluation_workflow.run --scenarios-dir ./my_scenarios     # custom scenario folder
  python -m evaluation_workflow.run --list-scenarios          # just list loaded scenarios
  python -m evaluation_workflow.run --validate-only           # validate scenarios without running

Scenarios:  evaluation_workflow/scenarios/*.json
Models:     evaluation_workflow/config/models.json
Settings:   evaluation_workflow/config/eval_config.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load .env files
load_dotenv("linkedin_ai_manager/.env")
load_dotenv(".env")

from evaluation_workflow.cli import App, AppConfig
from evaluation_workflow.models_integration import APIConfig, OpenAIAdapter


# ── Defaults ──────────────────────────────────────────────────────

_ROOT = Path(__file__).parent
DEFAULT_SCENARIOS_DIR = _ROOT / "scenarios"
DEFAULT_MODELS_CONFIG = _ROOT / "config" / "models.json"
DEFAULT_EVAL_CONFIG = _ROOT / "config" / "eval_config.json"


# ── Helpers ───────────────────────────────────────────────────────

def load_scenarios(directory: Path) -> List[Dict[str, Any]]:
    """Load all .json scenario files from a directory (skips TEMPLATE)."""
    files = sorted(glob.glob(str(directory / "*.json")))
    scenarios = []
    for f in files:
        if Path(f).stem.upper() == "TEMPLATE":
            continue
        with open(f) as fh:
            scenarios.append(json.load(fh))
    return scenarios


def load_models_config(path: Path) -> List[Dict[str, Any]]:
    """Load model definitions from config file."""
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("models", [])


def create_adapter(model_cfg: Dict[str, Any]):
    """Create a model adapter from config."""
    provider = model_cfg.get("provider", "openai")
    model_name = model_cfg.get("model_name", "gpt-4o-mini")

    # Resolve API key from env var name
    api_key_env = model_cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = model_cfg.get("api_key", "") or os.environ.get(api_key_env, "")
    base_url = model_cfg.get("base_url", "")

    if not api_key:
        return None, f"No API key found (checked env var {api_key_env})"

    api_config = APIConfig(api_key=api_key, base_url=base_url, timeout_seconds=60)

    if provider == "openai":
        return OpenAIAdapter(model_id=model_name, api_config=api_config), None
    else:
        return None, f"Unsupported provider: {provider}"


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI Model Evaluation Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", help="Run only this model ID")
    parser.add_argument("--category", help="Filter scenarios by category")
    parser.add_argument("--domain", help="Filter scenarios by domain")
    parser.add_argument("--difficulty", help="Filter scenarios by difficulty")
    parser.add_argument("--scenario", action="append", default=None,
                        help="Run specific scenario file(s). Can be repeated: --scenario 005_*.json --scenario 001_*.json")
    parser.add_argument("--scenarios-dir", default=str(DEFAULT_SCENARIOS_DIR),
                        help="Directory containing scenario JSON files")
    parser.add_argument("--models-config", default=str(DEFAULT_MODELS_CONFIG),
                        help="Path to models.json config")
    parser.add_argument("--eval-config", default=str(DEFAULT_EVAL_CONFIG),
                        help="Path to eval_config.json")
    parser.add_argument("--list-scenarios", action="store_true",
                        help="List loaded scenarios and exit")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate scenarios without running evaluation")
    args = parser.parse_args()

    # Load eval config
    eval_config_path = Path(args.eval_config)
    if eval_config_path.exists():
        app_config = AppConfig.from_file(str(eval_config_path))
    else:
        app_config = AppConfig.from_env()

    app = App(app_config)

    # ── Load scenarios ────────────────────────────────────────────
    scenarios_dir = Path(args.scenarios_dir)

    if args.scenario:
        # Load specific scenario files (supports glob patterns)
        raw_scenarios = []
        for pattern in args.scenario:
            matched = sorted(glob.glob(pattern))
            if not matched:
                # Try inside scenarios_dir
                matched = sorted(glob.glob(str(scenarios_dir / pattern)))
            if not matched:
                print(f"No files matching: {pattern}")
                return 1
            for f in matched:
                with open(f) as fh:
                    raw_scenarios.append(json.load(fh))
        print(f"Loaded {len(raw_scenarios)} scenario file(s) from command line\n")
    else:
        if not scenarios_dir.exists():
            print(f"Scenarios directory not found: {scenarios_dir}")
            return 1
        raw_scenarios = load_scenarios(scenarios_dir)
        if not raw_scenarios:
            print(f"No scenario files found in {scenarios_dir}")
            return 1
        print(f"Found {len(raw_scenarios)} scenario file(s) in {scenarios_dir}\n")

    # ── List mode ─────────────────────────────────────────────────
    if args.list_scenarios:
        for i, s in enumerate(raw_scenarios, 1):
            print(f"  {i}. {s.get('title', 'Untitled')}")
            print(f"     Category: {s.get('category')}  Domain: {s.get('domain')}  Difficulty: {s.get('difficulty')}")
            print(f"     Format: {s.get('answer_format')}  Contributor: {s.get('contributor', {}).get('name', 'Unknown')}")
            print()
        return 0

    # ── Submit scenarios ──────────────────────────────────────────
    submitted = 0
    for s in raw_scenarios:
        result = app.submit_scenario(s)
        title = s.get("title", "Untitled")
        if result.get("success"):
            submitted += 1
            if args.validate_only:
                print(f"  ✓ {title}")
            else:
                print(f"  ✓ Submitted: {title} ({result['scenario_id'][:8]}...)")
        else:
            errors = result.get("errors", [])
            print(f"  ✗ Rejected: {title}")
            for e in errors:
                if isinstance(e, dict):
                    print(f"      {e.get('field', '?')}: {e.get('message', '?')}")
                else:
                    print(f"      {e}")

    print(f"\n{submitted}/{len(raw_scenarios)} scenarios loaded successfully")

    if args.validate_only:
        return 0 if submitted == len(raw_scenarios) else 1

    if submitted == 0:
        print("No valid scenarios to evaluate.")
        return 1

    # ── Load models ───────────────────────────────────────────────
    models_config = load_models_config(Path(args.models_config))
    if not models_config:
        print(f"\nNo models configured in {args.models_config}")
        print("Create a models.json with your model definitions.")
        return 1

    # Filter to requested model if specified
    if args.model:
        models_config = [m for m in models_config if m["id"] == args.model]
        if not models_config:
            print(f"\nModel '{args.model}' not found in config.")
            return 1

    registered = []
    for mcfg in models_config:
        adapter, err = create_adapter(mcfg)
        if err:
            print(f"\n  ⚠ Skipping {mcfg['id']}: {err}")
            continue
        reg = app.register_model_instance(mcfg["id"], adapter)
        if reg.get("success"):
            registered.append(mcfg["id"])
            print(f"\n  ✓ Registered model: {mcfg['id']} ({mcfg.get('model_name', '?')})")
        else:
            print(f"\n  ✗ Failed to register {mcfg['id']}: {reg.get('error')}")

    if not registered:
        print("\nNo models registered. Check your API keys and config.")
        return 1

    # ── Run evaluations ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RUNNING EVALUATIONS")
    print("=" * 60)

    for model_id in registered:
        print(f"\n── Evaluating: {model_id} ──")
        eval_result = app.run_evaluation(
            model_id=model_id,
            category=args.category,
            domain=args.domain,
            difficulty=args.difficulty,
        )

        if not eval_result.get("success"):
            print(f"  Error: {eval_result.get('error')}")
            continue

        print(f"  Completed: {eval_result['completed']}/{eval_result['total_scenarios']} scenarios")
        print(f"  Accuracy:  {eval_result['overall_accuracy']:.1f}%")

        # Show per-scenario details
        run = app.engine.get_run(eval_result["run_id"])
        if run:
            for r in run.results:
                sr = r.scored_result
                scenario = app.repository.get_scenario(r.scenario_id)
                title = scenario.title if scenario else r.scenario_id[:8]
                status = "✓" if (sr and sr.final_score >= 0.5) else "✗"
                score = f"{sr.final_score:.1f}" if sr else "?"
                print(f"    {status} {title}: score={score}, latency={r.latency_ms}ms, tokens={r.tokens_used}")

    # ── Leaderboard ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LEADERBOARD")
    print("=" * 60)
    lb = app.leaderboard()
    if lb:
        print(f"  {'Model':<20} {'Accuracy':>10} {'Exact Match':>12} {'Avg Latency':>12}")
        print(f"  {'─' * 20} {'─' * 10} {'─' * 12} {'─' * 12}")
        for entry in lb:
            print(f"  {entry['model_id']:<20} {entry['accuracy']:>9.1f}% {entry['exact_match_rate']:>11.1f}% {entry['average_latency_ms']:>10.0f}ms")
    else:
        print("  No results yet.")

    # ── Export results ────────────────────────────────────────────
    results_dir = _ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    for entry in lb:
        run_id = entry["run_id"]
        export = app.export_results(run_id, "json")
        if export:
            out_path = results_dir / f"{entry['model_id']}_{run_id[:8]}.json"
            with open(out_path, "w") as f:
                f.write(export)
            print(f"\n  Results saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
