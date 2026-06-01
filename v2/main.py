"""
v2/main.py
----------
TraceLens v2 CLI — same flags as main.py, unified fusion architecture.

Usage:
    python main2.py
    python main2.py --llm --vlm --source areeb_salem
    python main2.py --trace github --llm --vlm
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from src.anomaly_detector import AnomalyDetector
from src.evaluation import Evaluator
from src.llm_reasoner import LlmReasoner
from src.ranker import Ranker
from src.report_generator import ReportGenerator
from src.screenshot_resolver import ScreenshotResolver
from src.vlm_reasoner import VlmReasoner

from v2.fusion import FusionWeights
from v2.runner import run_trace_v2


def load_config(base: Path, v2_overlay: Path | None = None) -> dict:
    with open(base, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    overlay_path = v2_overlay or (Path(__file__).parent / "config.yaml")
    if overlay_path.exists():
        with open(overlay_path, encoding="utf-8") as f:
            overlay = yaml.safe_load(f) or {}
        if "v2" in overlay:
            cfg.setdefault("v2", {}).update(overlay["v2"])
        if "outputs" in overlay:
            cfg["outputs"] = {**cfg.get("outputs", {}), **overlay["outputs"]}
    return cfg


def _print_overall_accuracy(agg: dict, excluded_count: int = 0) -> None:
    n = agg.get("n_traces", "?")
    fnd = agg.get("found_in_top5_count", "?")
    print("\n" + "=" * 50)
    print("OVERALL ACCURACY")
    print("=" * 50)
    if excluded_count:
        print(f"  Traces excluded (VLM failed): {excluded_count}")
    print(f"  Traces evaluated:            {n}  ({fnd} had fault in top-5)")
    print(f"  Hit@1  (fault ranked #1):    {round(agg.get('hit@1', 0) * 100, 1)}%")
    print(f"  Hit@3  (fault in top-3):     {round(agg.get('hit@3', 0) * 100, 1)}%")
    print(f"  Hit@5  (fault in top-5):     {round(agg.get('hit@5', 0) * 100, 1)}%")
    print(f"  Mean rank distance:          {agg.get('mean_rank_distance')}  (best=0, worst=5)")
    print(f"  Mean top-1 step distance:    {agg.get('mean_top1_step_distance')}  steps")
    print(f"  Mean MAD@5:                  {agg.get('mean_mad@5')}  steps")


def main() -> None:
    parser = argparse.ArgumentParser(description="TraceLens v2 pipeline (unified fusion)")
    parser.add_argument("--source", default=None, help="Filter by source")
    parser.add_argument("--trace", default=None, help="Filter by trace id")
    parser.add_argument("--skip", default=None, help="Comma-separated trace IDs to skip")
    parser.add_argument(
        "--from",
        default=None,
        dest="from_trace",
        help="Start from this trace ID (config order)",
    )
    parser.add_argument("--llm", action="store_true", help="Enable LLM rerank prior + diagnosis")
    parser.add_argument("--vlm", action="store_true", help="Enable VLM visual prior")
    parser.add_argument(
        "--no-pixel",
        action="store_true",
        help="Disable heuristic pixel boost in baseline tables",
    )
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.yaml"),
        help="Base config.yaml path",
    )
    args = parser.parse_args()

    skip_ids = set(args.skip.split(",")) if args.skip else set()
    cfg = load_config(Path(args.config))
    raw_base = cfg["data"]["raw_base"]

    with open(cfg["data"]["ground_truth"], encoding="utf-8") as f:
        ground_truth = json.load(f)

    detector = AnomalyDetector(weights=cfg["weights"])
    ranker = Ranker(
        top_k=cfg["ranking"]["top_k"],
        pre_llm_k=cfg["ranking"]["pre_llm_k"],
    )
    reporter = ReportGenerator()
    evaluator = Evaluator(top_k=cfg["ranking"]["top_k"])
    fusion_weights = FusionWeights.from_config(cfg)

    llm = None
    if args.llm:
        key_env = cfg["model"].get("api_key_env", "OPENROUTER_API_KEY")
        api_key = os.environ.get(key_env)
        if not api_key:
            print(f"ERROR: --llm requires {key_env} in .env")
            raise SystemExit(1)
        llm = LlmReasoner(
            api_key=api_key,
            model=cfg["model"]["llm_model"],
            temperature=cfg["model"]["temperature"],
            base_url=cfg["model"]["base_url"],
        )
        print(f"LLM enabled: {llm.model}")

    vlm = None
    resolver = ScreenshotResolver(raw_base=raw_base)
    if args.vlm:
        vlm_cfg = cfg.get("vlm", {})
        vlm_key_env = vlm_cfg.get("api_key_env", "OPENROUTER_API_KEY")
        vlm_api_key = os.environ.get(vlm_key_env)
        if not vlm_api_key:
            print(f"ERROR: --vlm requires {vlm_key_env} in .env")
            raise SystemExit(1)
        vlm = VlmReasoner(
            api_key=vlm_api_key,
            model=vlm_cfg.get("vlm_model", "google/gemma-4-31b-it:free"),
            temperature=vlm_cfg.get("temperature", 0.1),
            base_url=vlm_cfg.get("base_url", "https://openrouter.ai/api/v1"),
            per_step=vlm_cfg.get("per_step", True),
        )
        print(f"VLM enabled: {vlm.model}")

    print("TraceLens v2 — unified fusion ranking")
    if args.llm and args.vlm:
        print(f"Mode: v2 llm+vlm (fusion channels: text, causal, symptom, pixel, llm_prior, vlm_prior)")
    elif args.llm:
        print("Mode: v2 llm")
    elif args.vlm:
        print("Mode: v2 vlm")
    else:
        print("Mode: v2 heuristic")

    all_results = []
    trace_count = 0
    reached_from = args.from_trace is None

    for source, source_cfg in cfg["data"]["sources"].items():
        if args.source and source != args.source:
            continue
        for trace_cfg in source_cfg["traces"]:
            tid = trace_cfg["id"]
            if not reached_from:
                if tid == args.from_trace:
                    reached_from = True
                else:
                    print(f"  Skipping {source}/{tid} (before --from {args.from_trace})")
                    continue
            if args.trace and tid != args.trace:
                continue
            if tid in skip_ids:
                print(f"  Skipping {source}/{tid} (--skip)")
                continue
            if llm and trace_count > 0:
                time.sleep(5)
            trace_count += 1
            result = run_trace_v2(
                source=source,
                trace_cfg=trace_cfg,
                raw_base=raw_base,
                cfg=cfg,
                detector=detector,
                ranker=ranker,
                llm=llm,
                vlm=vlm,
                resolver=resolver,
                reporter=reporter,
                evaluator=evaluator,
                ground_truth=ground_truth,
                use_llm=args.llm,
                use_vlm=args.vlm,
                use_pixel=not args.no_pixel,
                use_eval=not args.no_eval,
                fusion_weights=fusion_weights,
            )
            all_results.append(result)

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.llm and args.vlm:
        run_mode = f"v2:llm+vlm ({cfg['model']['llm_model']} + {cfg.get('vlm', {}).get('vlm_model', 'vlm')})"
        run_prefix = "v2_llm+vlm"
    elif args.vlm:
        run_mode = f"v2:vlm ({cfg.get('vlm', {}).get('vlm_model', 'unknown')})"
        run_prefix = "v2_vlm"
    elif args.llm:
        run_mode = f"v2:llm ({cfg['model']['llm_model']})"
        run_prefix = "v2_llm"
    else:
        run_mode = "v2:heuristic"
        run_prefix = "v2_heuristic"

    if not args.no_eval:
        excluded_traces = [
            {
                "trace_id": r["trace_id"],
                "reason": r.get("vlm_error_code"),
                "error": r.get("vlm_error"),
            }
            for r in all_results
            if r.get("exclude_from_aggregate")
        ]
        eval_results = [
            r["eval"]
            for r in all_results
            if r.get("eval") and not r.get("exclude_from_aggregate")
        ]
        if eval_results:
            agg = evaluator.aggregate(eval_results)
            print("\n" + "=" * 50)
            print("AGGREGATE RESULTS")
            print("=" * 50)
            for k, v in agg.items():
                print(f"  {k}: {v}")
            if excluded_traces:
                print(f"  n_traces_excluded: {len(excluded_traces)}")
                for ex in excluded_traces:
                    print(f"    - {ex['trace_id']} ({ex.get('reason', 'unknown')})")

            per_trace_rows = [
                {
                    "trace_id": r["trace_id"],
                    **r["eval"],
                    "vlm_status": r.get("vlm_status"),
                    "exclude_from_aggregate": r.get("exclude_from_aggregate", False),
                }
                for r in all_results
                if r.get("eval")
            ]
            run_report = {
                "run_id": run_ts,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "mode": run_mode,
                "pipeline_version": "v2",
                "filter_source": args.source,
                "filter_trace": args.trace,
                "n_traces_run": trace_count,
                "n_traces_evaluated": len(eval_results),
                "n_traces_excluded": len(excluded_traces),
                "excluded_traces": excluded_traces,
                "aggregate": agg,
                "per_trace": per_trace_rows,
                "fusion_weights": fusion_weights.__dict__,
            }
            runs_dir = Path(cfg["outputs"]["metrics"]) / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            run_path = runs_dir / f"{run_prefix}_run_{run_ts}.json"
            with open(run_path, "w", encoding="utf-8") as f:
                json.dump(run_report, f, indent=2)
            print(f"\n  Run report saved → {run_path}")
            evaluator.save(agg, f"{cfg['outputs']['metrics']}/aggregate.json")
            evaluator.save(per_trace_rows, f"{cfg['outputs']['metrics']}/per_trace.json")
            _print_overall_accuracy(agg, excluded_count=len(excluded_traces))


if __name__ == "__main__":
    main()
