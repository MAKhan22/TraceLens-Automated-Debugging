"""
main.py
-------
TraceLens pipeline orchestrator.

Usage:
    # Run all traces
    python main.py

    # Run a single trace
    python main.py --source efe_irem --trace wikipedia

    # Heuristic-only (no LLM calls)
    python main.py --no-llm

    # Skip evaluation (if no ground truth)
    python main.py --no-eval
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from src.trace_parser import parse_trace, save_processed
from src.trace_aligner import align
from src.anomaly_detector import AnomalyDetector
from src.ranker import Ranker
from src.llm_reasoner import LlmReasoner
from src.report_generator import ReportGenerator
from src.evaluation import Evaluator


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_trace(
    source: str,
    trace_cfg: dict,
    raw_base: str,
    cfg: dict,
    detector: AnomalyDetector,
    ranker: Ranker,
    llm: LlmReasoner | None,
    reporter: ReportGenerator,
    evaluator: Evaluator,
    ground_truth: dict,
    use_llm: bool = True,
    use_eval: bool = True,
) -> dict:
    trace_id = f"{source}/{trace_cfg['id']}"
    print(f"\n{'='*50}")
    print(f"Processing: {trace_id}")

    source_base = Path(raw_base) / cfg["data"]["sources"][source]["base"]

    pass_path = str(source_base / trace_cfg["pass"])
    fail_path = str(source_base / trace_cfg["fail"])

    # 1. Parse
    passing = parse_trace(source, pass_path)
    failing = parse_trace(source, fail_path)
    print(f"  Parsed: {len(passing)} pass steps, {len(failing)} fail steps")

    # Save processed
    save_processed(passing, f"data/processed/{source}/{trace_cfg['id']}_pass.json")
    save_processed(failing, f"data/processed/{source}/{trace_cfg['id']}_fail.json")

    # 2. Align
    aligned = align(passing, failing)
    print(f"  Aligned: {len(aligned)} pairs")

    # 3. Score
    scored = detector.compute_scores(aligned)

    # 4. Rank (heuristic)
    candidates = ranker.candidates_for_llm(scored)   # top-10 for LLM
    heuristic_top = ranker.rank_heuristic(scored)    # top-5 heuristic

    heuristic_top = ranker.add_rank_metadata(heuristic_top)
    final_ranked = heuristic_top  # default: heuristic
    ranking_mode = "heuristic"

    # 5. LLM re-ranking + diagnosis
    llm_output = {}
    if use_llm and llm:
        try:
            print(f"  Running LLM re-ranking...")
            reranked_ids = llm.rerank(candidates)
            llm_ranked = ranker.apply_llm_reranking(reranked_ids, scored)
            llm_ranked = ranker.add_rank_metadata(llm_ranked)

            print(f"  Running LLM diagnosis...")
            llm_output = llm.run(candidates, llm_ranked)

            # If LLM diagnosis identifies a root cause step, promote it to #1
            # This ensures the ranking is consistent with the diagnosis
            rc_step = llm_output.get("diagnosis", {}).get("root_cause_step_id")
            if rc_step is not None:
                try:
                    rc_id = int(rc_step)
                    step_map = {s["step_id"]: s for s in scored}
                    if rc_id in step_map:
                        # Build new order: root cause first, then rest of LLM ranking
                        promoted = [step_map[rc_id]]
                        rest = [s for s in llm_ranked if s["step_id"] != rc_id]
                        final_ranked = ranker.add_rank_metadata(promoted + rest[:ranker.top_k - 1])
                        ranking_mode = "llm+diagnosis"
                    else:
                        final_ranked = llm_ranked
                        ranking_mode = "llm"
                except (ValueError, TypeError):
                    final_ranked = llm_ranked
                    ranking_mode = "llm"
            else:
                final_ranked = llm_ranked
                ranking_mode = "llm"

        except Exception as e:
            print(f"  LLM failed ({e}), falling back to heuristic ranking")
            final_ranked = heuristic_top
            ranking_mode = "heuristic (llm failed)"

    # 6. Evaluate
    eval_result = {}
    if use_eval:
        gt = ground_truth.get(source, {}).get(trace_cfg["id"], {})
        if gt:
            actual_fault = gt["fault_step"]
            eval_result = evaluator.evaluate_trace(final_ranked, actual_fault)
            print(f"  Hit@1={eval_result['hit@1']}  Hit@5={eval_result['hit@5']}  "
                  f"RankDist={eval_result['rank_distance']}  "
                  f"Top1StepDist={eval_result['top1_step_distance']}")
        else:
            print(f"  No ground truth for {trace_id}, skipping evaluation")

    # 7. Report
    slim_ranked = [{k: v for k, v in s.items() if k not in ("pass_step", "fail_step")}
                   for s in final_ranked]
    slim_heuristic = [{k: v for k, v in s.items() if k not in ("pass_step", "fail_step")}
                      for s in heuristic_top]

    report = reporter.build(
        trace_id=trace_id,
        ranked_steps=slim_ranked,
        heuristic_steps=slim_heuristic,
        ranking_mode=ranking_mode,
        llm_output=llm_output,
        eval_result=eval_result or None,
        metadata={"source": source, "fault_type": ground_truth.get(source, {})
                  .get(trace_cfg["id"], {}).get("fault_type")},
    )
    reporter.print_report(report)

    out_id = trace_cfg["id"]
    reporter.save_json(report, f"{cfg['outputs']['reports']}/{source}/{out_id}.json")
    reporter.save_text(report, f"{cfg['outputs']['reports']}/{source}/{out_id}.txt")
    ranker.save(slim_ranked, f"{cfg['outputs']['rankings']}/{source}/{out_id}.json")

    return {
        "trace_id":   trace_id,
        "eval":       eval_result,
        "llm_output": llm_output,
    }


def main():
    parser = argparse.ArgumentParser(description="TraceLens pipeline")
    parser.add_argument("--source", default=None, help="Filter by source (efe_irem/areeb_salem/ersel)")
    parser.add_argument("--trace",  default=None, help="Filter by trace id")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM calls (heuristic only)")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_base = cfg["data"]["raw_base"]

    with open(cfg["data"]["ground_truth"], encoding="utf-8") as f:
        ground_truth = json.load(f)

    detector = AnomalyDetector(weights=cfg["weights"])
    ranker   = Ranker(top_k=cfg["ranking"]["top_k"],
                      pre_llm_k=cfg["ranking"]["pre_llm_k"])
    reporter = ReportGenerator()
    evaluator = Evaluator(top_k=cfg["ranking"]["top_k"])

    llm = None
    if not args.no_llm:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("WARNING: GROQ_API_KEY not set. Running heuristic-only mode.")
        else:
            llm = LlmReasoner(
                api_key=api_key,
                model=cfg["model"]["llm_model"],
                temperature=cfg["model"]["temperature"],
                base_url=cfg["model"]["base_url"],
            )

    all_results = []
    trace_count = 0

    for source, source_cfg in cfg["data"]["sources"].items():
        if args.source and source != args.source:
            continue
        for trace_cfg in source_cfg["traces"]:
            if args.trace and trace_cfg["id"] != args.trace:
                continue
            # Inter-trace delay when using LLM to avoid Groq rate limits
            if llm and trace_count > 0:
                time.sleep(5)
            trace_count += 1
            result = run_trace(
                source=source,
                trace_cfg=trace_cfg,
                raw_base=raw_base,
                cfg=cfg,
                detector=detector,
                ranker=ranker,
                llm=llm,
                reporter=reporter,
                evaluator=evaluator,
                ground_truth=ground_truth,
                use_llm=(llm is not None),
                use_eval=(not args.no_eval),
            )
            all_results.append(result)

    # Aggregate evaluation + run report
    run_ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_mode = "heuristic" if args.no_llm else f"llm ({cfg['model']['llm_model']})"

    if not args.no_eval:
        eval_results = [r["eval"] for r in all_results if r.get("eval")]
        if eval_results:
            agg = evaluator.aggregate(eval_results)

            print("\n" + "=" * 50)
            print("AGGREGATE RESULTS")
            print("=" * 50)
            for k, v in agg.items():
                print(f"  {k}: {v}")

            # ── per-run timestamped report ─────────────────────────────────
            per_trace_rows = [
                {"trace_id": r["trace_id"], **r["eval"]}
                for r in all_results if r.get("eval")
            ]

            run_report = {
                "run_id":       run_ts,
                "timestamp":    datetime.now(timezone.utc).isoformat(),
                "mode":         run_mode,
                "filter_source": args.source,
                "filter_trace":  args.trace,
                "n_traces_run":  trace_count,
                "n_traces_evaluated": len(eval_results),
                "aggregate": agg,
                "per_trace": per_trace_rows,
            }

            runs_dir = Path(cfg["outputs"]["metrics"]) / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            run_path = runs_dir / f"run_{run_ts}.json"
            with open(run_path, "w", encoding="utf-8") as f:
                json.dump(run_report, f, indent=2)
            print(f"\n  Run report saved → {run_path}")

            # also overwrite the convenience latest files
            evaluator.save(agg, f"{cfg['outputs']['metrics']}/aggregate.json")
            evaluator.save(per_trace_rows,
                           f"{cfg['outputs']['metrics']}/per_trace.json")

            _print_overall_accuracy(agg)


def _print_overall_accuracy(agg: dict) -> None:
    n   = agg.get("n_traces", "?")
    fnd = agg.get("found_in_top5_count", "?")
    print("\n" + "=" * 50)
    print("OVERALL ACCURACY")
    print("=" * 50)
    print(f"  Traces evaluated:            {n}  ({fnd} had fault in top-5)")
    print(f"  Hit@1  (fault ranked #1):    {round(agg.get('hit@1', 0) * 100, 1)}%")
    print(f"  Hit@3  (fault in top-3):     {round(agg.get('hit@3', 0) * 100, 1)}%")
    print(f"  Hit@5  (fault in top-5):     {round(agg.get('hit@5', 0) * 100, 1)}%")
    print(f"  Mean rank distance:          {agg.get('mean_rank_distance')}  (best=0, worst=5)")
    print(f"  Mean top-1 step distance:    {agg.get('mean_top1_step_distance')}  steps")
    print(f"  Mean MAD@5:                  {agg.get('mean_mad@5')}  steps")


if __name__ == "__main__":
    main()
