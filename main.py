"""
main.py
-------
TraceLens pipeline orchestrator.

Usage:
    python main.py                          # heuristic only (default, pixel boost on)
    python main.py --no-pixel               # heuristic only, text signals only
    python main.py --llm                    # LLM re-rank + diagnosis (text only; heuristic+pixel tables)
    python main.py --vlm                    # VLM + screenshot/visual pipeline
    python main.py --llm --vlm              # hybrid (LLM text + VLM visual ensemble)
    python main.py --source efe_irem --trace wikipedia --llm
    python main.py --no-eval                # skip evaluation
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
from src.llm_reasoner import LlmReasoner, slim_steps_for_llm
from src.vlm_reasoner import VlmReasoner, VlmAnalysisError
from src.screenshot_resolver import (
    ScreenshotResolver,
    filter_steps_with_screenshot_pairs,
    valid_screenshot_pair,
)
from src.trace_filter import parse_trace_ids, trace_selected
from src.report_generator import ReportGenerator
from src.evaluation import Evaluator
from src.ranking_arbitrator import (
    is_strong_observer_pixel_lock,
    should_apply_deterministic_promote,
)
from src.visual_signals import (
    annotate_visual_causal_scores,
    summarize_screenshot_analysis,
    vlm_inject_step_ids,
    best_pixel_signal,
)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _classify_vlm_error(err: str) -> str:
    low = err.lower()
    if "no screenshot" in low or "missing pass or fail" in low:
        return "vlm_no_screenshots"
    if "all-zero" in low:
        return "vlm_all_zero"
    if "402" in err or "429" in err or "rate limit" in low or "credits" in low:
        return "vlm_api_error"
    return "vlm_error"


def run_trace(
    source: str,
    trace_cfg: dict,
    raw_base: str,
    cfg: dict,
    detector: AnomalyDetector,
    ranker: Ranker,
    llm: LlmReasoner | None,
    vlm: VlmReasoner | None,
    resolver: ScreenshotResolver | None,
    reporter: ReportGenerator,
    evaluator: Evaluator,
    ground_truth: dict,
    use_llm: bool = False,
    use_vlm: bool = False,
    use_pixel: bool = True,
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

    # 3. Score (text signals — always)
    scored = detector.compute_scores(aligned)

    # Local visual pipeline (pixel scan + visual-causal) for --llm and --vlm (no API)
    source_cfg_base = cfg["data"]["sources"][source]["base"]
    visual_causal_cfg = cfg.get("ranking", {}).get("visual_causal", {})
    pixel_scores: dict[int, dict] = {}
    has_visual_causal = False
    ran_screenshot_scan = False
    screenshot_analysis: dict = {}
    scored_visual = scored
    use_visual_scan = (use_llm or use_vlm) and resolver and visual_causal_cfg.get("enabled", True)

    if use_visual_scan:
        pixel_scores = ranker.pixel_scores_for_trace(
            scored, resolver, source, source_cfg_base, trace_cfg
        )
        ran_screenshot_scan = bool(pixel_scores)
        scored = ranker.merge_pixel_scan(scored, pixel_scores)
        scored_visual = annotate_visual_causal_scores(scored, pixel_scores, visual_causal_cfg)
        screenshot_analysis = summarize_screenshot_analysis(
            scored_visual, pixel_scores, visual_causal_cfg
        )
        for step in scored_visual:
            vc = float(step.get("visual_causal_score") or 0)
            if vc > 0:
                has_visual_causal = True
                root = step["step_id"]
                div = step.get("visual_divergence_step")
                nxt = step.get("visual_causal_next_step")
                reason = step.get("visual_causal_reason", "")
                loc = step.get("pixel_localized", 0)
                print(
                    f"  Visual divergence: first persistent at step {div} "
                    f"(localized={loc:.2f}) → root step {root} "
                    f"score={vc:.2f}"
                    + (f", visible at step {nxt}" if nxt is not None else "")
                    + (f" [{reason}]" if reason else "")
                )
                break
        if ran_screenshot_scan and not has_visual_causal:
            print(
                f"  Screenshot scan: {screenshot_analysis['steps_scanned']} steps — "
                "no persistent pass/fail divergence above threshold"
            )
        elif not ran_screenshot_scan:
            print("  Screenshot scan: no pass/fail PNG pairs found for this trace")

    # 4. Rank (heuristic baseline — text-only table; visual layer when scan ran)
    heuristic_text_only = ranker.add_rank_metadata(
        ranker.rank_heuristic(scored, include_visual_causal=False)
    )
    heuristic_rank_input = scored_visual if ran_screenshot_scan else scored
    heuristic_top = ranker.rank_heuristic(
        heuristic_rank_input,
        include_visual_causal=(ran_screenshot_scan and has_visual_causal),
    )
    heuristic_visual = (
        ranker.add_rank_metadata([{**s} for s in heuristic_top])
        if ran_screenshot_scan and has_visual_causal else None
    )

    # Pixel-diff boost: heuristic-only mutates final ranking; --llm/--vlm use report tables only.
    used_pixel_boost = False
    heuristic_pixel_boosted = None
    run_pixel = use_pixel and resolver
    if run_pixel and cfg.get("ranking", {}).get("heuristic_pixel_fallback", True):
        top_with_shots = resolver.attach_screenshots(
            heuristic_top, source, source_cfg_base, trace_cfg
        )
        if any(
            valid_screenshot_pair(s.get("pass_screenshot"), s.get("fail_screenshot"))
            for s in top_with_shots
        ):
            px_weight = cfg.get("ranking", {}).get("heuristic_pixel_weight", 0.35)
            boosted = ranker.apply_pixel_boost(
                [{**s} for s in heuristic_top], top_with_shots, weight=px_weight
            )
            used_pixel_boost = True
            if use_llm or use_vlm:
                heuristic_pixel_boosted = boosted
            else:
                heuristic_top = boosted
            print(f"  Heuristic pixel-diff boost applied (weight={px_weight})")

    heuristic_top = ranker.add_rank_metadata(heuristic_top)
    final_ranked = heuristic_top  # default: heuristic

    # Pixel scores on scored steps for LLM pool / VLM final (Hit@k)
    if heuristic_pixel_boosted:
        scored = ranker.merge_pixel_boost(scored, heuristic_pixel_boosted)
        scored_visual = ranker.merge_pixel_boost(scored_visual, heuristic_pixel_boosted)

    if used_pixel_boost:
        ranking_mode = "heuristic+pixel"
        if ran_screenshot_scan and has_visual_causal:
            ranking_mode += "+visual_causal"
    elif ran_screenshot_scan and has_visual_causal:
        ranking_mode = "heuristic+visual_causal"
    else:
        ranking_mode = "heuristic"
    llm_ranked_steps = None
    ranking_decisions: list[str] = []
    llm_scored = scored_visual if ran_screenshot_scan else scored

    # 5. LLM re-ranking + diagnosis (--llm; text + local visual signals when scan ran)
    llm_output = {}
    if use_llm and llm:
        try:
            print(f"  Running LLM re-ranking...")
            heuristic_order = [
                s["step_id"]
                for s in sorted(llm_scored, key=lambda x: x["combined_score"], reverse=True)
            ]
            if heuristic_pixel_boosted:
                px1 = heuristic_pixel_boosted[0]
                if float(px1.get("pixel_score") or 0) >= 0.65:
                    pid = px1["step_id"]
                    heuristic_order = [pid] + [x for x in heuristic_order if x != pid]
            elif ran_screenshot_scan:
                px_leader = max(llm_scored, key=best_pixel_signal, default=None)
                if px_leader and best_pixel_signal(px_leader) >= 0.65:
                    pid = px_leader["step_id"]
                    heuristic_order = [pid] + [x for x in heuristic_order if x != pid]
            pre_visual_ids = (
                [s["step_id"] for s in heuristic_visual] if heuristic_visual else None
            )
            rerank_pool = ranker.candidates_for_llm(llm_scored)
            reranked_ids = llm.rerank(
                rerank_pool, heuristic_order=heuristic_order, session_id=trace_id
            )
            slim_for_guard = slim_steps_for_llm(
                llm_scored, compact=True, heuristic_order=heuristic_order
            )
            llm_ranked, anchor_notes = ranker.apply_llm_reranking(
                reranked_ids,
                llm_scored,
                heuristic_top_ids=heuristic_order,
                slim_steps=slim_for_guard,
            )
            ranking_decisions.extend(anchor_notes)
            llm_ranked, hit_notes, hit1_lock = ranker.guard_llm_hit_at_k(
                llm_ranked,
                llm_scored,
                heuristic_pixel_boosted,
                pre_visual_ids or heuristic_order,
                ranker.top_k,
                text_only_ranked=heuristic_text_only,
            )
            ranking_decisions.extend(hit_notes)
            llm_ranked = ranker.add_rank_metadata(llm_ranked)

            det_root, det_reason = ranker.find_deterministic_root(
                llm_scored,
                pixel_scores=pixel_scores or None,
                visual_causal_cfg=visual_causal_cfg,
                include_visual_causal=False,
            )
            vc_roots = [
                s for s in llm_scored if float(s.get("visual_causal_score") or 0) > 0
            ]
            if vc_roots and det_reason != "visual_causal":
                vc = min(vc_roots, key=lambda s: s["step_id"])
                ranking_decisions.append(
                    f"Visual-causal root step {vc['step_id']} not hard-promoted on LLM path "
                    f"(score={float(vc.get('visual_causal_score') or 0):.2f}) — "
                    "handled via LLM rerank + post-LLM guards instead."
                )

            print(f"  Running LLM diagnosis...")
            diag_steps = ranker.diagnosis_candidates(llm_ranked, llm_scored)
            llm_output = llm.run(diag_steps, reranked_ids, session_id=trace_id)

            if det_root is not None:
                allow_det, det_note = should_apply_deterministic_promote(
                    det_root,
                    det_reason,
                    llm_ranked,
                    hit1_lock,
                    llm_scored,
                )
                if allow_det:
                    final_ranked = ranker.promote_step(
                        llm_ranked, llm_scored, det_root
                    )
                    ranking_mode = "llm+deterministic"
                    ranking_decisions.append(
                        f"Deterministic promote: step {det_root} ({det_reason})."
                    )
                else:
                    final_ranked = llm_ranked
                    ranking_mode = "llm"
                    ranking_decisions.append(
                        f"Deterministic promote: step {det_root} ({det_reason}) "
                        f"skipped — {det_note}."
                    )
            else:
                # If LLM diagnosis identifies a root cause step, promote it to #1
                # unless a strong visual Hit@k guard already locked a different step.
                rc_step = llm_output.get("diagnosis", {}).get("root_cause_step_id")
                if rc_step is not None:
                    try:
                        rc_id = int(rc_step)
                        step_map = {s["step_id"]: s for s in llm_scored}
                        top1_id = llm_ranked[0]["step_id"] if llm_ranked else None
                        top1_step = step_map.get(top1_id) if top1_id is not None else None
                        visual_lock_blocks = hit1_lock in (
                            "visible_symptom",
                            "pixel_leader",
                            "text_action_anchor",
                            "causal_over_observer",
                        ) or is_strong_observer_pixel_lock(hit1_lock, top1_step)
                        if (
                            rc_id in step_map
                            and visual_lock_blocks
                            and top1_id is not None
                            and rc_id != top1_id
                        ):
                            final_ranked = llm_ranked
                            ranking_mode = "llm"
                            ranking_decisions.append(
                                f"Diagnosis root step {rc_id} not promoted — "
                                f"Hit@k guard locked step {top1_id} ({hit1_lock})."
                            )
                        elif rc_id in step_map:
                            promoted = [step_map[rc_id]]
                            rest = [s for s in llm_ranked if s["step_id"] != rc_id]
                            final_ranked = ranker.add_rank_metadata(
                                promoted + rest[: ranker.top_k - 1]
                            )
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

            llm_ranked_steps = final_ranked

        except Exception as e:
            print(f"  LLM failed ({e}), falling back to heuristic ranking")
            det_root, det_reason = ranker.find_deterministic_root(
                llm_scored,
                pixel_scores=pixel_scores or None,
                visual_causal_cfg=visual_causal_cfg,
            )
            if det_root is not None:
                final_ranked = ranker.promote_step(heuristic_top, llm_scored, det_root)
                ranking_mode = "heuristic+deterministic"
                ranking_decisions.append(
                    f"Deterministic promote: step {det_root} ({det_reason or 'unknown'})."
                )
            else:
                final_ranked = heuristic_top
                ranking_mode = "heuristic (llm failed)"

    # 5b. VLM visual analysis — on failure, fall back and still emit report
    vlm_output = {}
    ran_vlm = False
    vlm_status = "not_run"
    vlm_error: str | None = None
    vlm_error_code: str | None = None
    exclude_from_aggregate = False
    ranking_fallback: str | None = None
    if use_vlm and vlm and resolver:
        vlm_k = cfg.get("vlm", {}).get("top_k_for_vlm", 5)
        vlm_weight = cfg.get("vlm", {}).get("ensemble_vlm_weight", 0.4)

        inject_vlm = (
            vlm_inject_step_ids(scored_visual, pixel_scores, visual_causal_cfg)
            if pixel_scores else []
        )
        inject_vlm = ranker.merge_vlm_inject_ids(
            inject_vlm, scored_visual, heuristic_text_only
        )

        top_for_vlm = ranker.vlm_candidate_steps(
            final_ranked, scored_visual, vlm_k, inject_step_ids=inject_vlm or None
        )
        top_with_shots = resolver.attach_screenshots(
            top_for_vlm, source, source_cfg_base, trace_cfg
        )
        vlm_steps = filter_steps_with_screenshot_pairs(top_with_shots)
        skipped = len(top_with_shots) - len(vlm_steps)

        if not vlm_steps:
            vlm_status = "failed"
            vlm_error = "no screenshot pairs found for VLM candidates"
            vlm_error_code = "vlm_no_screenshots"
            print(f"  VLM failed: {vlm_error} ({trace_id})")
        else:
            if skipped:
                print(
                    f"  VLM: skipping {skipped} candidate step(s) without both pass/fail PNGs"
                )

            print(f"  Running VLM visual analysis ({len(vlm_steps)} screenshot pairs)...")
            try:
                vlm_output = vlm.analyze_steps(vlm_steps)
                if has_visual_causal:
                    vlm_output = ranker.backfill_visual_causal_vlm_scores(
                        vlm_output, scored_visual
                    )

                final_ranked_vlm, vlm_mode, vlm_notes = vlm.ensemble_rankings(
                    final_ranked, vlm_output, vlm_weight=vlm_weight,
                    vlm_only=(not use_llm),
                    scored_steps=scored_visual,
                    top_k=ranker.top_k,
                    pixel_boost_ranked=heuristic_pixel_boosted,
                    text_only_ranked=heuristic_text_only,
                )
                ranking_decisions.extend(vlm_notes)
                final_ranked = ranker.add_rank_metadata(final_ranked_vlm)
                ranking_mode = vlm_mode
                ran_vlm = True
                vlm_status = "ok"
                print(f"  VLM visual root cause: step {vlm_output.get('visual_root_cause_step_id')}")
                print(f"  VLM summary: {vlm_output.get('visual_summary', '')[:100]}")
            except VlmAnalysisError as e:
                err = str(e)
                vlm_status = "failed"
                vlm_error = err
                vlm_error_code = _classify_vlm_error(err)
                print(f"  VLM failed: {e}")
                if "402" in err or "Insufficient credits" in err:
                    print("  Hint: add OpenRouter credits, or switch vlm_model to a :free model in config.yaml")
                elif "429" in err or "Rate limit" in err:
                    print("  Hint: free-model daily quota exhausted — add credits at openrouter.ai/settings/credits")
                    print("        or wait for quota reset, then retry")

        if vlm_status == "failed":
            ranking_fallback = ranking_mode
            if use_llm:
                ranking_mode = f"{ranking_mode}+vlm_failed"
            else:
                exclude_from_aggregate = True
                ranking_mode = f"{ranking_mode}+vlm_failed"
            ranking_decisions.append(
                f"VLM failed ({vlm_error_code}): {vlm_error}. "
                f"Final ranking uses {ranking_fallback} fallback."
            )
            if exclude_from_aggregate:
                ranking_decisions.append(
                    "Excluded from VLM-only aggregate metrics (VLM did not produce scores)."
                )
            else:
                ranking_decisions.append(
                    "Included in aggregate using LLM/heuristic fallback ranking."
                )
            print(f"  Report will use {ranking_fallback} fallback ranking")

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
    def _slim(steps: list[dict]) -> list[dict]:
        return [{k: v for k, v in s.items() if k not in ("pass_step", "fail_step")} for s in steps]

    slim_ranked = _slim(final_ranked)
    slim_heuristic = _slim(
        heuristic_text_only
        if (used_pixel_boost or (ran_screenshot_scan and has_visual_causal))
        else heuristic_top
    )
    slim_heuristic_pixel = _slim(
        ranker.add_rank_metadata(heuristic_pixel_boosted)
        if heuristic_pixel_boosted is not None
        else (heuristic_top if used_pixel_boost else None)
    ) if used_pixel_boost else None
    slim_heuristic_visual = _slim(heuristic_visual) if heuristic_visual else None
    slim_llm = _slim(llm_ranked_steps) if llm_ranked_steps else None

    report = reporter.build(
        trace_id=trace_id,
        ranked_steps=slim_ranked,
        heuristic_steps=slim_heuristic,
        heuristic_pixel_steps=slim_heuristic_pixel,
        heuristic_visual_steps=slim_heuristic_visual,
        used_pixel_boost=used_pixel_boost,
        has_visual_causal=(ran_screenshot_scan and has_visual_causal),
        ran_screenshot_scan=ran_screenshot_scan,
        screenshot_analysis=screenshot_analysis if ran_screenshot_scan else None,
        llm_ranked_steps=slim_llm,
        ran_vlm=ran_vlm,
        vlm_only=(ran_vlm and not use_llm),
        ranking_mode=ranking_mode,
        llm_output=llm_output,
        vlm_output=vlm_output if vlm_output else None,
        eval_result=eval_result or None,
        metadata={
            "source": source,
            "fault_type": ground_truth.get(source, {}).get(trace_cfg["id"], {}).get("fault_type"),
            "ranking_decisions": ranking_decisions,
            "vlm_status": vlm_status,
            "vlm_error": vlm_error,
            "vlm_error_code": vlm_error_code,
            "ranking_fallback": ranking_fallback,
            "exclude_from_aggregate": exclude_from_aggregate,
            "ran_screenshot_scan": ran_screenshot_scan,
            "screenshots_available_count": sum(
                1 for s in llm_scored if s.get("screenshots_available")
            ),
        },
    )
    reporter.print_report(report)

    out_id = trace_cfg["id"]
    reporter.save_json(report, f"{cfg['outputs']['reports']}/{source}/{out_id}.json")
    reporter.save_text(report, f"{cfg['outputs']['reports']}/{source}/{out_id}.txt")
    ranker.save(slim_ranked, f"{cfg['outputs']['rankings']}/{source}/{out_id}.json")

    return {
        "trace_id": trace_id,
        "eval": eval_result,
        "llm_output": llm_output,
        "vlm_output": vlm_output,
        "vlm_status": vlm_status,
        "vlm_error": vlm_error,
        "vlm_error_code": vlm_error_code,
        "exclude_from_aggregate": exclude_from_aggregate,
        "ranking_fallback": ranking_fallback,
    }


def main():
    parser = argparse.ArgumentParser(description="TraceLens pipeline")
    parser.add_argument("--source", default=None, help="Filter by source (efe_irem/areeb_salem/ersel)")
    parser.add_argument(
        "--trace",
        default=None,
        help="Run only these trace id(s), comma-separated (e.g. amazon,imdb,npm)",
    )
    parser.add_argument("--skip",   default=None, help="Comma-separated trace IDs to skip (e.g. saucedemo_1,gutenberg)")
    parser.add_argument("--from",   default=None, dest="from_trace",
                        help="Start from this trace ID (skip everything before it in config order)")
    parser.add_argument("--llm",     action="store_true", help="Enable LLM re-ranking and diagnosis")
    parser.add_argument("--vlm",     action="store_true", help="Enable VLM visual analysis")
    parser.add_argument("--no-pixel", action="store_true",
                        help="Disable pixel-diff boost in heuristic-only mode (text signals only)")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    skip_ids = set(args.skip.split(",")) if args.skip else set()
    trace_filter = parse_trace_ids(args.trace)

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
            enable_prompt_cache=cfg["model"].get("prompt_cache", True),
        )
        cache_note = "prompt cache on" if llm.enable_prompt_cache else "prompt cache off"
        print(f"LLM enabled: {llm.model} ({cache_note})")

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
            temperature=vlm_cfg.get("temperature", 0.0),
            base_url=vlm_cfg.get("base_url", "https://openrouter.ai/api/v1"),
            per_step=vlm_cfg.get("per_step", True),
        )
        print(f"VLM enabled: {vlm.model}")

    if args.llm and args.vlm:
        w = cfg.get("vlm", {}).get("ensemble_vlm_weight", 0.4)
        print(f"Mode: llm+vlm hybrid ({int((1 - w) * 100)}% LLM / {int(w * 100)}% VLM)")
    elif args.llm:
        print("Mode: llm")
    elif args.vlm:
        print("Mode: vlm")
    else:
        label = "heuristic (text only)" if args.no_pixel else "heuristic + pixel boost"
        print(f"Mode: {label}")

    all_results = []
    trace_count = 0
    reached_from = (args.from_trace is None)  # if no --from, start immediately

    for source, source_cfg in cfg["data"]["sources"].items():
        if args.source and source != args.source:
            continue
        for trace_cfg in source_cfg["traces"]:
            tid = trace_cfg["id"]
            # --from: skip everything until we reach the named trace
            if not reached_from:
                if tid == args.from_trace:
                    reached_from = True
                else:
                    print(f"  Skipping {source}/{tid} (before --from {args.from_trace})")
                    continue
            if not trace_selected(tid, trace_filter):
                continue
            if tid in skip_ids:
                print(f"  Skipping {source}/{tid} (--skip)")
                continue
            # Inter-trace delay when using LLM to avoid rate limits
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
                vlm=vlm,
                resolver=resolver,
                reporter=reporter,
                evaluator=evaluator,
                ground_truth=ground_truth,
                use_llm=args.llm,
                use_vlm=args.vlm,
                use_pixel=(not args.no_pixel),
                use_eval=(not args.no_eval),
            )
            all_results.append(result)
            if result.get("vlm_status") == "failed":
                note = "excluded from aggregate" if result.get("exclude_from_aggregate") else "LLM fallback used"
                print(f"  VLM failed for {result['trace_id']} — report saved ({note})")

    # Aggregate evaluation + run report
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.llm and args.vlm:
        run_mode   = f"llm+vlm ({cfg['model']['llm_model']} + {cfg.get('vlm', {}).get('vlm_model', 'vlm')})"
        run_prefix = "llm+vlm"
    elif args.vlm:
        run_mode   = f"vlm ({cfg.get('vlm', {}).get('vlm_model', 'unknown')})"
        run_prefix = "vlm"
    elif args.llm:
        run_mode   = f"llm ({cfg['model']['llm_model']})"
        run_prefix = "llm"
    else:
        run_mode   = "heuristic"
        run_prefix = "heuristic"

    if not args.no_eval:
        excluded_traces = [
            {
                "trace_id": r["trace_id"],
                "reason": r.get("vlm_error_code"),
                "error": r.get("vlm_error"),
            }
            for r in all_results if r.get("exclude_from_aggregate")
        ]
        eval_results = [
            r["eval"] for r in all_results
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

            # ── per-run timestamped report ─────────────────────────────────
            per_trace_rows = [
                {
                    "trace_id": r["trace_id"],
                    **r["eval"],
                    "vlm_status": r.get("vlm_status"),
                    "exclude_from_aggregate": r.get("exclude_from_aggregate", False),
                }
                for r in all_results if r.get("eval")
            ]

            run_report = {
                "run_id": run_ts,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "mode": run_mode,
                "filter_source": args.source,
                "filter_trace": args.trace,
                "n_traces_run": trace_count,
                "n_traces_evaluated": len(eval_results),
                "n_traces_excluded": len(excluded_traces),
                "excluded_traces": excluded_traces,
                "aggregate": agg,
                "per_trace": per_trace_rows,
            }

            runs_dir = Path(cfg["outputs"]["metrics"]) / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            run_path = runs_dir / f"{run_prefix}_run_{run_ts}.json"
            with open(run_path, "w", encoding="utf-8") as f:
                json.dump(run_report, f, indent=2)
            print(f"\n  Run report saved → {run_path}")

            # also overwrite the convenience latest files
            evaluator.save(agg, f"{cfg['outputs']['metrics']}/aggregate.json")
            evaluator.save(per_trace_rows,
                           f"{cfg['outputs']['metrics']}/per_trace.json")

            _print_overall_accuracy(agg, excluded_count=len(excluded_traces))


def _print_overall_accuracy(agg: dict, excluded_count: int = 0) -> None:
    n   = agg.get("n_traces", "?")
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


if __name__ == "__main__":
    main()
