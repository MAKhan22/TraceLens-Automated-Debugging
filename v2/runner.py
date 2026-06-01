"""
v2/runner.py
--------------
TraceLens v2 pipeline: extract features → fuse → rank once → diagnose (read-only).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.anomaly_detector import AnomalyDetector
from src.evaluation import Evaluator
from src.llm_reasoner import LlmReasoner, slim_steps_for_llm
from src.ranker import Ranker
from src.report_generator import ReportGenerator
from src.screenshot_resolver import (
    ScreenshotResolver,
    filter_steps_with_screenshot_pairs,
    valid_screenshot_pair,
)
from src.trace_aligner import align
from src.trace_parser import parse_trace, save_processed
from src.visual_signals import (
    annotate_visual_causal_scores,
    best_pixel_signal,
    summarize_screenshot_analysis,
    vlm_inject_step_ids,
)
from src.vlm_reasoner import VlmAnalysisError, VlmReasoner

from v2.fusion import (
    FusionWeights,
    llm_prior_from_order,
    rank_by_fusion,
    vlm_prior_from_output,
)


def _classify_vlm_error(err: str) -> str:
    low = err.lower()
    if "no screenshot" in low or "missing pass or fail" in low:
        return "vlm_no_screenshots"
    if "all-zero" in low:
        return "vlm_all_zero"
    if "402" in err or "429" in err or "rate limit" in low or "credits" in low:
        return "vlm_api_error"
    return "vlm_error"


def _ranking_pool(
    ranker: Ranker,
    scored_steps: list[dict],
    heuristic_text_only: list[dict],
) -> list[dict]:
    """Candidate steps considered by fusion (same inject philosophy as v1)."""
    by_id = {s["step_id"]: s for s in scored_steps}
    ids: list[int] = []
    seen: set[int] = set()

    def add(sid: int | None) -> None:
        if sid is not None and sid not in seen and sid in by_id:
            ids.append(sid)
            seen.add(sid)

    for s in sorted(scored_steps, key=lambda x: x["combined_score"], reverse=True):
        add(s["step_id"])
        if len(ids) >= ranker.pre_llm_k:
            break
    for row in heuristic_text_only[:3]:
        add(row["step_id"])
    from src.causal_signals import find_causal_root_step

    add(find_causal_root_step(scored_steps))
    for s in scored_steps:
        if float(s.get("action_score") or 0) > 0:
            add(s["step_id"])
        if float(s.get("console_score") or 0) >= 0.5:
            add(s["step_id"])
        if float(s.get("visual_causal_score") or 0) > 0:
            add(s["step_id"])
        if best_pixel_signal(s) >= 0.65:
            add(s["step_id"])
    return [by_id[sid] for sid in ids]


def run_trace_v2(
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
    *,
    use_llm: bool = False,
    use_vlm: bool = False,
    use_pixel: bool = True,
    use_eval: bool = True,
    fusion_weights: FusionWeights | None = None,
) -> dict:
    trace_id = f"{source}/{trace_cfg['id']}"
    print(f"\n{'='*50}")
    print(f"Processing (v2): {trace_id}")

    v2_cfg = cfg.get("v2", {})
    strong_px = float(v2_cfg.get("strong_observer_pixel", 0.95))
    weights = fusion_weights or FusionWeights.from_config(cfg)

    source_base = Path(raw_base) / cfg["data"]["sources"][source]["base"]
    pass_path = str(source_base / trace_cfg["pass"])
    fail_path = str(source_base / trace_cfg["fail"])

    passing = parse_trace(source, pass_path)
    failing = parse_trace(source, fail_path)
    print(f"  Parsed: {len(passing)} pass steps, {len(failing)} fail steps")

    save_processed(passing, f"data/processed/{source}/{trace_cfg['id']}_pass.json")
    save_processed(failing, f"data/processed/{source}/{trace_cfg['id']}_fail.json")

    aligned = align(passing, failing)
    print(f"  Aligned: {len(aligned)} pairs")

    scored = detector.compute_scores(aligned)
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
                print(
                    f"  Visual divergence: first persistent at step {div} "
                    f"→ root step {root} score={vc:.2f}"
                )
                break
        if ran_screenshot_scan and not has_visual_causal:
            print(
                f"  Screenshot scan: {screenshot_analysis['steps_scanned']} steps — "
                "no persistent pass/fail divergence above threshold"
            )
        elif not ran_screenshot_scan:
            print("  Screenshot scan: no pass/fail PNG pairs found for this trace")

    llm_scored = scored_visual if ran_screenshot_scan else scored

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
        if ran_screenshot_scan and has_visual_causal
        else None
    )

    used_pixel_boost = False
    heuristic_pixel_boosted = None
    if use_pixel and resolver and cfg.get("ranking", {}).get("heuristic_pixel_fallback", True):
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
            heuristic_pixel_boosted = boosted
            scored = ranker.merge_pixel_boost(scored, boosted)
            llm_scored = ranker.merge_pixel_boost(
                scored_visual if ran_screenshot_scan else scored, boosted
            )
            print(f"  Heuristic pixel-diff boost applied (weight={px_weight})")

    heuristic_top = ranker.add_rank_metadata(heuristic_top)

    ranking_decisions: list[str] = []
    llm_output: dict = {}
    llm_prior: dict[int, float] = {}
    reranked_ids: list[int] = []

    if use_llm and llm:
        try:
            print("  Running LLM re-ranking (soft prior only)...")
            heuristic_order = [
                s["step_id"]
                for s in sorted(llm_scored, key=lambda x: x["combined_score"], reverse=True)
            ]
            pool = ranker.candidates_for_llm(llm_scored)
            reranked_ids = llm.rerank(pool, heuristic_order=heuristic_order)
            llm_prior = llm_prior_from_order(reranked_ids)
            ranking_decisions.append(
                "LLM rerank mapped to fusion llm_prior channel (no anchor/Hit@k guards)."
            )
        except Exception as e:
            print(f"  LLM rerank failed ({e}); fusion without llm_prior")
            ranking_decisions.append(f"LLM rerank failed: {e}")

    vlm_output: dict = {}
    vlm_prior: dict[int, float] = {}
    ran_vlm = False
    vlm_status = "not_run"
    vlm_error: str | None = None
    vlm_error_code: str | None = None
    exclude_from_aggregate = False
    ranking_fallback: str | None = None

    if use_vlm and vlm and resolver:
        vlm_k = cfg.get("vlm", {}).get("top_k_for_vlm", 5)
        inject_vlm = (
            vlm_inject_step_ids(scored_visual, pixel_scores, visual_causal_cfg)
            if pixel_scores
            else []
        )
        inject_vlm = ranker.merge_vlm_inject_ids(
            inject_vlm, scored_visual, heuristic_text_only
        )
        prelim = heuristic_top
        top_for_vlm = ranker.vlm_candidate_steps(
            prelim, llm_scored, vlm_k, inject_step_ids=inject_vlm or None
        )
        top_with_shots = resolver.attach_screenshots(
            top_for_vlm, source, source_cfg_base, trace_cfg
        )
        vlm_steps = filter_steps_with_screenshot_pairs(top_with_shots)
        if not vlm_steps:
            vlm_status = "failed"
            vlm_error = "no screenshot pairs found for VLM candidates"
            vlm_error_code = "vlm_no_screenshots"
            print(f"  VLM failed: {vlm_error}")
        else:
            print(f"  Running VLM visual analysis ({len(vlm_steps)} screenshot pairs)...")
            try:
                vlm_output = vlm.analyze_steps(vlm_steps)
                if has_visual_causal:
                    vlm_output = ranker.backfill_visual_causal_vlm_scores(
                        vlm_output, scored_visual
                    )
                vlm_prior = vlm_prior_from_output(vlm_output)
                ran_vlm = True
                vlm_status = "ok"
                ranking_decisions.append(
                    "VLM scores mapped to fusion vlm_prior channel (no post-ensemble guards)."
                )
                print(
                    f"  VLM visual root cause: step "
                    f"{vlm_output.get('visual_root_cause_step_id')}"
                )
            except VlmAnalysisError as e:
                vlm_status = "failed"
                vlm_error = str(e)
                vlm_error_code = _classify_vlm_error(vlm_error)
                print(f"  VLM failed: {e}")

        if vlm_status == "failed":
            if use_llm:
                ranking_decisions.append(
                    f"VLM failed ({vlm_error_code}): fusion without vlm_prior."
                )
            else:
                exclude_from_aggregate = True
                ranking_decisions.append(
                    f"VLM failed ({vlm_error_code}): excluded from VLM-only aggregate."
                )

    pool_steps = _ranking_pool(ranker, llm_scored, heuristic_text_only)
    has_visual = ran_screenshot_scan or bool(vlm_prior) or used_pixel_boost

    final_ranked, fusion_notes, _channels = rank_by_fusion(
        pool_steps,
        weights=weights,
        use_llm=bool(use_llm and llm_prior),
        use_vlm=bool(use_vlm and vlm_prior),
        has_visual=has_visual,
        llm_prior=llm_prior,
        vlm_prior=vlm_prior,
        strong_observer_pixel=strong_px,
        top_k=ranker.top_k,
    )
    ranking_decisions.extend(fusion_notes)
    final_ranked = ranker.add_rank_metadata(final_ranked)

    if use_llm and use_vlm and llm_prior and vlm_prior:
        ranking_mode = "v2:llm+vlm"
    elif use_llm and llm_prior:
        ranking_mode = "v2:llm"
    elif use_vlm and vlm_prior:
        ranking_mode = "v2:vlm"
    elif use_vlm and vlm_status == "failed":
        ranking_mode = "v2:heuristic+vlm_failed"
    else:
        ranking_mode = "v2:heuristic"

    llm_ranked_steps = None
    if use_llm and llm and llm_prior:
        by_id = {s["step_id"]: s for s in pool_steps}
        llm_ranked_steps = ranker.add_rank_metadata(
            [by_id[sid] for sid in reranked_ids[: ranker.top_k] if sid in by_id]
        )

    if use_llm and llm and llm_prior:
        try:
            print("  Running LLM diagnosis (read-only, does not change rank)...")
            diag_steps = final_ranked[: ranker.top_k]
            llm_output = llm.run(diag_steps, reranked_ids or [s["step_id"] for s in diag_steps])
            ranking_decisions.append(
                "LLM diagnosis is explanatory only — ranking frozen after fusion."
            )
        except Exception as e:
            print(f"  LLM diagnosis failed ({e})")
            ranking_decisions.append(f"LLM diagnosis failed: {e}")

    eval_result = {}
    if use_eval:
        gt = ground_truth.get(source, {}).get(trace_cfg["id"], {})
        if gt:
            actual_fault = gt["fault_step"]
            eval_result = evaluator.evaluate_trace(final_ranked, actual_fault)
            print(
                f"  Hit@1={eval_result['hit@1']}  Hit@5={eval_result['hit@5']}  "
                f"RankDist={eval_result['rank_distance']}  "
                f"Top1StepDist={eval_result['top1_step_distance']}"
            )
        else:
            print(f"  No ground truth for {trace_id}, skipping evaluation")

    def _slim(steps: list[dict]) -> list[dict]:
        return [
            {k: v for k, v in s.items() if k not in ("pass_step", "fail_step")}
            for s in steps
        ]

    slim_ranked = _slim(final_ranked)
    slim_heuristic = _slim(
        heuristic_text_only
        if (used_pixel_boost or (ran_screenshot_scan and has_visual_causal))
        else heuristic_top
    )
    slim_heuristic_pixel = (
        _slim(ranker.add_rank_metadata(heuristic_pixel_boosted))
        if heuristic_pixel_boosted
        else None
    )
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
            "pipeline_version": "v2",
            "source": source,
            "fault_type": ground_truth.get(source, {}).get(trace_cfg["id"], {}).get("fault_type"),
            "ranking_decisions": ranking_decisions,
            "fusion_weights": weights.__dict__,
            "vlm_status": vlm_status,
            "vlm_error": vlm_error,
            "vlm_error_code": vlm_error_code,
            "ranking_fallback": ranking_fallback,
            "exclude_from_aggregate": exclude_from_aggregate,
            "ran_screenshot_scan": ran_screenshot_scan,
        },
    )
    reporter.print_report(report)

    out_rankings = cfg["outputs"]["rankings"]
    out_reports = cfg["outputs"]["reports"]
    reporter.save_json(report, f"{out_reports}/{source}/{trace_cfg['id']}.json")
    reporter.save_text(report, f"{out_reports}/{source}/{trace_cfg['id']}.txt")
    ranker.save(slim_ranked, f"{out_rankings}/{source}/{trace_cfg['id']}.json")

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
