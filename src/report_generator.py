"""
report_generator.py
-------------------
Assembles and saves the final diagnosis report.
"""

import json
from datetime import datetime
from pathlib import Path


class ReportGenerator:
    def build(
        self,
        trace_id: str,
        ranked_steps: list[dict],
        heuristic_steps: list[dict] | None = None,
        heuristic_pixel_steps: list[dict] | None = None,
        heuristic_visual_steps: list[dict] | None = None,
        used_pixel_boost: bool = False,
        has_visual_causal: bool = False,
        ran_screenshot_scan: bool = False,
        screenshot_analysis: dict | None = None,
        llm_ranked_steps: list[dict] | None = None,
        ran_vlm: bool = False,
        vlm_only: bool = False,
        ranking_mode: str = "heuristic",
        llm_output: dict | None = None,
        vlm_output: dict | None = None,
        eval_result: dict | None = None,
        metadata: dict | None = None,
    ) -> dict:
        llm_output = llm_output or {}
        vlm_output = vlm_output or {}
        diagnosis = llm_output.get("diagnosis", {})

        report = {
            "trace_id":    trace_id,
            "generated":   datetime.utcnow().isoformat() + "Z",
            "metadata":    metadata or {},
            "ranking_mode": ranking_mode,

            "ranked_suspicious_steps": ranked_steps,
            "heuristic_steps":         heuristic_steps or ranked_steps,
            "heuristic_pixel_steps":   heuristic_pixel_steps,
            "heuristic_visual_steps":  heuristic_visual_steps,
            "llm_ranked_steps":        llm_ranked_steps,
            "screenshot_analysis":     screenshot_analysis,

            "technical_diagnosis": {
                "root_cause_step_id": diagnosis.get("root_cause_step_id"),
                "root_cause_summary": diagnosis.get("root_cause_summary"),
                "downstream_steps":   diagnosis.get("downstream_steps", []),
                "failure_chain":      diagnosis.get("failure_chain"),
            },

            "stakeholder_summary": llm_output.get("stakeholder_summary", ""),

            "visual_analysis": {
                "visual_root_cause_step_id": vlm_output.get("visual_root_cause_step_id"),
                "visual_summary":            vlm_output.get("visual_summary"),
                "visual_scores":             vlm_output.get("visual_scores", []),
                "steps_with_screenshots":    vlm_output.get("steps_with_screenshots", 0),
            } if vlm_output else None,

            "evaluation": eval_result or {},

            "_text_report": self._format_text(
                trace_id, ranked_steps, heuristic_steps or ranked_steps,
                heuristic_pixel_steps, heuristic_visual_steps,
                used_pixel_boost, has_visual_causal, ran_screenshot_scan,
                screenshot_analysis,
                llm_ranked_steps, ran_vlm, vlm_only,
                ranking_mode, diagnosis,
                llm_output.get("stakeholder_summary", ""),
                eval_result,
                vlm_output,
                metadata,
            ),
        }
        return report

    # ── formatting ────────────────────────────────────────────────────────────

    @staticmethod
    def _layers_label(base: str, layers: list[str]) -> str:
        """Build table title listing every signal layer that affects that ranking."""
        if not layers:
            return base
        return f"{base} ({' + '.join(layers)})"

    def _step_flags(self, s: dict) -> str:
        flags = []
        if s.get("network_score", 0) >= 0.5:
            flags.append("network anomaly")
        if s.get("console_score", 0) >= 0.5:
            flags.append("console error")
        if s.get("intent_score", 0) >= 0.9:
            flags.append("verification failed")
        if s.get("action_score", 0) >= 0.5:
            flags.append("action divergence")
        vc = float(s.get("visual_causal_score") or 0)
        if vc > 0:
            nxt = s.get("visual_causal_next_step")
            reason = s.get("visual_causal_reason") or "visual causal"
            if nxt is not None:
                flags.append(f"visual causal ({reason}, visible@{nxt}, score={vc:.2f})")
            else:
                flags.append(f"visual causal ({reason}, score={vc:.2f})")
        if s.get("screenshots_available") is False:
            flags.append("no screenshots")
        return f"  [{', '.join(flags)}]" if flags else ""

    def _format_ranked_list(self, steps: list[dict], label: str,
                             actual_fault: int | None = None,
                             show_pixel: bool = False,
                             show_visual_causal: bool = False,
                             show_vlm_score: bool = False) -> list[str]:
        lines = [label, "-" * 40]
        for s in steps:
            rank  = s.get("rank", "?")
            sid   = s.get("step_id", "?")
            act   = s.get("action", "")[:70]
            score = s.get("rank_score", s.get("combined_score", 0))
            flags = self._step_flags(s)
            marker = " ← ACTUAL FAULT" if (actual_fault is not None and sid == actual_fault) else ""
            px = s.get("pixel_score")
            vc = float(s.get("visual_causal_score") or 0)
            vlm = s.get("vlm_visual_score")
            parts = []
            if show_vlm_score and vlm is not None:
                parts.append(f"vlm={vlm:.3f}")
            else:
                parts.append(f"score={score:.3f}")
            if show_visual_causal and vc > 0:
                parts.append(f"visual_causal={vc:.3f}")
            if show_pixel and px is not None:
                parts.append(f"pixel={px:.3f}")
            score_str = ", ".join(parts)
            lines.append(f"  #{rank}  Step {sid}: {act}{flags}  ({score_str}){marker}")
        return lines

    def _format_fusion_list(
        self,
        steps: list[dict],
        label: str,
        actual_fault: int | None = None,
    ) -> list[str]:
        """V2 final ranking table with fusion_score and active channels."""
        lines = [label, "-" * 40]
        for s in steps:
            rank = s.get("rank", "?")
            sid = s.get("step_id", "?")
            act = s.get("action", "")[:70]
            flags = self._step_flags(s)
            marker = (
                " ← ACTUAL FAULT"
                if (actual_fault is not None and sid == actual_fault)
                else ""
            )
            fs = float(s.get("fusion_score") or s.get("rank_score") or 0)
            ch = s.get("fusion_channels") or {}
            active = [
                f"{k}={float(v):.2f}"
                for k, v in sorted(ch.items())
                if v is not None and float(v) > 0
            ]
            ch_str = ", ".join(active) if active else "n/a"
            lines.append(
                f"  #{rank}  Step {sid}: {act}{flags}  "
                f"(fusion={fs:.3f}; {ch_str}){marker}"
            )
        return lines

    def _format_screenshot_analysis(self, analysis: dict | None) -> list[str]:
        if not analysis:
            return []
        lines = [
            "SCREENSHOT ANALYSIS (pass/fail after images)",
            "-" * 40,
            f"Steps scanned:              {analysis.get('steps_scanned', 0)}",
        ]
        if analysis.get("has_signal"):
            lines += [
                f"First persistent divergence: step {analysis.get('first_persistent_divergence')} "
                f"(score={analysis.get('divergence_score', 0):.3f})",
                f"Attributed root step:        step {analysis.get('attributed_root_step')} "
                f"(visual_causal={analysis.get('visual_causal_score', 0):.3f})",
            ]
            if analysis.get("visible_at_step") is not None:
                lines.append(
                    f"Visible symptom step:        step {analysis.get('visible_at_step')}"
                )
            if analysis.get("reason"):
                lines.append(f"Reason:                      {analysis.get('reason')}")
        else:
            lines.append(
                "No persistent pass/fail screenshot divergence above threshold."
            )
        lines.append(
            "  (Runs with --vlm or --llm --vlm on traces that have screenshots.)"
        )
        return lines

    def _format_text(self, trace_id, ranked, heuristic, heuristic_pixel,
                     heuristic_visual, used_pixel_boost, has_visual_causal,
                     ran_screenshot_scan, screenshot_analysis,
                     llm_ranked, ran_vlm, vlm_only, ranking_mode,
                     diagnosis, summary, eval_result, vlm_output=None,
                     metadata=None) -> str:
        actual = eval_result.get("actual_fault_step") if eval_result else None
        meta = metadata or {}
        is_v2 = meta.get("pipeline_version") == "v2"
        vlm_status = meta.get("vlm_status", "not_run")
        vlm_error = meta.get("vlm_error")
        vlm_error_code = meta.get("vlm_error_code")
        ranking_fallback = meta.get("ranking_fallback")
        exclude_from_aggregate = meta.get("exclude_from_aggregate", False)

        title = "TraceLens v2 Diagnosis Report" if is_v2 else "TraceLens Diagnosis Report"
        lines = [
            "=" * 62,
            f"{title}: {trace_id}",
            f"Ranking mode: {ranking_mode}",
            "=" * 62,
            "",
        ]

        if vlm_status == "failed":
            lines += [
                "VLM STATUS",
                "-" * 40,
                f"  Status: FAILED ({vlm_error_code or 'vlm_error'})",
                f"  Reason: {vlm_error or 'unknown'}",
                f"  Final ranking: {ranking_fallback or ranking_mode} fallback",
            ]
            if exclude_from_aggregate:
                lines.append("  Aggregate: excluded from VLM-only Hit@k metrics")
            else:
                lines.append("  Aggregate: included using LLM/heuristic fallback ranking")
            lines.append("")

        # 1. Heuristic tables (text → visual causal → pixel when applicable)
        if has_visual_causal and heuristic_visual:
            lines += self._format_ranked_list(
                heuristic,
                self._layers_label("HEURISTIC TOP 5", ["text signals only"]),
                actual,
            )
            lines += [""]
            lines += self._format_ranked_list(
                heuristic_visual,
                self._layers_label("HEURISTIC TOP 5", ["text", "visual causal"]),
                actual,
                show_visual_causal=True,
            )
            if used_pixel_boost and heuristic_pixel:
                lines += [""]
                lines += self._format_ranked_list(
                    heuristic_pixel,
                    self._layers_label(
                        "HEURISTIC TOP 5",
                        ["text", "visual causal", "pixel"],
                    ),
                    actual,
                    show_pixel=True,
                    show_visual_causal=True,
                )
        elif used_pixel_boost and heuristic_pixel:
            lines += self._format_ranked_list(
                heuristic,
                self._layers_label("HEURISTIC TOP 5", ["text signals only"]),
                actual,
            )
            lines += [""]
            pixel_layers = ["text", "pixel"]
            if has_visual_causal:
                pixel_layers = ["text", "visual causal", "pixel"]
            lines += self._format_ranked_list(
                heuristic_pixel,
                self._layers_label("HEURISTIC TOP 5", pixel_layers),
                actual,
                show_pixel=True,
                show_visual_causal=has_visual_causal,
            )
        else:
            single_layers = ["text"]
            if has_visual_causal:
                single_layers.append("visual causal")
            if used_pixel_boost:
                single_layers.append("pixel")
            lines += self._format_ranked_list(
                heuristic,
                self._layers_label("HEURISTIC TOP 5", single_layers),
                actual,
                show_visual_causal=has_visual_causal,
                show_pixel=used_pixel_boost,
            )

        # 2. LLM rerank table (when LLM ran)
        if llm_ranked:
            llm_layers = ["text"]
            if ran_screenshot_scan:
                if has_visual_causal:
                    llm_layers.append("visual causal")
                llm_layers.append("pixel")
            llm_title = (
                "LLM RERANK TOP 5 (prior for fusion — not final rank)"
                if is_v2
                else "LLM TOP 5"
            )
            lines += [""]
            lines += self._format_ranked_list(
                llm_ranked,
                self._layers_label(llm_title, llm_layers),
                actual,
                show_visual_causal=has_visual_causal and ran_screenshot_scan,
                show_pixel=ran_screenshot_scan,
            )

        # 3. Final ranking table
        if is_v2:
            fusion_parts = ["text", "navigation", "causal", "symptom", "pixel", "visual causal"]
            if llm_ranked:
                fusion_parts.append("llm_prior")
            if ran_vlm:
                fusion_parts.append("vlm_prior")
            lines += [""]
            lines += self._format_fusion_list(
                ranked,
                self._layers_label("FUSION TOP 5 (final rank)", fusion_parts),
                actual,
            )
        elif ran_vlm:
            vlm_layers = ["text", "pixel", "visual causal", "VLM"]
            if not vlm_only:
                vlm_layers = ["text", "pixel", "visual causal", "VLM", "LLM rank prior"]
            vlm_base = "VLM TOP 5" if vlm_only else "VLM + LLM TOP 5"
            lines += [""]
            lines += self._format_ranked_list(
                ranked,
                self._layers_label(vlm_base, vlm_layers),
                actual,
                show_vlm_score=True,
                show_visual_causal=has_visual_causal,
                show_pixel=ran_screenshot_scan,
            )
        elif not llm_ranked and not used_pixel_boost:
            pass  # heuristic-only v1: single table already shown
        elif llm_ranked:
            pass  # LLM-only v1: final == llm_ranked, already shown
        elif vlm_status == "failed":
            fallback_label = (
                f"FINAL RANKING ({ranking_fallback or ranking_mode} fallback — VLM failed)"
            )
            lines += [""]
            lines += self._format_ranked_list(ranked, fallback_label, actual)

        if ran_screenshot_scan and screenshot_analysis:
            lines += [""] + self._format_screenshot_analysis(screenshot_analysis)
            n_avail = (metadata or {}).get("screenshots_available_count")
            if n_avail is not None:
                lines += [
                    f"  Steps with pass/fail PNG pairs: {n_avail} "
                    f"(pixel/visual-causal skipped on steps marked [no screenshots])",
                ]

        ranking_decisions = (metadata or {}).get("ranking_decisions") if metadata else None
        if ranking_decisions:
            decisions_title = (
                "FUSION DECISIONS (channels and weights)"
                if is_v2
                else "RANKING DECISIONS (rules applied)"
            )
            lines += [
                "",
                decisions_title,
                "-" * 40,
            ]
            for note in ranking_decisions:
                lines.append(f"  • {note}")
            if is_v2 and meta.get("fusion_weights"):
                fw = meta["fusion_weights"]
                w_str = ", ".join(f"{k}={v}" for k, v in sorted(fw.items()))
                lines.append(f"  • Fusion weights: {w_str}")

        lines += [
            "",
            "TECHNICAL ROOT CAUSE",
            "-" * 40,
        ]
        if diagnosis.get("root_cause_step_id") is not None:
            lines += [
                f"Primary step:  Step {diagnosis.get('root_cause_step_id')}",
                f"Summary:       {diagnosis.get('root_cause_summary', '')}",
                f"Failure chain: {diagnosis.get('failure_chain', '')}",
                f"Downstream:    {diagnosis.get('downstream_steps', [])}",
            ]
        else:
            lines.append("(LLM diagnosis not run — heuristic mode)")

        # VLM per-step visual scores (when VLM succeeded)
        if vlm_output and vlm_output.get("steps_with_screenshots", 0) > 0 and vlm_status != "failed":
            lines += [
                "",
                "VISUAL ANALYSIS (VLM)",
                "-" * 40,
                f"Visual root cause: Step {vlm_output.get('visual_root_cause_step_id')}",
                "Summary:",
            ]
            summary = vlm_output.get("visual_summary", "") or ""
            for line in summary.split("\n"):
                if line.strip():
                    lines.append(f"  {line.strip()}")
            lines += [
                "",
                "  Per-step visual scores:",
            ]
            analyzed = sorted(
                vlm_output.get("visual_scores", []),
                key=lambda vs: (-float(vs.get("visual_score") or 0), vs.get("step_id", 0)),
            )
            final_ids = {s.get("step_id") for s in ranked}
            for vs in analyzed:
                sid = vs["step_id"]
                vis = vs.get("visual_score", 0)
                note = vs.get("visual_note", "")[:80]
                tag = "  [in final top-5]" if sid in final_ids else "  [analyzed only]"
                lines.append(
                    f"    Step {sid:>3}  score={vis:.2f}  {note}{tag}"
                )
            lines += [
                "",
                "  Note: VLM summary lists only steps with meaningful visual differences",
                "  (identical/noise responses from per-step API calls are filtered out).",
                "  'Analyzed only' steps were sent to the VLM but ranked outside the final top-5.",
            ]
        elif vlm_status == "failed":
            lines += [
                "",
                "VISUAL ANALYSIS (VLM)",
                "-" * 40,
                "VLM analysis did not complete — no visual scores available.",
                f"Failure code: {vlm_error_code or 'vlm_error'}",
            ]

        lines += [
            "",
            "PLAIN LANGUAGE SUMMARY",
            "-" * 40,
            summary if summary else "(LLM not run)",
        ]

        if eval_result:
            step_dists  = eval_result.get("step_distances", [])
            rank_pos    = eval_result.get("rank_position", -1)
            rank_pos_str = str(rank_pos) if rank_pos > 0 else "not in top-5"

            lines += [
                "",
                "EVALUATION",
                "-" * 40,
            ]
            if exclude_from_aggregate:
                lines.append("(Per-trace eval shown; excluded from VLM-only aggregate Hit@k)")
            lines += [
                f"Actual fault step: {actual}",
                f"Hit@1={eval_result.get('hit@1')}  "
                f"Hit@3={eval_result.get('hit@3')}  "
                f"Hit@5={eval_result.get('hit@5')}",
                f"Rank position of actual fault: {rank_pos_str}",
                f"Overall rank distance (|actual_rank - 1|): {eval_result.get('rank_distance')}",
                "",
                "Per-rank breakdown:",
                f"  {'Pred.Rank':<10}  {'Step':>5}  {'True Rank':>10}  {'rank_dist':>10}  {'step_dist':>10}",
                f"  {'(our rank)':<10}  {'':>5}  {'(1=fault)':>10}  {'|pred-true|':>10}  {'|step-actual|':>13}",
                f"  {'-'*10}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}",
            ]
            for d in step_dists:
                k          = d["rank"]
                is_fault   = d["step_id"] == actual
                true_rank  = "1" if is_fault else "N/A"
                rd         = str(abs(k - 1)) if is_fault else "N/A"
                marker     = "  ← ACTUAL FAULT" if is_fault else ""
                lines.append(
                    f"  {k:<10}  {d['step_id']:>5}  {true_rank:>10}  {rd:>10}  {d['step_distance']:>10}{marker}"
                )
            lines += [
                "",
                f"Top-1 step distance: {eval_result.get('top1_step_distance')}",
                f"MAD@5 (mean step-dist over top-5): {eval_result.get('mad@5')}",
            ]

        lines.append("=" * 62)
        return "\n".join(lines)

    # ── I/O ───────────────────────────────────────────────────────────────────

    def save_json(self, report: dict, output_path: str) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def save_text(self, report: dict, output_path: str) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report.get("_text_report", ""), encoding="utf-8")

    def print_report(self, report: dict) -> None:
        print(report.get("_text_report", json.dumps(report, indent=2)))
