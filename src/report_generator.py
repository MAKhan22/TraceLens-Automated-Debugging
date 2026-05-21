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
        ranking_mode: str = "heuristic",
        llm_output: dict | None = None,
        eval_result: dict | None = None,
        metadata: dict | None = None,
    ) -> dict:
        llm_output = llm_output or {}
        diagnosis = llm_output.get("diagnosis", {})

        report = {
            "trace_id":    trace_id,
            "generated":   datetime.utcnow().isoformat() + "Z",
            "metadata":    metadata or {},
            "ranking_mode": ranking_mode,

            "ranked_suspicious_steps": ranked_steps,
            "heuristic_steps":         heuristic_steps or ranked_steps,

            "technical_diagnosis": {
                "root_cause_step_id": diagnosis.get("root_cause_step_id"),
                "root_cause_summary": diagnosis.get("root_cause_summary"),
                "downstream_steps":   diagnosis.get("downstream_steps", []),
                "failure_chain":      diagnosis.get("failure_chain"),
            },

            "stakeholder_summary": llm_output.get("stakeholder_summary", ""),
            "evaluation": eval_result or {},

            "_text_report": self._format_text(
                trace_id, ranked_steps, heuristic_steps or ranked_steps,
                ranking_mode, diagnosis,
                llm_output.get("stakeholder_summary", ""),
                eval_result,
            ),
        }
        return report

    # ── formatting ────────────────────────────────────────────────────────────

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
        return f"  [{', '.join(flags)}]" if flags else ""

    def _format_ranked_list(self, steps: list[dict], label: str,
                             actual_fault: int | None = None) -> list[str]:
        lines = [label, "-" * 40]
        for s in steps:
            rank  = s.get("rank", "?")
            sid   = s.get("step_id", "?")
            act   = s.get("action", "")[:70]
            score = s.get("combined_score", 0)
            flags = self._step_flags(s)
            marker = " ← ACTUAL FAULT" if (actual_fault is not None and sid == actual_fault) else ""
            lines.append(f"  #{rank}  Step {sid}: {act}{flags}  (score={score:.3f}){marker}")
        return lines

    def _format_text(self, trace_id, ranked, heuristic, ranking_mode,
                     diagnosis, summary, eval_result) -> str:
        actual = eval_result.get("actual_fault_step") if eval_result else None

        lines = [
            "=" * 62,
            f"TraceLens Diagnosis Report: {trace_id}",
            f"Ranking mode: {ranking_mode}",
            "=" * 62,
            "",
        ]

        # Final ranking
        lines += self._format_ranked_list(ranked, "FINAL RANKED SUSPICIOUS STEPS", actual)

        # Heuristic ranking if LLM changed it
        heuristic_ids = [s.get("step_id") for s in heuristic]
        final_ids     = [s.get("step_id") for s in ranked]
        if heuristic_ids != final_ids:
            lines += [""]
            lines += self._format_ranked_list(heuristic, "HEURISTIC RANKING (before LLM)", actual)

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
