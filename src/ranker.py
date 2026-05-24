"""
ranker.py
---------
Takes scored steps from anomaly_detector and produces a ranked list.

Two ranking modes:
  "heuristic"  – pure score sort (fast, no LLM needed)
  "llm"        – all steps (or heuristic top-K if pre_llm_k > 0) go to LLM for re-ranking

The heuristic rank is always computed first.
LLM re-ranking is done in llm_reasoner.py and passed back here for final output.
"""

import json
from pathlib import Path

from src.llm_reasoner import slim_steps_for_llm
from src.causal_signals import find_causal_root_step, is_observer_step
from src.navigation_signals import compute_navigation_signals
from src.visual_signals import scan_pixel_scores, annotate_visual_causal_scores


class Ranker:
    def __init__(self, top_k: int = 5, pre_llm_k: int = 10):
        self.top_k = top_k          # final output size
        self.pre_llm_k = pre_llm_k  # candidates sent to LLM for re-ranking

    def rank_heuristic(
        self,
        scored_steps: list[dict],
        *,
        include_visual_causal: bool = False,
    ) -> list[dict]:
        """Sort by combined_score; optionally fold in visual_causal_score."""
        def sort_key(step: dict) -> float:
            text = float(step.get("combined_score") or 0)
            if not include_visual_causal:
                return text
            vc = float(step.get("visual_causal_score") or 0)
            return max(text, vc)

        ranked = sorted(scored_steps, key=sort_key, reverse=True)
        out = []
        for step in ranked[: self.top_k]:
            text = float(step.get("combined_score") or 0)
            vc = float(step.get("visual_causal_score") or 0)
            rank_score = max(text, vc) if include_visual_causal else text
            out.append({**step, "rank_score": round(rank_score, 6)})
        return out

    def candidates_for_llm(self, scored_steps: list[dict]) -> list[dict]:
        """Return candidates to send to the LLM for re-ranking.
        If pre_llm_k is 0, returns all steps so the LLM is not bottlenecked
        by the heuristic ranking (i.e. can discover faults ranked low by heuristic).
        """
        ranked = sorted(scored_steps, key=lambda x: x["combined_score"], reverse=True)
        if self.pre_llm_k == 0:
            return ranked  # all steps
        return ranked[:self.pre_llm_k]

    def apply_llm_reranking(self, llm_order: list[int],
                             scored_steps: list[dict]) -> list[dict]:
        """
        Apply LLM-provided re-ranking.

        Args:
            llm_order:    list of step_ids in LLM-preferred order
            scored_steps: original scored list

        Returns:
            re-ranked list (top_k)
        """
        step_map = {s["step_id"]: s for s in scored_steps}
        reranked = []
        for step_id in llm_order:
            if step_id in step_map:
                reranked.append(step_map[step_id])
        # append any not mentioned by LLM, sorted by score
        mentioned = set(llm_order)
        rest = sorted(
            [s for s in scored_steps if s["step_id"] not in mentioned],
            key=lambda x: x["combined_score"], reverse=True
        )
        reranked.extend(rest)
        return reranked[:self.top_k]

    def diagnosis_candidates(self, llm_ranked: list[dict],
                             scored_steps: list[dict]) -> list[dict]:
        """
        Steps sent to LLM diagnosis. Ensures high-confidence root-cause steps are
        included even if the reranker buried them below the top-k cutoff:
          - wrong_navigation (wrong page loaded)
          - errors_observed_on_next_step (silent action, error on next verify step)
        """
        base = llm_ranked[:self.top_k]
        seen = {s["step_id"] for s in base}
        slim_by_id = {sl["step_id"]: sl for sl in slim_steps_for_llm(scored_steps)}

        def inject(step: dict) -> list[dict]:
            if step["step_id"] not in seen:
                return [step] + base[: self.top_k - 1]
            return base

        # Priority 1: wrong page navigation
        for step in scored_steps:
            sl = slim_by_id.get(step["step_id"], {})
            if sl.get("wrong_navigation"):
                return inject(step)

        # Priority 2: action step whose immediate next verify/wait step logged new errors
        for step in scored_steps:
            sl = slim_by_id.get(step["step_id"], {})
            if is_observer_step(sl.get("fail_action", ""), sl.get("action_type", "")):
                continue
            obs = sl.get("errors_observed_on_next_step")
            if obs and (obs.get("console_errors") or obs.get("network_errors")):
                return inject(step)

        return base

    def find_deterministic_root(
        self,
        scored_steps: list[dict],
        pixel_scores: dict[int, dict] | None = None,
        visual_causal_cfg: dict | None = None,
    ) -> int | None:
        """High-confidence root cause from code signals (no LLM needed)."""
        for step in scored_steps:
            fail = step.get("fail_step") or {}
            pass_ = step.get("pass_step") or {}
            action = fail.get("action") or pass_.get("action") or step.get("action", "")
            nav = compute_navigation_signals(
                pass_.get("network_logs", []),
                fail.get("network_logs", []),
                action,
            )
            if nav["wrong_navigation"]:
                return step["step_id"]

        causal = find_causal_root_step(scored_steps)
        if causal is not None:
            return causal

        if pixel_scores:
            cfg = visual_causal_cfg or {}
            annotated = annotate_visual_causal_scores(scored_steps, pixel_scores, cfg)
            hits = [s for s in annotated if float(s.get("visual_causal_score") or 0) > 0]
            if hits:
                return min(hits, key=lambda s: s["step_id"])["step_id"]
        return None

    def collect_screenshot_paths(
        self,
        scored_steps: list[dict],
        resolver,
        source: str,
        source_base: str,
        trace_cfg: dict,
    ) -> dict[int, tuple[str | None, str | None]]:
        """Map step_id -> (pass_screenshot, fail_screenshot) for all steps."""
        paths: dict[int, tuple[str | None, str | None]] = {}
        for step in scored_steps:
            sid = step["step_id"]
            paths[sid] = resolver.get_paths(source, source_base, trace_cfg, sid)
        return paths

    def pixel_scores_for_trace(
        self,
        scored_steps: list[dict],
        resolver,
        source: str,
        source_base: str,
        trace_cfg: dict,
    ) -> dict[int, dict]:
        paths = self.collect_screenshot_paths(
            scored_steps, resolver, source, source_base, trace_cfg
        )
        return scan_pixel_scores(scored_steps, paths)

    def vlm_candidate_steps(
        self,
        ranked: list[dict],
        scored_steps: list[dict],
        vlm_k: int,
        *,
        inject_step_ids: list[int] | None = None,
    ) -> list[dict]:
        """Build VLM candidate list; inject extra steps (e.g. visual-causal pair)."""
        step_map = {s["step_id"]: s for s in scored_steps}
        seen: set[int] = set()
        candidates: list[dict] = []

        for sid in inject_step_ids or []:
            if sid in step_map and sid not in seen:
                candidates.append(step_map[sid])
                seen.add(sid)

        for step in ranked:
            sid = step["step_id"]
            if sid not in seen:
                candidates.append(step)
                seen.add(sid)
            if len(candidates) >= vlm_k:
                break

        return candidates[: max(vlm_k, len(inject_step_ids or []))]

    def backfill_visual_causal_vlm_scores(
        self,
        vlm_output: dict,
        scored_steps: list[dict],
    ) -> dict:
        """
        When VLM scores the visible consequence step (N+1), attribute that score
        back to the silent causal step (N) using the existing visual_causal link.
        """
        scores = list(vlm_output.get("visual_scores") or [])
        by_id = {s["step_id"]: s for s in scores}

        for step in scored_steps:
            sid = step["step_id"]
            vc = float(step.get("visual_causal_score") or 0)
            if vc <= 0:
                continue

            nxt = step.get("visual_causal_next_step")
            sources: list[tuple[int, float, str]] = []
            if nxt and nxt in by_id:
                nxt_entry = by_id[nxt]
                sources.append((
                    nxt,
                    float(nxt_entry.get("visual_score") or 0),
                    str(nxt_entry.get("visual_note") or ""),
                ))
            # Same-step divergence: attribute max VLM from later analyzed steps
            for other_sid, entry in by_id.items():
                if other_sid <= sid:
                    continue
                sources.append((
                    other_sid,
                    float(entry.get("visual_score") or 0),
                    str(entry.get("visual_note") or ""),
                ))

            if not sources:
                continue
            best_sid, best_score, best_note = max(sources, key=lambda x: x[1])
            if best_score <= 0:
                continue

            if sid in by_id:
                cur = float(by_id[sid].get("visual_score") or 0)
                if best_score > cur:
                    by_id[sid]["visual_score"] = best_score
                    by_id[sid]["visual_note"] = (
                        f"Attributed from step {best_sid}: {best_note[:60]}"
                    )
            else:
                scores.append({
                    "step_id": sid,
                    "visual_score": best_score,
                    "visual_note": f"Attributed from step {best_sid} screenshot diff",
                })
                by_id[sid] = scores[-1]

        vlm_output = {**vlm_output, "visual_scores": scores}
        rc = vlm_output.get("visual_root_cause_step_id")
        if rc is not None:
            for step in scored_steps:
                if step.get("visual_causal_next_step") == rc:
                    vlm_output["visual_root_cause_step_id"] = step["step_id"]
                    break
        return vlm_output

    def promote_step(self, ranked: list[dict], scored_steps: list[dict],
                     step_id: int) -> list[dict]:
        """Move step_id to rank #1, keep remaining order stable."""
        step_map = {s["step_id"]: s for s in scored_steps}
        if step_id not in step_map:
            return ranked
        promoted = [step_map[step_id]]
        rest = [s for s in ranked if s["step_id"] != step_id]
        return self.add_rank_metadata(promoted + rest[: self.top_k - 1])

    def apply_pixel_boost(
        self,
        ranked: list[dict],
        steps_with_shots: list[dict],
        weight: float = 0.35,
    ) -> list[dict]:
        """Re-rank heuristic top-K using local pass/fail pixel diff (heuristic-only mode)."""
        from src.visual_diff import pixel_diff_score

        shot_map = {s["step_id"]: s for s in steps_with_shots}
        boosted = []
        for step in ranked:
            sid = step["step_id"]
            shots = shot_map.get(sid, {})
            px = pixel_diff_score(
                shots.get("pass_screenshot"), shots.get("fail_screenshot")
            )
            h = float(step.get("rank_score") or step.get("combined_score") or 0)
            combined = (1 - weight) * h + weight * px
            boosted.append({
                **step,
                "pixel_score": round(px, 3),
                "combined_score": combined,
                "rank_score": round(combined, 6),
            })
        boosted.sort(key=lambda x: x["combined_score"], reverse=True)
        return boosted[: self.top_k]

    def add_rank_metadata(self, ranked: list[dict]) -> list[dict]:
        """
        Return a new list where each entry is a shallow copy with rank added.
        Does NOT mutate the original dicts — important because scored/heuristic/llm
        lists all share the same underlying dict objects.
        """
        return [{**step, "rank": i + 1} for i, step in enumerate(ranked)]

    def save(self, ranked: list[dict], output_path: str) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # strip bulky pass_step/fail_step from saved output
        slim = []
        for s in ranked:
            slim.append({k: v for k, v in s.items()
                         if k not in ("pass_step", "fail_step")})
        with open(out, "w", encoding="utf-8") as f:
            json.dump(slim, f, indent=2)
