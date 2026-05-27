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
import re
from pathlib import Path

from src.llm_reasoner import slim_steps_for_llm
from src.causal_signals import (
    find_causal_root_step,
    is_observer_step,
)
from src.navigation_signals import compute_navigation_signals
from src.visual_signals import (
    scan_pixel_scores,
    annotate_visual_causal_scores,
    best_pixel_signal,
)


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

        When pre_llm_k > 0, starts from the top-K heuristic steps and injects
        any step with strong code-detected signals (wrong_navigation,
        action_changed, errors_observed_on_next_step, causal root) so the LLM
        can still see important steps the score sort buried.
        pre_llm_k == 0 returns all steps (legacy / max-recall mode).
        """
        ranked = sorted(scored_steps, key=lambda x: x["combined_score"], reverse=True)
        if self.pre_llm_k == 0:
            return ranked

        pool_ids: set[int] = set()
        candidates: list[dict] = []

        def add(step: dict) -> None:
            sid = step["step_id"]
            if sid not in pool_ids:
                pool_ids.add(sid)
                candidates.append(step)

        for step in ranked[: self.pre_llm_k]:
            add(step)

        slim_by_id = {
            s["step_id"]: s
            for s in slim_steps_for_llm(ranked, compact=True)
        }
        for step in scored_steps:
            sl = slim_by_id.get(step["step_id"], {})
            if sl.get("wrong_navigation") or sl.get("action_changed"):
                add(step)
            elif sl.get("errors_observed_on_next_step"):
                add(step)

        causal_id = find_causal_root_step(scored_steps)
        if causal_id is not None:
            step_map = {s["step_id"]: s for s in scored_steps}
            if causal_id in step_map:
                add(step_map[causal_id])

        step_map = {s["step_id"]: s for s in scored_steps}
        for step in scored_steps:
            vc = float(step.get("visual_causal_score") or 0)
            if vc <= 0:
                continue
            add(step)
            nxt = step.get("visual_causal_next_step")
            if nxt is not None and nxt in step_map:
                add(step_map[nxt])

        # Strong global pixel (full scan or report table) — keep visible fault steps in pool
        px_leader = max(
            scored_steps,
            key=lambda s: best_pixel_signal(s),
            default=None,
        )
        if px_leader and best_pixel_signal(px_leader) >= 0.65:
            add(px_leader)

        return candidates

    def apply_llm_reranking(
        self,
        llm_order: list[int],
        scored_steps: list[dict],
        *,
        heuristic_top_ids: list[int] | None = None,
        slim_steps: list[dict] | None = None,
    ) -> tuple[list[dict], list[str]]:
        """
        Apply LLM-provided re-ranking.

        Returns:
            (re-ranked top_k list, anchor-guard notes)
        """
        step_map = {s["step_id"]: s for s in scored_steps}
        slim_by_id = {s["step_id"]: s for s in (slim_steps or [])}
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
        reranked = reranked[: self.top_k]

        if heuristic_top_ids and slim_steps:
            reranked, anchor_notes = self._guard_heuristic_anchors(
                reranked, heuristic_top_ids, step_map, slim_by_id
            )
        else:
            anchor_notes = []
        return reranked[: self.top_k], anchor_notes

    def _guard_heuristic_anchors(
        self,
        llm_ranked: list[dict],
        heuristic_top_ids: list[int],
        step_map: dict[int, dict],
        slim_by_id: dict[int, dict],
    ) -> tuple[list[dict], list[str]]:
        """
        Prevent the LLM from demoting strong heuristic candidates without cause.
        Keeps action_changed / wrong_navigation / causal steps in the top-K.
        """
        notes: list[str] = []
        llm_ids = [s["step_id"] for s in llm_ranked]
        anchors = heuristic_top_ids[: self.top_k]

        def is_strong_anchor(step_id: int) -> bool:
            slim = slim_by_id.get(step_id, {})
            if slim.get("page_load_noise_only"):
                return False
            if slim.get("wrong_navigation") or slim.get("action_changed"):
                return True
            if slim.get("errors_observed_on_next_step"):
                return True
            step = step_map.get(step_id, {})
            if float(step.get("combined_score") or 0) >= 0.55:
                return True
            if float(step.get("pixel_score") or 0) >= 0.65:
                return True
            if float(step.get("visual_causal_score") or 0) > 0:
                return True
            return False

        protected = [sid for sid in anchors if sid in step_map and is_strong_anchor(sid)]
        if not protected:
            return llm_ranked, notes

        notes.append(
            "LLM anchor guard: kept strong heuristic step(s) "
            f"{protected} in top-{self.top_k} (action change, wrong navigation, "
            "high text score, pixel score ≥ 0.65, or visual-causal root)."
        )

        # Pull protected anchors back if LLM dropped them below top-K or demoted far
        out_ids: list[int] = []
        if llm_ids:
            out_ids.append(llm_ids[0])
        for sid in protected:
            if sid not in out_ids:
                out_ids.append(sid)
        for sid in llm_ids[1:]:
            if sid not in out_ids:
                out_ids.append(sid)
        for sid in anchors:
            if sid not in out_ids:
                out_ids.append(sid)

        return [step_map[sid] for sid in out_ids[: self.top_k]], notes

    def guard_llm_hit_at_k(
        self,
        llm_ranked: list[dict],
        scored_steps: list[dict],
        pixel_boost_ranked: list[dict] | None,
        pre_heuristic_ids: list[int] | None,
        top_k: int,
        *,
        text_only_ranked: list[dict] | None = None,
    ) -> tuple[list[dict], list[str], str | None]:
        """
        Post-LLM promotion for Hit@1 using local pixel + visual-causal (no VLM API).
        Skips pixel-based rules when screenshots are unavailable on a step.

        Returns (ranked, notes, top1_lock) where top1_lock describes why #1 was
        set by a visual guard (used to avoid LLM diagnosis overriding pixel/symptom).
        """
        notes: list[str] = []
        if not llm_ranked:
            return llm_ranked, notes, None

        step_map = {s["step_id"]: s for s in scored_steps}
        step_by_id = {s["step_id"]: s for s in llm_ranked}
        top_before = llm_ranked[0]["step_id"]

        def promote(sid: int, note: str, lock: str) -> tuple[list[dict], list[str], str]:
            nonlocal llm_ranked
            if sid not in step_by_id:
                return llm_ranked[:top_k], notes, lock
            notes.append(
                note
                + (f" — promoted to #1 (was step {top_before})."
                   if llm_ranked[0]["step_id"] != sid else f" (step {sid} already #1).")
            )
            if llm_ranked[0]["step_id"] != sid:
                rest = [s for s in llm_ranked if s["step_id"] != sid]
                promoted = step_map.get(sid, step_by_id[sid])
                llm_ranked = [promoted] + rest[: top_k - 1]
            return llm_ranked[:top_k], notes, lock

        # Text-only #1 with action divergence over spurious visual-causal leader
        if text_only_ranked:
            text_top = text_only_ranked[0]
            tid = text_top["step_id"]
            if (
                tid in step_by_id
                and float(text_top.get("action_score") or 0) > 0
                and float(text_top.get("visual_causal_score") or 0) == 0
            ):
                vc_top = llm_ranked[0]["step_id"]
                vc_score = float(llm_ranked[0].get("visual_causal_score") or 0)
                if vc_score > 0 and vc_top != tid:
                    return promote(
                        tid,
                        f"LLM Hit@k guard: text-only #1 step {tid} (action divergence) "
                        f"over visual-causal leader step {vc_top}",
                        "text_action_anchor",
                    )

        # 1) Visible symptom step after visual-causal root
        for step in step_map.values():
            nxt = step.get("visual_causal_next_step")
            if nxt is None or nxt not in step_by_id:
                continue
            nxt_step = step_map.get(nxt, {})
            if nxt_step.get("screenshots_available") is False:
                continue
            px_n = best_pixel_signal(nxt_step)
            if px_n >= 0.35:
                notes.append(
                    f"LLM Hit@k guard: visible symptom step {nxt} after visual-causal root "
                    f"{step['step_id']} (pixel={px_n:.2f}, no VLM)"
                    + (f" — promoted to #1 (was step {top_before})."
                       if llm_ranked[0]["step_id"] != nxt else " (already #1).")
                )
                if llm_ranked[0]["step_id"] != nxt:
                    rest = [s for s in llm_ranked if s["step_id"] != nxt]
                    promoted = step_map.get(nxt, step_by_id[nxt])
                    llm_ranked = [promoted] + rest[: top_k - 1]
                return llm_ranked[:top_k], notes, "visible_symptom"

        # 2) Pixel leader or visual-causal root with strong pixel
        px_source = pixel_boost_ranked or []
        if not px_source:
            leader = max(scored_steps, key=best_pixel_signal, default=None)
            if leader and best_pixel_signal(leader) >= 0.65:
                px_source = [leader]
        if px_source:
            px1 = px_source[0]
            sid = px1["step_id"]
            step = step_map.get(sid, px1)
            if step.get("screenshots_available") is not False and sid in step_by_id:
                px = best_pixel_signal(step)
                vc = float(step.get("visual_causal_score") or 0)
                if px >= 0.65 or vc > 0:
                    notes.append(
                        f"LLM Hit@k guard: pixel/visual-causal leader step {sid} "
                        f"(pixel={px:.2f}, visual_causal={vc:.2f})"
                        + (f" — promoted to #1 (was step {top_before})."
                           if llm_ranked[0]["step_id"] != sid else f" (step {sid} already #1).")
                    )
                    if llm_ranked[0]["step_id"] != sid:
                        rest = [s for s in llm_ranked if s["step_id"] != sid]
                        promoted = step_map.get(sid, step_by_id[sid])
                        llm_ranked = [promoted] + rest[: top_k - 1]
                    lock = "pixel_leader" if px >= 0.65 else "visual_causal_leader"
                    return llm_ranked[:top_k], notes, lock

        # 3) Pre-heuristic visual-causal #1
        if pre_heuristic_ids:
            for sid in pre_heuristic_ids:
                step = step_map.get(sid)
                if not step:
                    continue
                vc = float(step.get("visual_causal_score") or 0)
                if vc > 0 and sid in step_by_id:
                    notes.append(
                        f"LLM Hit@k guard: visual-causal heuristic #1 step {sid} "
                        f"(visual_causal={vc:.2f})"
                        + (f" — promoted to #1 (was step {top_before})."
                           if llm_ranked[0]["step_id"] != sid else f" (step {sid} already #1).")
                    )
                    if llm_ranked[0]["step_id"] != sid:
                        rest = [s for s in llm_ranked if s["step_id"] != sid]
                        promoted = step_map.get(sid, step_by_id[sid])
                        llm_ranked = [promoted] + rest[: top_k - 1]
                    return llm_ranked[:top_k], notes, "visual_causal_heuristic"

        return llm_ranked[:top_k], notes, None

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
        *,
        include_visual_causal: bool = True,
    ) -> tuple[int | None, str | None]:
        """
        High-confidence root cause from code signals (no LLM).

        Returns (step_id, reason) where reason is one of:
          wrong_navigation, text_causal, visual_causal

        On the LLM path, pass include_visual_causal=False so screenshot
        attribution stays in LLM input + post-LLM guards (no double promote).
        """
        strong_later = any(
            float(s.get("combined_score") or 0) >= 0.55
            for s in scored_steps
            if s["step_id"] != 0
        )
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
                if self._is_spurious_homepage_wrong_nav(step, nav, strong_later):
                    continue
                return step["step_id"], "wrong_navigation"

        causal = find_causal_root_step(scored_steps)
        if causal is not None:
            return causal, "text_causal"

        if include_visual_causal and pixel_scores:
            cfg = visual_causal_cfg or {}
            annotated = annotate_visual_causal_scores(scored_steps, pixel_scores, cfg)
            hits = [s for s in annotated if float(s.get("visual_causal_score") or 0) > 0]
            if hits:
                sid = min(hits, key=lambda s: s["step_id"])["step_id"]
                return sid, "visual_causal"
        return None, None

    @staticmethod
    def _is_spurious_homepage_wrong_nav(
        step: dict,
        nav: dict,
        strong_later: bool,
    ) -> bool:
        """Skip step-0 wrong_navigation when a later step has clearer fault signals."""
        if step["step_id"] != 0 or not strong_later:
            return False
        action = (
            (step.get("fail_step") or {}).get("action")
            or (step.get("pass_step") or {}).get("action")
            or step.get("action", "")
        ).lower()
        if not re.search(r"\b(navigate|visit|open|home|load)\b", action):
            return False
        missing = nav.get("missing_expected_pages") or []
        wrong = nav.get("wrong_pages_loaded") or []
        # Homepage telemetry drift: extra pages in fail, no action-relevant page lost
        return bool(wrong) and not missing

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

    def merge_vlm_inject_ids(
        self,
        base_ids: list[int],
        scored_steps: list[dict],
        text_only_ranked: list[dict] | None = None,
    ) -> list[int]:
        """Extend VLM screenshot candidates with text-strong steps the visual layer may bury."""
        seen = set(base_ids)
        out = list(base_ids)

        def add(sid: int | None) -> None:
            if sid is not None and sid not in seen:
                out.append(sid)
                seen.add(sid)

        if text_only_ranked:
            for row in text_only_ranked[:2]:
                add(row["step_id"])
        add(find_causal_root_step(scored_steps))
        for step in scored_steps:
            if float(step.get("action_score") or 0) > 0:
                add(step["step_id"])
        return out

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

    def merge_pixel_scan(
        self,
        scored_steps: list[dict],
        pixel_scores: dict[int, dict],
    ) -> list[dict]:
        """Attach full-trace pixel scan + screenshot availability to each step."""
        merged = []
        for step in scored_steps:
            sid = step["step_id"]
            px = pixel_scores.get(sid)
            if px:
                effective = float(px.get("effective") or px.get("global") or 0)
                merged.append({
                    **step,
                    "screenshots_available": True,
                    "pixel_global": px.get("global", step.get("pixel_global", 0.0)),
                    "pixel_localized": px.get("localized", step.get("pixel_localized", 0.0)),
                    "pixel_score": round(effective, 4),
                })
            else:
                merged.append({
                    **step,
                    "screenshots_available": False,
                })
        return merged

    def merge_pixel_boost(
        self,
        scored_steps: list[dict],
        pixel_boosted: list[dict] | None,
    ) -> list[dict]:
        """Attach pixel_score / boosted rank_score from the pixel table onto scored steps."""
        if not pixel_boosted:
            return scored_steps
        px_by_id = {s["step_id"]: s for s in pixel_boosted}
        merged = []
        for step in scored_steps:
            px = px_by_id.get(step["step_id"])
            if not px:
                merged.append(step)
                continue
            merged.append({
                **step,
                "pixel_score": px.get("pixel_score", step.get("pixel_score")),
                "pixel_global": step.get("pixel_global", px.get("pixel_score")),
            })
        return merged

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
