"""
ranking_arbitrator.py
---------------------
Shared Hit@1 arbitration for LLM and VLM final ranking.

Single precedence policy for promoting a step to #1 after model/heuristic
ordering. Both pipelines call these rules so guard logic stays aligned.

Policy order (first match wins):
  1. Causal action over observer pixel leader (action caused errors on next verify)
  2. Text-only #1 with action divergence over spurious visual-causal leader
  3. Visible symptom step after visual-causal root (pixel on next step)
  4. Pixel / visual-causal leader with strong pixel (optional VLM confirm)
  5. Pre-heuristic visual-causal #1
  6. VLM-only: pre-VLM anchor, raw VLM root

Observer pixel lock: when the pixel leader is a verify/observer step with
pixel >= STRONG_OBSERVER_PIXEL, later text_causal deterministic promote is
skipped so the visible fault step can stay #1 (github/youtube-style).

Causal-over-observer still runs at strong pixel when text-only #1 is the same
interactive action step as find_causal_root_step (hackernews/npm-style).

Hybrid (`llm+vlm`) uses the same post-blend `arbitrate_hit_at_k` pass as VLM-only
(`vlm_path=True`), after the weighted visual + LLM-rank blend.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.causal_signals import (
    errors_on_next_observer_step,
    find_causal_root_step,
    is_observer_step,
)
from src.visual_signals import best_pixel_signal

# Observer verify step with very strong screenshot diff — blocks text_causal deterministic only
STRONG_OBSERVER_PIXEL = 0.95

_INTERACTIVE_ACTION = re.compile(
    r"\b(open|click|navigate|select|enter|submit|press|type|add|choose|visit)\b",
    re.I,
)


@dataclass
class ArbitrationResult:
    ranked: list[dict]
    notes: list[str]
    lock_reason: str | None = None


def promote_to_top(
    ranked: list[dict],
    step_id: int,
    step_map: dict[int, dict],
    top_k: int,
) -> list[dict]:
    """Move step_id to rank 1, preserving other steps up to top_k."""
    if not ranked:
        return ranked
    by_id = {s["step_id"]: s for s in ranked}
    if step_id not in by_id:
        if step_id not in step_map:
            return ranked[:top_k]
        promoted = step_map[step_id]
        return [promoted] + ranked[: top_k - 1]
    rest = [s for s in ranked if s["step_id"] != step_id]
    promoted = step_map.get(step_id, by_id[step_id])
    return [promoted] + rest[: top_k - 1]


def _step_action(step: dict) -> str:
    fail = step.get("fail_step") or {}
    return fail.get("action") or step.get("action", "")


def _step_action_type(step: dict) -> str:
    fail = step.get("fail_step") or {}
    return fail.get("action_type") or ""


def _has_text_leader_signal(step: dict) -> bool:
    """Strong text-heuristic signal without visual-causal attribution."""
    return (
        float(step.get("action_score") or 0) > 0
        or float(step.get("network_score") or 0) >= 0.65
        or float(step.get("console_score") or 0) >= 0.5
    )


def _is_interactive_causal_step(step: dict) -> bool:
    """Open/click/navigate-style action (not verify/scroll-only symptom steps)."""
    if is_observer_step(_step_action(step), _step_action_type(step)):
        return False
    return bool(_INTERACTIVE_ACTION.search(_step_action(step)))


def _cur_is_visual_wrong_leader(
    cur: dict,
    cur_id: int,
    visual_map: dict[int, float],
) -> bool:
    """Current #1 was likely set by visual-causal, pixel, or VLM on verify."""
    if float(cur.get("visual_causal_score") or 0) > 0:
        return True
    if is_observer_step(_step_action(cur), _step_action_type(cur)):
        if best_pixel_signal(cur) >= 0.65:
            return True
        if visual_map and float(visual_map.get(cur_id, 0)) >= 0.5:
            return True
    return False


def _observer_pixel_leader(
    pixel_boost_ranked: list[dict] | None,
    step_map: dict[int, dict],
) -> tuple[int | None, float]:
    """Return (step_id, pixel) if pixel-boost #1 is an observer step."""
    if not pixel_boost_ranked:
        return None, 0.0
    px1 = pixel_boost_ranked[0]
    sid = px1["step_id"]
    step = step_map.get(sid, px1)
    fail = step.get("fail_step") or {}
    action = fail.get("action") or step.get("action", "")
    action_type = fail.get("action_type") or ""
    if not is_observer_step(action, action_type):
        return None, 0.0
    return sid, best_pixel_signal(step)


def is_strong_observer_pixel_lock(
    lock_reason: str | None,
    top1_step: dict | None,
) -> bool:
    """True when #1 was set by pixel on a verify step with very strong pixel."""
    if lock_reason != "pixel_leader" or not top1_step:
        return False
    fail = top1_step.get("fail_step") or {}
    action = fail.get("action") or top1_step.get("action", "")
    action_type = fail.get("action_type") or ""
    if not is_observer_step(action, action_type):
        return False
    return best_pixel_signal(top1_step) >= STRONG_OBSERVER_PIXEL


def should_apply_deterministic_promote(
    det_root: int,
    det_reason: str | None,
    ranked: list[dict],
    lock_reason: str | None,
    scored_steps: list[dict],
) -> tuple[bool, str]:
    """
    Whether deterministic promote may override current #1.

    Returns (allowed, explanation note fragment).
    """
    if not ranked or det_root is None:
        return False, "no root"

    top1 = ranked[0]
    if top1["step_id"] == det_root:
        return False, "already #1"

    if det_reason == "wrong_navigation":
        return True, "wrong_navigation always applies"

    if det_reason != "text_causal":
        return True, f"deterministic reason {det_reason}"

    if is_strong_observer_pixel_lock(lock_reason, top1):
        return (
            False,
            f"text_causal blocked — strong observer pixel lock on step {top1['step_id']}",
        )

    return True, "text_causal allowed"


def arbitrate_hit_at_k(
    ranked: list[dict],
    scored_steps: list[dict],
    top_k: int,
    *,
    pixel_boost_ranked: list[dict] | None = None,
    text_only_ranked: list[dict] | None = None,
    pre_heuristic_ids: list[int] | None = None,
    visual_map: dict[int, float] | None = None,
    vlm_output: dict | None = None,
    pre_vlm_ranked: list[dict] | None = None,
    path_label: str = "Hit@k guard",
    vlm_path: bool = False,
) -> ArbitrationResult:
    """
    Apply unified Hit@1 arbitration. Pass visual_map/vlm_output only on VLM path.

    vlm_path: True from VLM ensemble (legacy guard parity). Causal-over-observer
    is not blocked by STRONG_OBSERVER_PIXEL. LLM path keeps the strong-pixel gate
    for text_causal deterministic promote only.
    """
    notes: list[str] = []
    if not ranked:
        return ArbitrationResult(ranked=[], notes=notes)

    visual_map = visual_map or {}
    step_map = {s["step_id"]: s for s in scored_steps}
    step_by_id = {s["step_id"]: s for s in ranked}
    top_before = ranked[0]["step_id"]
    scored_list = list(step_map.values())

    def finish(sid: int, note: str, lock: str) -> ArbitrationResult:
        updated = promote_to_top(ranked, sid, step_map, top_k)
        suffix = (
            f" — promoted to #1 (was step {top_before})."
            if updated and updated[0]["step_id"] != top_before
            else f" (step {sid} already #1)."
        )
        notes.append(note + suffix)
        return ArbitrationResult(ranked=updated, notes=notes, lock_reason=lock)

    causal_id = find_causal_root_step(scored_list)

    # 1) Causal action over observer pixel leader
    obs_sid, obs_px = _observer_pixel_leader(pixel_boost_ranked, step_map)
    if obs_sid is not None:
        if causal_id is not None and (
            causal_id in step_by_id or causal_id in step_map
        ):
            causal = step_map[causal_id]
            nxt = step_map.get(causal_id + 1)
            obs = errors_on_next_observer_step(causal, nxt)
            if obs and obs["next_step_id"] == obs_sid:
                if vlm_path:
                    allow = True
                else:
                    allow = obs_px < STRONG_OBSERVER_PIXEL
                    if (
                        not allow
                        and text_only_ranked
                        and text_only_ranked[0]["step_id"] == causal_id
                        and _is_interactive_causal_step(causal)
                    ):
                        allow = True
                if allow:
                    return finish(
                        causal_id,
                        f"{path_label}: causal action step {causal_id} over "
                        f"observer pixel leader step {obs_sid}",
                        "causal_over_observer",
                    )

    # 2) Text-only #1 over spurious visual / observer leader
    if text_only_ranked:
        text_top = text_only_ranked[0]
        tid = text_top["step_id"]
        cur = ranked[0]
        cur_id = cur["step_id"]
        if (
            (tid in step_by_id or tid in step_map)
            and tid != cur_id
            and float(text_top.get("visual_causal_score") or 0) == 0
            and _has_text_leader_signal(text_top)
            and not is_observer_step(_step_action(text_top), _step_action_type(text_top))
            and _cur_is_visual_wrong_leader(cur, cur_id, visual_map)
        ):
            if float(cur.get("visual_causal_score") or 0) > 0:
                # Over spurious VC: require action divergence or proven causal root
                # (avoids promoting early open/nav over click/upvote VC root — producthunt)
                if float(text_top.get("action_score") or 0) > 0 or tid == causal_id:
                    return finish(
                        tid,
                        f"{path_label}: text-only #1 step {tid} (text leader) "
                        f"over visual-causal leader step {cur_id}",
                        "text_action_anchor",
                    )

    # 3) Visible symptom after visual-causal root
    for step in step_map.values():
        nxt = step.get("visual_causal_next_step")
        if nxt is None or nxt not in step_by_id:
            continue
        nxt_step = step_map.get(nxt, {})
        if nxt_step.get("screenshots_available") is False:
            continue
        px_n = best_pixel_signal(nxt_step)
        vis = float(visual_map.get(nxt, 0))
        if visual_map:
            if vis >= 0.5 and px_n >= 0.35:
                return finish(
                    nxt,
                    f"{path_label}: visible symptom step {nxt} after visual-causal root "
                    f"{step['step_id']} (vlm={vis:.2f}, pixel={px_n:.2f})",
                    "visible_symptom",
                )
        elif px_n >= 0.35:
            return finish(
                nxt,
                f"{path_label}: visible symptom step {nxt} after visual-causal root "
                f"{step['step_id']} (pixel={px_n:.2f}, no VLM)",
                "visible_symptom",
            )

    # 4) Pixel / visual-causal leader
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
            vis = float(visual_map.get(sid, 0))
            if visual_map:
                if vis >= 0.5 and (px >= 0.65 or vc > 0):
                    return finish(
                        sid,
                        f"{path_label}: pixel-boost #1 + VLM confirm "
                        f"(pixel={px:.2f}, visual_causal={vc:.2f}, vlm={vis:.2f})",
                        "pixel_leader",
                    )
            elif px >= 0.65 or vc > 0:
                return finish(
                    sid,
                    f"{path_label}: pixel/visual-causal leader step {sid} "
                    f"(pixel={px:.2f}, visual_causal={vc:.2f})",
                    "pixel_leader" if px >= 0.65 else "visual_causal_leader",
                )

    # 5) Pre-heuristic / pre-VLM visual-causal #1
    vc_ids = pre_heuristic_ids or (
        [pre_vlm_ranked[0]["step_id"]] if pre_vlm_ranked else []
    )
    for sid in vc_ids or []:
        step = step_map.get(sid)
        if not step:
            continue
        vc = float(step.get("visual_causal_score") or 0)
        vis = float(visual_map.get(sid, 0))
        if sid not in step_by_id or vc <= 0:
            continue
        if visual_map and vis >= 0.5:
            label = "pre-VLM visual-causal #1 + VLM confirm"
        else:
            label = "visual-causal heuristic #1"
        return finish(
            sid,
            f"{path_label}: {label} step {sid} "
            f"(visual_causal={vc:.2f}"
            + (f", vlm={vis:.2f})" if visual_map else ")"),
            "visual_causal_heuristic",
        )

    # 6) VLM-only: raw VLM root
    if vlm_output:
        vlm_rc = vlm_output.get("visual_root_cause_step_id")
        if vlm_rc is not None and vlm_rc in step_by_id:
            vis = float(visual_map.get(vlm_rc, 0))
            causal_roots = [
                s["step_id"]
                for s in step_map.values()
                if float(s.get("visual_causal_score") or 0) > 0
            ]
            is_downstream = any(vlm_rc > r for r in causal_roots)
            if vis >= 0.65 and not is_downstream:
                return finish(
                    vlm_rc,
                    f"{path_label}: VLM visual root step {vlm_rc} (vlm={vis:.2f})",
                    "vlm_root",
                )
            if is_downstream and vis >= 0.65:
                notes.append(
                    f"{path_label}: skipped VLM root step {vlm_rc} — downstream of "
                    f"visual-causal root(s) {causal_roots}."
                )

    return ArbitrationResult(ranked=ranked[:top_k], notes=notes, lock_reason=None)
