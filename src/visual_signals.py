"""
visual_signals.py
-----------------
Screenshot-based signals: earliest persistent divergence, visual-causal attribution,
and VLM candidate windows.
"""

from src.causal_signals import is_observer_step
from src.visual_diff import pixel_diff_details

_ACTION_TYPES = frozenset({"type", "click", "keyboard_press"})


def scan_pixel_scores(
    scored_steps: list[dict],
    screenshot_paths: dict[int, tuple[str | None, str | None]],
) -> dict[int, dict]:
    """Return pixel diff details per step_id for pass/fail after screenshots."""
    scores: dict[int, dict] = {}
    for step in scored_steps:
        sid = step["step_id"]
        paths = screenshot_paths.get(sid)
        if not paths:
            continue
        pass_path, fail_path = paths
        scores[sid] = pixel_diff_details(pass_path, fail_path)
    return scores


def _effective(scores: dict[int, dict], sid: int) -> float:
    return float(scores.get(sid, {}).get("effective") or 0.0)


def _is_transient_divergence(sid: int, scores: dict[int, dict], min_div: float) -> bool:
    """True when diff at sid does not persist (pass/fail resync on next steps)."""
    if _effective(scores, sid) < min_div:
        return False
    n1 = _effective(scores, sid + 1)
    n2 = _effective(scores, sid + 2)
    resync = n1 < min_div * 0.15 and n2 < min_div * 0.15
    return bool(scores.get(sid + 1, {}).get("identical")) or resync


def find_first_persistent_divergence(
    scored_steps: list[dict],
    pixel_scores: dict[int, dict],
    *,
    min_divergence: float = 0.05,
) -> tuple[int | None, float]:
    """First step where pass/fail visually diverge and the diff persists."""
    for step in sorted(scored_steps, key=lambda s: s["step_id"]):
        sid = step["step_id"]
        details = pixel_scores.get(sid)
        if not details or details.get("identical"):
            continue
        eff = float(details.get("effective") or 0)
        if eff < min_divergence:
            continue
        if _is_transient_divergence(sid, pixel_scores, min_divergence):
            continue
        return sid, eff
    return None, 0.0


def attribute_visual_root(
    divergence_sid: int,
    scored_steps: list[dict],
    pixel_scores: dict[int, dict],
    *,
    max_action_score: float = 0.1,
    max_self_pixel: float = 0.15,
) -> tuple[int, int | None, str]:
    """
    Map a divergence step to the most likely root-cause action step.

    Returns:
        (root_step_id, visible_divergence_step_id, reason)
    """
    by_id = {s["step_id"]: s for s in scored_steps}
    step = by_id.get(divergence_sid)
    if step is None:
        return divergence_sid, divergence_sid, "earliest_divergence"

    fail = step.get("fail_step") or {}
    action_type = fail.get("action_type") or ""
    text_score = float(step.get("combined_score") or 0)

    # Divergence on a silent type/keyboard action → fault is here (missing input)
    if action_type in ("type", "keyboard_press") and text_score <= max_action_score:
        return divergence_sid, divergence_sid, "earliest_localized_divergence"

    # Divergence on click/submit → walk back to silent type/click on previous step
    prev = by_id.get(divergence_sid - 1)
    if prev is not None:
        prev_fail = prev.get("fail_step") or {}
        prev_type = prev_fail.get("action_type") or ""
        prev_text = float(prev.get("combined_score") or 0)
        prev_self = _effective(pixel_scores, divergence_sid - 1)
        if (
            prev_type in _ACTION_TYPES
            and prev_text <= max_action_score
            and prev_self <= max_self_pixel
        ):
            return prev["step_id"], divergence_sid, "symptom_on_next_step"

    if is_observer_step(fail.get("action") or "", action_type):
        if prev is not None and prev_type in _ACTION_TYPES:
            return prev["step_id"], divergence_sid, "symptom_on_verify_step"

    return divergence_sid, divergence_sid, "earliest_divergence"


def annotate_visual_causal_scores(
    scored_steps: list[dict],
    pixel_scores: dict[int, dict],
    cfg: dict | None = None,
) -> list[dict]:
    """
    Add visual_causal_score and metadata to each step.

    Uses earliest *persistent* localized divergence, attributing root cause to the
    silent action step when the visible break appears one step later.
    """
    cfg = cfg or {}
    min_div = cfg.get("min_divergence", 0.05)
    max_action = cfg.get("max_action_score", 0.1)
    max_self = cfg.get("max_self_pixel", 0.15)

    div_sid, div_eff = find_first_persistent_divergence(
        scored_steps, pixel_scores, min_divergence=min_div
    )

    root_sid: int | None = None
    visible_sid: int | None = None
    reason = ""
    if div_sid is not None:
        root_sid, visible_sid, reason = attribute_visual_root(
            div_sid,
            scored_steps,
            pixel_scores,
            max_action_score=max_action,
            max_self_pixel=max_self,
        )

    annotated = []
    for step in scored_steps:
        sid = step["step_id"]
        entry = {
            **step,
            "visual_causal_score": 0.0,
            "visual_causal_next_step": None,
            "visual_divergence_step": None,
            "visual_causal_reason": None,
            "pixel_global": pixel_scores.get(sid, {}).get("global", 0.0),
            "pixel_localized": pixel_scores.get(sid, {}).get("localized", 0.0),
        }
        if sid == root_sid and root_sid is not None:
            visible = visible_sid if visible_sid is not None else div_sid
            score_src = visible if visible != root_sid else div_sid
            score = _effective(pixel_scores, score_src or sid)
            entry["visual_causal_score"] = round(score, 3)
            entry["visual_causal_next_step"] = (
                visible if visible is not None and visible != root_sid else None
            )
            entry["visual_divergence_step"] = div_sid
            entry["visual_causal_reason"] = reason
        annotated.append(entry)

    return annotated


def steps_with_visual_causal_signal(scored_steps: list[dict]) -> list[dict]:
    return [s for s in scored_steps if float(s.get("visual_causal_score") or 0) > 0]


def divergent_window_step_ids(
    scored_steps: list[dict],
    pixel_scores: dict[int, dict],
    cfg: dict | None = None,
) -> list[int]:
    """Step ids around the first persistent divergence for VLM analysis."""
    cfg = cfg or {}
    window = int(cfg.get("divergent_window", 2))
    min_div = cfg.get("min_divergence", 0.05)

    div_sid, _ = find_first_persistent_divergence(
        scored_steps, pixel_scores, min_divergence=min_div
    )
    if div_sid is None:
        return []

    by_id = {s["step_id"]: s for s in scored_steps}
    lo = max(0, div_sid - window)
    hi = div_sid + window
    ids: list[int] = []
    for sid in range(lo, hi + 1):
        if sid in by_id:
            ids.append(sid)

    # Include type/keyboard steps in the window (action-conditioned checks)
    for sid in ids:
        step = by_id[sid]
        at = (step.get("fail_step") or {}).get("action_type") or ""
        if at in ("type", "keyboard_press") and sid not in ids:
            ids.append(sid)

    return sorted(set(ids))


def summarize_screenshot_analysis(
    scored_steps: list[dict],
    pixel_scores: dict[int, dict],
    cfg: dict | None = None,
) -> dict:
    """Summary for reports — always populated when a screenshot scan ran."""
    cfg = cfg or {}
    div_sid, div_eff = find_first_persistent_divergence(
        scored_steps, pixel_scores, min_divergence=cfg.get("min_divergence", 0.05)
    )
    annotated = annotate_visual_causal_scores(scored_steps, pixel_scores, cfg)
    root = next((s for s in annotated if float(s.get("visual_causal_score") or 0) > 0), None)
    return {
        "steps_scanned": len(pixel_scores),
        "first_persistent_divergence": div_sid,
        "divergence_score": round(div_eff, 3) if div_sid is not None else 0.0,
        "attributed_root_step": root["step_id"] if root else None,
        "visual_causal_score": root.get("visual_causal_score") if root else 0.0,
        "visible_at_step": root.get("visual_causal_next_step") if root else None,
        "reason": root.get("visual_causal_reason") if root else None,
        "has_signal": root is not None,
    }


def vlm_inject_step_ids(
    scored_steps: list[dict],
    pixel_scores: dict[int, dict] | None = None,
    cfg: dict | None = None,
) -> list[int]:
    """Steps to prioritize for VLM: divergent window + visual-causal pair."""
    cfg = cfg or {}
    seen: set[int] = set()
    ids: list[int] = []

    def add(sid: int | None) -> None:
        if sid is not None and sid not in seen:
            ids.append(sid)
            seen.add(sid)

    if pixel_scores:
        for sid in divergent_window_step_ids(scored_steps, pixel_scores, cfg):
            add(sid)

    for step in steps_with_visual_causal_signal(scored_steps):
        add(step["step_id"])
        add(step.get("visual_causal_next_step"))

    return ids
