"""
v2/features.py
--------------
Extract interpretable per-step channels for unified fusion ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.causal_signals import (
    errors_on_next_observer_step,
    find_causal_root_step,
    is_observer_step,
)
from src.llm_reasoner import slim_steps_for_llm
from src.navigation_signals import compute_navigation_signals
from src.visual_signals import best_pixel_signal


@dataclass
class StepChannels:
    step_id: int
    text: float = 0.0
    navigation: float = 0.0
    causal_action: float = 0.0
    observer_symptom: float = 0.0
    visual_causal: float = 0.0
    pixel: float = 0.0
    llm_prior: float = 0.0
    vlm_prior: float = 0.0
    is_observer: bool = False
    is_causal_root: bool = False
    downstream_of_vc: bool = False
    notes: list[str] = field(default_factory=list)


def _step_action(step: dict) -> str:
    fail = step.get("fail_step") or {}
    return fail.get("action") or step.get("action", "")


def _step_action_type(step: dict) -> str:
    fail = step.get("fail_step") or {}
    return fail.get("action_type") or ""


def _navigation_score(step: dict) -> float:
    fail = step.get("fail_step") or {}
    pass_ = step.get("pass_step") or {}
    nav = compute_navigation_signals(
        pass_.get("network_logs", []),
        fail.get("network_logs", []),
        _step_action(step),
    )
    if nav.get("wrong_navigation"):
        return 1.0
    if nav.get("navigation_mismatch"):
        return 0.75
    if nav.get("missing_expected_pages"):
        return 0.5
    return 0.0


def _text_score(step: dict) -> float:
    return min(1.0, max(0.0, float(step.get("combined_score") or 0)))


def build_step_channels(
    step: dict,
    *,
    step_map: dict[int, dict],
    causal_root_id: int | None,
    visual_causal_root_id: int | None,
    llm_prior: dict[int, float],
    vlm_prior: dict[int, float],
    strong_observer_pixel: float,
) -> StepChannels:
    sid = step["step_id"]
    action = _step_action(step)
    action_type = _step_action_type(step)
    observer = is_observer_step(action, action_type)
    px = best_pixel_signal(step)
    vc = float(step.get("visual_causal_score") or 0)
    vlm = float(vlm_prior.get(sid, 0))
    llm = float(llm_prior.get(sid, 0))

    ch = StepChannels(
        step_id=sid,
        text=_text_score(step),
        navigation=_navigation_score(step),
        visual_causal=min(1.0, vc),
        pixel=min(1.0, px),
        llm_prior=llm,
        vlm_prior=vlm,
        is_observer=observer,
        is_causal_root=(causal_root_id is not None and sid == causal_root_id),
        downstream_of_vc=(
            visual_causal_root_id is not None
            and sid > visual_causal_root_id
            and vc <= 0
        ),
    )

    if ch.is_causal_root:
        ch.causal_action = 1.0
        ch.notes.append("causal_root")
    else:
        nxt = step_map.get(sid + 1)
        if nxt and not observer:
            obs = errors_on_next_observer_step(step, nxt)
            if obs:
                ch.causal_action = max(ch.causal_action, 0.85)
                ch.notes.append(f"caused_errors_on_step_{obs['next_step_id']}")

    if observer:
        ch.observer_symptom = max(px, vlm * 0.85)
        if px >= strong_observer_pixel:
            ch.observer_symptom = max(ch.observer_symptom, 0.98)
            ch.notes.append("strong_observer_pixel")
        elif vlm >= 0.65 and px >= 0.35:
            ch.observer_symptom = max(ch.observer_symptom, 0.75)
            ch.notes.append("vlm+pixel_symptom")
    elif px >= 0.65 or vlm >= 0.5:
        ch.observer_symptom = max(px, vlm * 0.7) * 0.5

    slim = slim_steps_for_llm([step], compact=True)
    if slim and slim[0].get("action_changed"):
        ch.causal_action = max(ch.causal_action, 0.7)

    return ch


def build_all_channels(
    scored_steps: list[dict],
    *,
    llm_prior: dict[int, float] | None = None,
    vlm_prior: dict[int, float] | None = None,
    strong_observer_pixel: float = 0.95,
) -> dict[int, StepChannels]:
    step_map = {s["step_id"]: s for s in scored_steps}
    causal_root_id = find_causal_root_step(scored_steps)
    vc_roots = [
        s["step_id"]
        for s in scored_steps
        if float(s.get("visual_causal_score") or 0) > 0
    ]
    visual_causal_root_id = min(vc_roots) if vc_roots else None
    llm_prior = llm_prior or {}
    vlm_prior = vlm_prior or {}

    return {
        s["step_id"]: build_step_channels(
            s,
            step_map=step_map,
            causal_root_id=causal_root_id,
            visual_causal_root_id=visual_causal_root_id,
            llm_prior=llm_prior,
            vlm_prior=vlm_prior,
            strong_observer_pixel=strong_observer_pixel,
        )
        for s in scored_steps
    }
