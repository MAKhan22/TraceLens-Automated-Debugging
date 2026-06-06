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
    is_vlm_root: bool = False
    action_changed: bool = False
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

    for other in step_map.values():
        if (
            other.get("visual_causal_reason") == "symptom_on_next_step"
            and other.get("visual_causal_next_step") == sid
        ):
            ch.observer_symptom = max(ch.observer_symptom, px, vlm, 0.75)
            if vlm >= 0.5 or px >= 0.35:
                ch.observer_symptom = max(ch.observer_symptom, 0.85)
            ch.notes.append("visual_symptom_step")
            break

    slim = slim_steps_for_llm([step], compact=True)
    if slim and slim[0].get("action_changed"):
        ch.action_changed = True
        ch.causal_action = max(ch.causal_action, 0.7)

    return ch


def _infer_prefer_vlm_root_observer(
    channels: dict[int, StepChannels],
    vlm_root_step_id: int | None,
) -> bool:
    """
    Infer visual-dominant failure from trace + VLM output only.
    Trust VLM root on observer verify only when LLM does not clearly favor
    the prior causal action (hackernews/npm: pixel on verify is symptom).
    """
    if vlm_root_step_id is None:
        return False
    ch = channels.get(vlm_root_step_id)
    if ch is None or not ch.is_observer:
        return False
    prev = channels.get(vlm_root_step_id - 1)
    if (
        prev is not None
        and prev.is_causal_root
        and prev.llm_prior > ch.llm_prior + 0.05
    ):
        return False
    return ch.pixel >= 0.35 or ch.vlm_prior >= 0.5 or ch.observer_symptom >= 0.75


def _infer_auth_action_leader(
    channels: dict[int, StepChannels],
    top_llm: float,
) -> bool:
    """
    Infer auth/login-style faults from trace channels (action change + causal/text).
    Replaces dataset fault_type labels at inference time.
    """
    return any(
        ch.action_changed
        and ch.llm_prior >= top_llm - 0.02
        and ch.causal_action >= 0.7
        and (ch.is_causal_root or ch.text >= 0.2)
        for ch in channels.values()
    )


def _cap_observer_after_causal_root(
    channels: dict[int, StepChannels],
    *,
    vlm_root_step_id: int | None,
    prefer_vlm_root: bool,
    min_cause_text: float = 0.5,
    cap: float = 0.55,
    vlm_cap: float = 0.2,
    vlm_root_llm_min: float = 0.8,
) -> None:
    """
    When LLM ranks the causal action above the next verify step, dampen
    observer/pixel/vlm on that verify — unless VLM root + LLM both favor it.
    """
    for sid, ch in channels.items():
        if not ch.is_observer:
            continue
        prev = channels.get(sid - 1)
        if prev is None or not prev.is_causal_root or prev.text < min_cause_text:
            continue
        if ch.llm_prior >= prev.llm_prior - 0.02:
            continue
        if (
            prefer_vlm_root
            and vlm_root_step_id is not None
            and sid == vlm_root_step_id
            and ch.llm_prior >= vlm_root_llm_min
        ):
            continue
        tight = cap * 0.55 if prev.llm_prior >= 0.95 else cap
        ch.observer_symptom = min(ch.observer_symptom, tight)
        ch.pixel = min(ch.pixel, tight)
        ch.vlm_prior = min(ch.vlm_prior, vlm_cap)
        ch.causal_action = 0.0
        ch.notes.append("capped_observer_after_causal_root")


def _boost_vlm_visual_root(
    channels: dict[int, StepChannels],
    *,
    vlm_root_step_id: int | None,
    prefer_vlm_root: bool,
    vlm_root_llm_min: float = 0.8,
) -> None:
    """UI/console-fault verify steps VLM names as root (youtube/github)."""
    if not prefer_vlm_root or vlm_root_step_id is None:
        return
    ch = channels.get(vlm_root_step_id)
    if ch is None or not ch.is_observer or ch.llm_prior < vlm_root_llm_min:
        return
    ch.is_vlm_root = True
    ch.vlm_prior = max(ch.vlm_prior, 0.88)
    ch.observer_symptom = max(ch.observer_symptom, 0.88)
    ch.notes.append("vlm_visual_root_boost")


def _apply_llm_leader_rules(channels: dict[int, StepChannels]) -> None:
    """Strengthen steps the LLM ranked #1 with clear action/causal signal."""
    if not channels:
        return
    top_llm = max(ch.llm_prior for ch in channels.values())

    for ch in channels.values():
        if ch.is_causal_root and ch.llm_prior >= top_llm - 0.02:
            ch.causal_action = 1.0
            ch.notes.append("causal_llm_leader")
        if ch.action_changed and ch.llm_prior >= top_llm - 0.02:
            ch.causal_action = 1.0
            ch.notes.append("action_changed_llm_leader")

    has_action_leader = any(
        ch.action_changed and ch.llm_prior >= top_llm - 0.02
        for ch in channels.values()
    )
    if not has_action_leader:
        return
    vc_scale = 0.25 if _infer_auth_action_leader(channels, top_llm) else 0.4
    for ch in channels.values():
        if (
            ch.visual_causal >= 0.85
            and ch.llm_prior < top_llm - 0.05
            and not ch.action_changed
            and not ch.is_causal_root
        ):
            ch.visual_causal *= vc_scale
            ch.notes.append("vc_discounted_for_action_leader")


def build_all_channels(
    scored_steps: list[dict],
    *,
    llm_prior: dict[int, float] | None = None,
    vlm_prior: dict[int, float] | None = None,
    vlm_root_step_id: int | None = None,
    strong_observer_pixel: float = 0.95,
    observer_after_causal_cap: float = 0.55,
    observer_after_causal_min_text: float = 0.5,
    observer_vlm_cap: float = 0.2,
    vlm_root_llm_min: float = 0.8,
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

    channels = {
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
    prefer_vlm = _infer_prefer_vlm_root_observer(channels, vlm_root_step_id)
    _cap_observer_after_causal_root(
        channels,
        vlm_root_step_id=vlm_root_step_id,
        prefer_vlm_root=prefer_vlm,
        min_cause_text=observer_after_causal_min_text,
        cap=observer_after_causal_cap,
        vlm_cap=observer_vlm_cap,
        vlm_root_llm_min=vlm_root_llm_min,
    )
    _boost_vlm_visual_root(
        channels,
        vlm_root_step_id=vlm_root_step_id,
        prefer_vlm_root=prefer_vlm,
        vlm_root_llm_min=vlm_root_llm_min,
    )
    _apply_llm_leader_rules(channels)
    return channels
