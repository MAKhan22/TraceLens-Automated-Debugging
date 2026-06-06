"""
v2/fusion.py
------------
Single-pass fusion scorer: weighted channels → sort once (no post-hoc promotes).
"""

from __future__ import annotations

from dataclasses import dataclass

from v2.features import StepChannels, build_all_channels


@dataclass
class FusionWeights:
    text: float = 0.28
    navigation: float = 0.12
    causal_action: float = 0.20
    observer_symptom: float = 0.20
    visual_causal: float = 0.14
    pixel: float = 0.12
    llm_prior: float = 0.28
    vlm_prior: float = 0.18
    downstream_penalty: float = 0.05
    downstream_text_exempt: float = 0.25  # skip penalty when step has log/text signal
    observer_after_causal_cap: float = 0.55
    observer_after_causal_min_text: float = 0.5
    observer_vlm_cap: float = 0.2
    vlm_root_llm_min: float = 0.75
    causal_llm_bonus: float = 0.06
    vlm_root_bonus: float = 0.05
    action_llm_bonus: float = 0.05

    @classmethod
    def from_config(cls, cfg: dict) -> FusionWeights:
        raw = cfg.get("v2", {}).get("fusion", {})
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})


def _active_channels(
    use_llm: bool,
    use_vlm: bool,
    has_visual: bool,
) -> set[str]:
    active = {"text", "navigation", "causal_action", "observer_symptom"}
    if has_visual:
        active |= {"visual_causal", "pixel"}
    if use_llm:
        active.add("llm_prior")
    if use_vlm:
        active.add("vlm_prior")
    return active


def fusion_score(
    ch: StepChannels,
    weights: FusionWeights,
    active: set[str],
) -> float:
    total_w = 0.0
    score = 0.0

    def add(name: str, value: float, w: float) -> None:
        nonlocal total_w, score
        if name not in active or w <= 0:
            return
        total_w += w
        score += w * min(1.0, max(0.0, value))

    add("text", ch.text, weights.text)
    add("navigation", ch.navigation, weights.navigation)
    add("causal_action", ch.causal_action, weights.causal_action)
    add("observer_symptom", ch.observer_symptom, weights.observer_symptom)
    add("visual_causal", ch.visual_causal, weights.visual_causal)
    add("pixel", ch.pixel, weights.pixel)
    add("llm_prior", ch.llm_prior, weights.llm_prior)
    add("vlm_prior", ch.vlm_prior, weights.vlm_prior)

    if total_w <= 0:
        return 0.0
    fused = score / total_w
    if (
        ch.downstream_of_vc
        and "visual_causal" in active
        and ch.text < weights.downstream_text_exempt
    ):
        fused -= weights.downstream_penalty
    if ch.is_causal_root and ch.llm_prior >= 0.98:
        fused += weights.causal_llm_bonus
    if ch.is_vlm_root and ch.vlm_prior >= 0.65:
        fused += weights.vlm_root_bonus
    if ch.action_changed and ch.llm_prior >= 0.98:
        fused += weights.action_llm_bonus
    return max(0.0, min(1.0, fused))


def llm_prior_from_order(step_ids: list[int]) -> dict[int, float]:
    if not step_ids:
        return {}
    n = max(len(step_ids) - 1, 1)
    return {sid: 1.0 - (i / n) for i, sid in enumerate(step_ids)}


def vlm_prior_from_output(vlm_output: dict) -> dict[int, float]:
    out: dict[int, float] = {}
    for entry in vlm_output.get("visual_scores") or []:
        try:
            sid = int(entry.get("step_id"))
            raw = entry.get("visual_score", entry.get("score"))
            out[sid] = float(raw or 0)
        except (TypeError, ValueError):
            continue
    return out


def rank_by_fusion(
    pool_steps: list[dict],
    *,
    weights: FusionWeights,
    use_llm: bool,
    use_vlm: bool,
    has_visual: bool,
    llm_prior: dict[int, float] | None = None,
    vlm_prior: dict[int, float] | None = None,
    vlm_root_step_id: int | None = None,
    strong_observer_pixel: float = 0.95,
    top_k: int = 5,
) -> tuple[list[dict], list[str], dict[int, StepChannels]]:
    """
    Rank pool steps once by fusion score. Returns (ranked_steps, notes, channels).
    """
    channels = build_all_channels(
        pool_steps,
        llm_prior=llm_prior,
        vlm_prior=vlm_prior,
        vlm_root_step_id=vlm_root_step_id,
        strong_observer_pixel=strong_observer_pixel,
        observer_after_causal_cap=weights.observer_after_causal_cap,
        observer_after_causal_min_text=weights.observer_after_causal_min_text,
        observer_vlm_cap=weights.observer_vlm_cap,
        vlm_root_llm_min=weights.vlm_root_llm_min,
    )
    active = _active_channels(use_llm, use_vlm, has_visual)

    scored_rows: list[tuple[float, int, dict, StepChannels]] = []
    for step in pool_steps:
        sid = step["step_id"]
        ch = channels[sid]
        fs = fusion_score(ch, weights, active)
        row = {
            **step,
            "fusion_score": round(fs, 4),
            "fusion_channels": {
                "text": round(ch.text, 3),
                "navigation": round(ch.navigation, 3),
                "causal_action": round(ch.causal_action, 3),
                "observer_symptom": round(ch.observer_symptom, 3),
                "visual_causal": round(ch.visual_causal, 3),
                "pixel": round(ch.pixel, 3),
                "llm_prior": round(ch.llm_prior, 3),
                "vlm_prior": round(ch.vlm_prior, 3),
            },
        }
        scored_rows.append((fs, sid, row, ch))

    scored_rows.sort(key=lambda x: (-x[0], x[3].is_observer, x[1]))

    notes: list[str] = [
        "V2 fusion rank = weighted channels, single sort (no post-hoc promotes).",
        f"Active channels: {', '.join(sorted(active))}.",
    ]
    if scored_rows:
        top = scored_rows[0]
        ch = top[3]
        if ch.notes:
            notes.append(f"Top-1 channel hints: {', '.join(ch.notes)}.")

    ranked = [row for _, _, row, _ in scored_rows[:top_k]]
    return ranked, notes, channels
