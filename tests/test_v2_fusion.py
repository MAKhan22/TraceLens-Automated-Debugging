"""Unit tests for v2 fusion helpers."""

from v2.fusion import (
    FusionWeights,
    fusion_score,
    llm_prior_from_order,
    rank_by_fusion,
    vlm_prior_from_output,
)
from v2.features import StepChannels, build_all_channels


def test_vlm_prior_reads_visual_score():
    out = vlm_prior_from_output(
        {
            "visual_scores": [
                {"step_id": 12, "visual_score": 0.7, "visual_note": "diff"},
                {"step_id": 13, "score": 0.5},
            ]
        }
    )
    assert out[12] == 0.7
    assert out[13] == 0.5


def test_llm_prior_top_one():
    prior = llm_prior_from_order([20, 14, 5])
    assert prior[20] == 1.0
    assert prior[5] == 0.0


def test_downstream_penalty_skipped_for_strong_text():
    """Login click with 401 (saucedemo_1): text signal exempts step from vc downstream penalty."""
    w = FusionWeights()
    active = {"text", "observer_symptom", "pixel", "visual_causal", "llm_prior", "vlm_prior",
              "navigation", "causal_action"}
    ch_weak = StepChannels(8, downstream_of_vc=True, text=0.0, observer_symptom=1.0, pixel=1.0)
    ch_strong = StepChannels(
        7, downstream_of_vc=True, text=0.327, observer_symptom=0.5,
        pixel=1.0, llm_prior=1.0, vlm_prior=0.8,
    )
    assert fusion_score(ch_weak, w, active) < fusion_score(ch_strong, w, active)


def test_vlm_root_capped_when_llm_favors_causal():
    """hackernews/npm: VLM root on verify must not beat causal when LLM ranks cause #1."""
    steps = [
        {
            "step_id": 12,
            "combined_score": 0.72,
            "network_score": 0.85,
            "fail_step": {"action": "Open comments", "action_type": "click"},
            "pass_step": {"step_id": 12},
        },
        {
            "step_id": 13,
            "combined_score": 0.333,
            "pixel_score": 0.835,
            "fail_step": {
                "step_id": 13,
                "action": "Verify comments page loaded",
                "action_type": "verify",
                "console_logs": [{"type": "error", "text": "500 Internal Server Error"}],
            },
            "pass_step": {"step_id": 13, "console_logs": []},
        },
    ]
    channels = build_all_channels(
        steps,
        llm_prior={12: 1.0, 13: 0.857},
        vlm_prior={13: 0.88},
        vlm_root_step_id=13,
    )
    assert channels[13].observer_symptom <= 0.55
    assert "vlm_visual_root_boost" not in channels[13].notes


def test_vlm_root_observer_not_capped_when_llm_favors_verify():
    """youtube: VLM root verify step keeps full channels when LLM ranks verify #1."""
    steps = [
        {
            "step_id": 12,
            "combined_score": 0.72,
            "fail_step": {"action": "Scroll", "action_type": "click"},
            "pass_step": {"step_id": 12, "console_logs": []},
        },
        {
            "step_id": 13,
            "combined_score": 0.333,
            "pixel_score": 1.0,
            "fail_step": {
                "step_id": 13,
                "action": "Verify comments",
                "action_type": "verify",
                "console_logs": [{"type": "error", "text": "400 Bad Request"}],
            },
            "pass_step": {"step_id": 13, "console_logs": []},
        },
    ]
    channels = build_all_channels(
        steps,
        llm_prior={12: 0.5, 13: 1.0},
        vlm_prior={13: 0.7},
        vlm_root_step_id=13,
    )
    assert channels[13].is_vlm_root
    assert channels[13].observer_symptom >= 0.85
    assert "capped_observer_after_causal_root" not in channels[13].notes


def test_observer_cap_when_llm_favors_causal_root():
    """npm/hackernews: verify step 17/13 must not beat causal root when LLM ranks cause #1."""
    steps = [
        {
            "step_id": 16,
            "combined_score": 0.72,
            "network_score": 0.85,
            "fail_step": {"action": "Navigate to tab", "action_type": "click"},
            "pass_step": {"step_id": 16},
        },
        {
            "step_id": 17,
            "combined_score": 0.33,
            "console_score": 1.0,
            "pixel_score": 1.0,
            "fail_step": {
                "step_id": 17,
                "action": "Verify list visible",
                "action_type": "verify",
                "console_logs": [{"type": "error", "text": "500 Internal Server Error"}],
            },
            "pass_step": {"step_id": 17, "console_logs": []},
        },
    ]
    channels = build_all_channels(
        steps,
        llm_prior={16: 1.0, 17: 0.93},
        vlm_prior={17: 0.7},
    )
    assert channels[17].observer_symptom <= 0.55
    assert channels[16].is_causal_root


def test_observer_not_capped_when_llm_favors_verify():
    """youtube good run: LLM #1 on verify step — leave observer channels intact."""
    steps = [
        {
            "step_id": 12,
            "combined_score": 0.72,
            "network_score": 0.85,
            "fail_step": {"action": "Scroll to comments", "action_type": "click"},
            "pass_step": {"step_id": 12},
        },
        {
            "step_id": 13,
            "combined_score": 0.33,
            "pixel_score": 1.0,
            "fail_step": {
                "step_id": 13,
                "action": "Verify comments loaded",
                "action_type": "verify",
                "console_logs": [{"type": "error", "text": "400 Bad Request"}],
            },
            "pass_step": {"step_id": 13, "console_logs": []},
        },
    ]
    channels = build_all_channels(
        steps,
        llm_prior={12: 0.5, 13: 1.0},
        vlm_prior={13: 0.7},
    )
    assert channels[13].observer_symptom >= 0.75


def test_downstream_penalty_applies_without_text():
    w = FusionWeights()
    active = {"text", "visual_causal", "observer_symptom", "pixel"}
    ch = StepChannels(8, downstream_of_vc=True, text=0.0, observer_symptom=1.0, pixel=1.0)
    penalized = fusion_score(ch, w, active)
    w_no_pen = FusionWeights(downstream_penalty=0.0)
    assert penalized < fusion_score(ch, w_no_pen, active)


if __name__ == "__main__":
    test_vlm_prior_reads_visual_score()
    test_llm_prior_top_one()
    test_downstream_penalty_skipped_for_strong_text()
    test_vlm_root_capped_when_llm_favors_causal()
    test_vlm_root_observer_not_capped_when_llm_favors_verify()
    test_observer_cap_when_llm_favors_causal_root()
    test_observer_not_capped_when_llm_favors_verify()
    test_downstream_penalty_applies_without_text()
    print("ok")
