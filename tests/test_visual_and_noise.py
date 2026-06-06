"""Tests for visual attribution and placeholder network noise filtering."""

from src.anomaly_detector import _score_network
from src.causal_signals import is_noise_network
from src.visual_signals import attribute_visual_root


def test_consecutive_click_divergence_stays_on_fault_step():
    """efe/wikipedia: GT on second TOC click, not the prior Economy click."""
    scored = [
        {
            "step_id": 12,
            "combined_score": 0.0,
            "fail_step": {"action_type": "click", "action": "Jump to Economy via table of contents"},
        },
        {
            "step_id": 13,
            "combined_score": 0.0,
            "fail_step": {"action_type": "click", "action": "Jump to Etymology via table of contents"},
        },
    ]
    pixel_scores = {12: {"effective": 0.01}, 13: {"effective": 1.0}}
    root, visible, reason = attribute_visual_root(13, scored, pixel_scores)
    assert root == 13
    assert visible == 13
    assert reason == "earliest_divergence"


def test_type_then_click_still_walks_back():
    """ersel/opencart_purchase: bad type input, visible break on search click."""
    scored = [
        {
            "step_id": 21,
            "combined_score": 0.0,
            "fail_step": {"action_type": "type", "action": "Type 'xyznonexistent123' in search box"},
        },
        {
            "step_id": 22,
            "combined_score": 0.0,
            "fail_step": {"action_type": "click", "action": "Click search button"},
        },
    ]
    pixel_scores = {21: {"effective": 0.01}, 22: {"effective": 0.73}}
    root, visible, reason = attribute_visual_root(22, scored, pixel_scores)
    assert root == 21
    assert visible == 22
    assert reason == "symptom_on_next_step"


def test_example_com_placeholder_url_is_noise():
    assert is_noise_network("https://example.com/assets/logo.png", "net::ERR_ABORTED")
    assert is_noise_network("https://example.com/api/tracking", "")


def test_real_site_urls_are_not_placeholder_noise():
    assert not is_noise_network("https://demo.opencart.com/", "")
    assert not is_noise_network("https://www.wikipedia.org/", "")


def test_score_network_ignores_example_com_distractors():
    pass_logs = [{"url": "https://www.wikipedia.org/", "status": 200, "error": None}]
    fail_logs = [
        {"url": "https://www.wikipedia.org/", "status": 200, "error": None},
        {"url": "https://example.com/assets/logo.png", "status": 500, "error": "net::ERR_ABORTED"},
    ]
    assert _score_network(pass_logs, fail_logs) == 0.0


if __name__ == "__main__":
    test_consecutive_click_divergence_stays_on_fault_step()
    test_type_then_click_still_walks_back()
    test_example_com_placeholder_url_is_noise()
    test_real_site_urls_are_not_placeholder_noise()
    test_score_network_ignores_example_com_distractors()
    print("ok")
