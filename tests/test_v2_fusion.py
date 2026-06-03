"""Unit tests for v2 fusion helpers."""

from v2.fusion import llm_prior_from_order, vlm_prior_from_output


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


if __name__ == "__main__":
    test_vlm_prior_reads_visual_score()
    test_llm_prior_top_one()
    print("ok")
