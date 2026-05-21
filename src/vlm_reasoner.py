"""
vlm_reasoner.py  [PLACEHOLDER — Phase 2]
-----------------------------------------
Vision-language model analysis of screenshot pairs for suspicious steps.

Will use Gemini 2.0 Flash API (supports image input, free tier).
Planned integration:
  - Take top-5 suspicious steps
  - For each: send (screenshot_before, screenshot_after) from pass + fail
  - Ask VLM to describe visual differences relevant to failure
  - Merge visual anomaly score into combined suspicion score

NOT IMPLEMENTED YET. Phase 1 is text-only.
"""


class VlmReasoner:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "VLM reasoning is a Phase 2 feature. "
            "Phase 1 uses text-only analysis via llm_reasoner.py."
        )

    def analyze_step(self, pass_before, pass_after, fail_before, fail_after):
        """Compare screenshot pairs for a suspicious step."""
        raise NotImplementedError

    def score_visual_anomaly(self, steps: list) -> list:
        """Return visual anomaly scores for top-k steps."""
        raise NotImplementedError
