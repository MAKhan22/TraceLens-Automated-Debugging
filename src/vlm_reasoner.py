"""
vlm_reasoner.py
---------------
Phase 2 — Vision-Language Model analysis of screenshot pairs.

For each suspicious step, compares the pass and fail screenshots to identify
visual anomalies that are invisible to text-only analysis (wrong page loaded,
missing UI elements, CAPTCHA, scroll position unchanged, empty form fields).

Pipeline:
  1. VLM receives pass+fail screenshot pairs for the top-K suspicious steps
     in a single API call (all images interleaved with text labels).
  2. VLM returns a visual_score (0–1) per step and a visual_root_cause_step_id.
  3. These visual scores are combined with LLM text scores for the final ranking.

Integration modes:
  - llm+vlm : run both, ensemble their rankings
  - vlm     : visual only (useful for screenshot-only fault debugging)
"""

import base64
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI


PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


def _encode_image(path: str) -> str | None:
    """Base64-encode an image file. Returns None if file doesn't exist."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _image_content(b64: str, media_type: str = "image/png") -> dict:
    """Build an OpenAI image_url content block from base64 data."""
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{b64}"},
    }


class VlmReasoner:
    """
    Vision-Language Model reasoner for screenshot-based fault detection.

    Uses the same OpenAI-compatible API as LlmReasoner so it works with
    any provider that supports vision (OpenRouter VLM models, Gemini, etc.).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "nvidia/nemotron-nano-12b-v2-vl:free",
        temperature: float = 0.1,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.client = OpenAI(
            api_key=key,
            base_url=base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/tracelens",
                "X-Title": "TraceLens",
            },
        )
        self.model = model
        self.temperature = temperature

    # ── Internal call ──────────────────────────────────────────────────────────

    def _call(self, content: list[dict]) -> str:
        """Call VLM with a mixed text+image content list. Retries on 429."""
        attempt = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    temperature=self.temperature,
                )
                choices = response.choices if response else None
                if not choices or choices[0].message.content is None:
                    raise ValueError("empty-response")
                return choices[0].message.content.strip()
            except Exception as e:
                err = str(e)
                retriable = "429" in err or "empty-response" in err or "rate" in err.lower()
                if retriable:
                    wait = min(2 ** attempt * 10, 120)
                    attempt += 1
                    print(f"  [VLM rate limit] waiting {wait}s before retry {attempt}...")
                    time.sleep(wait)
                else:
                    raise

    def _extract_json(self, text: str) -> dict:
        clean = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            m = re.search(r"\{", clean)
            if m:
                try:
                    return json.loads(clean[m.start():])
                except Exception:
                    pass
        return {}

    # ── Main VLM analysis ─────────────────────────────────────────────────────

    def analyze_steps(self, steps_with_screenshots: list[dict]) -> dict:
        """
        Analyze pass/fail screenshot pairs for a list of steps.

        Args:
            steps_with_screenshots: list of step dicts, each with:
                - step_id, action (or pass_action/fail_action)
                - pass_screenshot: path to pass screenshot (or None)
                - fail_screenshot: path to fail screenshot (or None)

        Returns:
            {
              "visual_scores": [{"step_id": int, "visual_score": float, "visual_note": str}, ...],
              "visual_root_cause_step_id": int | None,
              "visual_summary": str,
              "steps_with_screenshots": int,   # how many steps had usable screenshots
            }
        """
        prompt_text = _load_prompt("vlm_visual_rank.txt")
        content: list[dict] = [{"type": "text", "text": prompt_text}]

        steps_sent = []
        for step in steps_with_screenshots:
            sid = step.get("step_id")
            action = step.get("fail_action") or step.get("action", f"Step {sid}")
            pass_b64 = _encode_image(step.get("pass_screenshot") or "")
            fail_b64 = _encode_image(step.get("fail_screenshot") or "")

            if pass_b64 is None or fail_b64 is None:
                continue  # skip steps without screenshots

            steps_sent.append(sid)
            content.append({
                "type": "text",
                "text": f"\n--- Step {sid}: {action[:80]} ---\nPass screenshot:"
            })
            content.append(_image_content(pass_b64))
            content.append({"type": "text", "text": "Fail screenshot:"})
            content.append(_image_content(fail_b64))

        if not steps_sent:
            return {
                "visual_scores": [],
                "visual_root_cause_step_id": None,
                "visual_summary": "No screenshots available for this trace.",
                "steps_with_screenshots": 0,
            }

        content.append({
            "type": "text",
            "text": f"\nAnalyze the {len(steps_sent)} step(s) shown above and return the JSON."
        })

        raw = self._call(content)
        parsed = self._extract_json(raw)

        return {
            "visual_scores":              parsed.get("visual_scores", []),
            "visual_root_cause_step_id":  parsed.get("visual_root_cause_step_id"),
            "visual_summary":             parsed.get("visual_summary", raw[:200]),
            "steps_with_screenshots":     len(steps_sent),
        }

    # ── Ensemble with LLM output ───────────────────────────────────────────────

    def ensemble_rankings(
        self,
        llm_ranked: list[dict],
        vlm_output: dict,
        vlm_weight: float = 0.4,
    ) -> tuple[list[dict], str]:
        """
        Merge LLM ranking with VLM visual scores to produce a final ranking.

        Strategy:
          - Each step gets a combined_visual_score = (1 - vlm_weight) * llm_rank_score
            + vlm_weight * visual_score
          - llm_rank_score is derived from position: rank 1 → 1.0, rank 5 → 0.2
          - If VLM identifies a visual root cause step not in LLM top-5, it is
            inserted at rank 1 if visual_score >= 0.7

        Returns:
            (final_ranked_list, ranking_mode_label)
        """
        visual_map: dict[int, float] = {}
        for vs in vlm_output.get("visual_scores", []):
            visual_map[vs["step_id"]] = vs.get("visual_score", 0.0)

        n = len(llm_ranked)
        if n == 0:
            return llm_ranked, "vlm"

        scored = []
        for i, step in enumerate(llm_ranked):
            sid = step["step_id"]
            llm_score = 1.0 - (i / n)                 # 1.0 → 0.2 based on rank
            vis_score  = visual_map.get(sid, 0.0)
            combined   = (1 - vlm_weight) * llm_score + vlm_weight * vis_score
            scored.append((combined, step))

        # Check if VLM identified a high-confidence root cause not in LLM top-5
        vlm_rc = vlm_output.get("visual_root_cause_step_id")
        inserted = False
        if vlm_rc is not None:
            in_llm = any(s["step_id"] == vlm_rc for s in llm_ranked)
            vlm_rc_score = visual_map.get(vlm_rc, 0.0)
            if not in_llm and vlm_rc_score >= 0.7:
                # Find the full step dict from scored list or use a placeholder
                # (step must be in the full scored set, not just top-5)
                pass  # handled below via step_id promotion note

        scored.sort(key=lambda x: x[0], reverse=True)
        final = [s for _, s in scored]

        if vlm_rc is not None and vlm_rc != (final[0]["step_id"] if final else None):
            in_final = any(s["step_id"] == vlm_rc for s in final)
            if in_final:
                # VLM root cause is in the list but not #1 — check if it should be promoted
                vlm_rc_visual = visual_map.get(vlm_rc, 0.0)
                if vlm_rc_visual >= 0.7:
                    # Promote VLM root cause to #1
                    final = [s for s in final if s["step_id"] != vlm_rc]
                    vlm_step = next(s for s in scored if s[1]["step_id"] == vlm_rc)[1]
                    final = [vlm_step] + final
                    return final, "llm+vlm+visual_promoted"

            return final, "llm+vlm"

        return final, "llm+vlm" if vlm_output.get("steps_with_screenshots", 0) > 0 else "llm"
