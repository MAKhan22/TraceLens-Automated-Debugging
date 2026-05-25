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


class VlmAnalysisError(Exception):
    """Raised when VLM analysis fails or returns unusable output."""

SINGLE_STEP_PROMPT = """You compare PASS vs FAIL browser screenshots for ONE test step.

Return ONLY valid JSON:
{{
  "visual_scores": [
    {{"step_id": <int>, "visual_score": <0.0-1.0>, "visual_note": "<short reason>"}}
  ],
  "visual_root_cause_step_id": <int or null>,
  "visual_summary": "<one sentence>"
}}

Rules:
- visual_score 0.0 = identical, 1.0 = completely different UI state
- Score layout, visible text, buttons, page content — not minor anti-aliasing
- If FAIL shows wrong page, missing element, error dialog, or different content → score >= 0.7
- visual_root_cause_step_id = step_id with highest meaningful difference, or null if none

Step {step_id}: {action}
"""


def _load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


def _encode_image(path: str | None) -> str | None:
    """Base64-encode an image file. Returns None if path missing or not a file."""
    if not path or not str(path).strip():
        return None
    p = Path(path)
    if not p.is_file():
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
        model: str = "google/gemma-4-31b-it:free",
        temperature: float = 0.1,
        base_url: str = "https://openrouter.ai/api/v1",
        per_step: bool = True,
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
        self.per_step = per_step

    # ── Internal call ──────────────────────────────────────────────────────────

    def _call(self, content: list[dict]) -> str:
        """Call VLM; retry indefinitely on 429/empty-response, give up on other errors."""
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
                    print(f"  [rate limit] waiting {wait}s before retry {attempt}...")
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

    def _validate_result(self, steps_ready: list[dict], result: dict) -> dict:
        """Ensure every step got a usable VLM response."""
        scores = result.get("visual_scores") or []
        if not scores:
            raise VlmAnalysisError(f"VLM returned no visual_scores ({self.model})")

        by_id = {s["step_id"]: s for s in scores if "step_id" in s}
        expected = {s["step_id"] for s in steps_ready}
        missing = expected - set(by_id.keys())
        if missing:
            raise VlmAnalysisError(
                f"VLM missing scores for step(s) {sorted(missing)} ({self.model})"
            )

        for sid, entry in by_id.items():
            note = str(entry.get("visual_note") or "")
            if "VLM error" in note:
                raise VlmAnalysisError(note)

        max_score = max(float(s.get("visual_score") or 0) for s in scores)
        if max_score < 0.01:
            raise VlmAnalysisError(
                f"VLM returned all-zero visual scores — {self.model} did not detect any differences"
            )

        return result

    def _analyze_one_step(self, step: dict) -> dict:
        sid = step.get("step_id")
        action = step.get("fail_action") or step.get("action", f"Step {sid}")
        pass_b64 = _encode_image(step.get("pass_screenshot"))
        fail_b64 = _encode_image(step.get("fail_screenshot"))
        if pass_b64 is None or fail_b64 is None:
            raise VlmAnalysisError(f"Step {sid}: missing pass or fail screenshot")

        content: list[dict] = [
            {"type": "text", "text": SINGLE_STEP_PROMPT.format(step_id=sid, action=action[:120])},
            {"type": "text", "text": "Pass screenshot:"},
            _image_content(pass_b64),
            {"type": "text", "text": "Fail screenshot:"},
            _image_content(fail_b64),
        ]
        raw = self._call(content)
        parsed = self._extract_json(raw)
        if not parsed.get("visual_scores"):
            raise VlmAnalysisError(f"Step {sid}: VLM returned empty or unparseable JSON")
        return parsed

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
        steps_ready = []
        for step in steps_with_screenshots:
            pass_b64 = _encode_image(step.get("pass_screenshot"))
            fail_b64 = _encode_image(step.get("fail_screenshot"))
            if pass_b64 is not None and fail_b64 is not None:
                steps_ready.append(step)

        if not steps_ready:
            raise VlmAnalysisError("No screenshots available for VLM analysis")

        try:
            if self.per_step:
                all_scores: list[dict] = []
                summary_parts: list[str] = []
                root_id = None
                root_score = -1.0
                for step in steps_ready:
                    parsed = self._analyze_one_step(step)
                    for vs in parsed.get("visual_scores") or []:
                        all_scores.append(vs)
                        sc = float(vs.get("visual_score") or 0)
                        if sc > root_score:
                            root_score, root_id = sc, vs.get("step_id")
                    if parsed.get("visual_summary"):
                        summary_parts.append(parsed["visual_summary"])

                result = {
                    "visual_scores": all_scores,
                    "visual_root_cause_step_id": root_id,
                    "visual_summary": "; ".join(summary_parts) if summary_parts else "",
                    "steps_with_screenshots": len(steps_ready),
                }
            else:
                prompt_text = _load_prompt("vlm_visual_rank.txt")
                content: list[dict] = [{"type": "text", "text": prompt_text}]
                steps_sent = []
                for step in steps_ready:
                    sid = step.get("step_id")
                    action = step.get("fail_action") or step.get("action", f"Step {sid}")
                    pass_b64 = _encode_image(step.get("pass_screenshot"))
                    fail_b64 = _encode_image(step.get("fail_screenshot"))
                    steps_sent.append(sid)
                    content.append({
                        "type": "text",
                        "text": f"\n--- Step {sid}: {action[:80]} ---\nPass screenshot:"
                    })
                    content.append(_image_content(pass_b64))
                    content.append({"type": "text", "text": "Fail screenshot:"})
                    content.append(_image_content(fail_b64))

                content.append({
                    "type": "text",
                    "text": f"\nAnalyze the {len(steps_sent)} step(s) shown above and return the JSON."
                })
                raw = self._call(content)
                parsed = self._extract_json(raw)
                if not parsed.get("visual_scores"):
                    raise VlmAnalysisError("VLM batch call returned empty or unparseable JSON")
                result = {
                    "visual_scores": parsed.get("visual_scores", []),
                    "visual_root_cause_step_id": parsed.get("visual_root_cause_step_id"),
                    "visual_summary": parsed.get("visual_summary", raw[:200]),
                    "steps_with_screenshots": len(steps_sent),
                }
        except VlmAnalysisError:
            raise
        except Exception as e:
            raise VlmAnalysisError(str(e)) from e

        return self._validate_result(steps_ready, result)

    # ── Ensemble with LLM output ───────────────────────────────────────────────

    def ensemble_rankings(
        self,
        llm_ranked: list[dict],
        vlm_output: dict,
        vlm_weight: float = 0.4,
        vlm_only: bool = False,
        scored_steps: list[dict] | None = None,
        top_k: int = 5,
    ) -> tuple[list[dict], str]:
        """
        Merge text ranking with VLM visual scores to produce a final ranking.

        vlm_only: rank all VLM-analyzed steps by visual score (consequence steps
                  rank below their attributed cause when scores tie).
        llm+vlm:  blend position-based text score with VLM score for the candidate
                  pool; high-scoring analyzed steps outside the text top-k can enter.
        """
        visual_map: dict[int, float] = {}
        for vs in vlm_output.get("visual_scores", []):
            visual_map[vs["step_id"]] = vs.get("visual_score", 0.0)

        if not visual_map:
            return llm_ranked[:top_k], "vlm" if vlm_only else "llm"

        step_map = {s["step_id"]: s for s in (scored_steps or llm_ranked)}
        consequence_of = {
            s["visual_causal_next_step"]: s["step_id"]
            for s in (scored_steps or [])
            if s.get("visual_causal_next_step") is not None
        }

        if vlm_only:
            return self._vlm_only_ranking(
                llm_ranked, visual_map, step_map, consequence_of, top_k
            )

        return self._hybrid_vlm_ranking(
            llm_ranked, visual_map, step_map, consequence_of,
            vlm_output, vlm_weight, top_k,
        )

    def _vlm_only_ranking(
        self,
        llm_ranked: list[dict],
        visual_map: dict[int, float],
        step_map: dict[int, dict],
        consequence_of: dict[int, int],
        top_k: int,
    ) -> tuple[list[dict], str]:
        """Rank by max(visual-causal/heuristic, VLM); root beats downstream symptoms."""
        visual_roots = {
            s["step_id"]: float(s.get("visual_causal_score") or 0)
            for s in step_map.values()
            if float(s.get("visual_causal_score") or 0) > 0
        }
        primary_root = (
            max(visual_roots, key=lambda k: (visual_roots[k], -k))
            if visual_roots else None
        )
        root_floor = 0.0
        if primary_root is not None:
            root_step = step_map.get(primary_root, {})
            root_floor = max(
                visual_roots[primary_root],
                float(root_step.get("rank_score") or 0),
                float(root_step.get("combined_score") or 0),
            )

        pool: list[dict] = list(llm_ranked)
        in_pool = {s["step_id"] for s in pool}
        for sid in visual_map:
            if sid not in in_pool and sid in step_map:
                pool.append(step_map[sid])
                in_pool.add(sid)

        scored: list[tuple[float, int, dict]] = []
        for step in pool:
            sid = step["step_id"]
            base = max(
                float(step.get("visual_causal_score") or 0),
                float(step.get("rank_score") or 0),
                float(step.get("combined_score") or 0),
            )
            vis = float(visual_map.get(sid, 0))
            combined = max(base, vis)
            if (
                primary_root is not None
                and sid > primary_root
                and sid not in visual_roots
            ):
                combined = min(combined, root_floor)
            scored.append((
                combined,
                sid,
                {**step, "vlm_visual_score": round(vis, 3)},
            ))

        def sort_key(item: tuple[float, int, dict]) -> tuple:
            combined, sid, _ = item
            is_consequence = sid in consequence_of or (
                primary_root is not None
                and sid > primary_root
                and sid not in visual_roots
            )
            return (-combined, 1 if is_consequence else 0, sid)

        scored.sort(key=sort_key)
        return [step for _, _, step in scored[:top_k]], "vlm"

    def _hybrid_vlm_ranking(
        self,
        llm_ranked: list[dict],
        visual_map: dict[int, float],
        step_map: dict[int, dict],
        consequence_of: dict[int, int],
        vlm_output: dict,
        vlm_weight: float,
        top_k: int,
    ) -> tuple[list[dict], str]:
        """Blend text rank position with VLM scores; include strong VLM-only steps."""
        pool: list[dict] = list(llm_ranked)
        in_pool = {s["step_id"] for s in pool}

        for sid, vis in visual_map.items():
            if sid in in_pool or sid not in step_map:
                continue
            if float(vis) >= 0.5:
                pool.append({**step_map[sid], "vlm_visual_score": round(float(vis), 3)})
                in_pool.add(sid)

        n = len(llm_ranked) or 1
        rank_pos = {s["step_id"]: i for i, s in enumerate(llm_ranked)}

        scored: list[tuple[float, int, dict]] = []
        for step in pool:
            sid = step["step_id"]
            pos = rank_pos.get(sid)
            text_score = (1.0 - (pos / n)) if pos is not None else 0.0
            vis_score = float(visual_map.get(sid, 0.0))
            combined = (1 - vlm_weight) * text_score + vlm_weight * vis_score
            scored.append((
                combined,
                sid,
                {**step, "vlm_visual_score": round(vis_score, 3)},
            ))

        def sort_key(item: tuple[float, int, dict]) -> tuple:
            combined, sid, _ = item
            is_consequence = sid in consequence_of
            return (-combined, 1 if is_consequence else 0, sid)

        scored.sort(key=sort_key)
        final = [step for _, _, step in scored[:top_k]]

        vlm_rc = vlm_output.get("visual_root_cause_step_id")
        base_mode = "llm+vlm"
        if vlm_rc is not None and final and final[0]["step_id"] != vlm_rc:
            vlm_rc_visual = float(visual_map.get(vlm_rc, 0.0))
            if vlm_rc_visual >= 0.7 and any(s["step_id"] == vlm_rc for s in final):
                final = [s for s in final if s["step_id"] != vlm_rc]
                promoted = next(s for _, _, s in scored if s["step_id"] == vlm_rc)
                final = [promoted] + final[: top_k - 1]
                return final, f"{base_mode}+visual_promoted"

        return final, base_mode
