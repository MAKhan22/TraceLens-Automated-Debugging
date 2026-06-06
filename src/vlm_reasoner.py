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

from src.ranking_arbitrator import arbitrate_hit_at_k
from src.visual_signals import best_pixel_signal


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
        temperature: float = 0.0,
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

    def _compile_visual_summary(self, visual_scores: list[dict]) -> str:
        """Build a readable summary from per-step VLM notes (drop identical/noise)."""
        ranked = sorted(
            visual_scores,
            key=lambda vs: (-float(vs.get("visual_score") or 0), vs.get("step_id", 0)),
        )
        lines: list[str] = []
        seen: set[str] = set()
        for vs in ranked:
            sc = float(vs.get("visual_score") or 0)
            note = str(vs.get("visual_note") or "").strip()
            if sc < 0.35 or not note:
                continue
            low = note.lower()
            if "identical" in low and sc < 0.5:
                continue
            if note in seen:
                continue
            seen.add(note)
            sid = vs.get("step_id", "?")
            lines.append(f"Step {sid} (score={sc:.2f}): {note}")
        if not lines:
            for vs in ranked:
                sc = float(vs.get("visual_score") or 0)
                if sc <= 0:
                    continue
                sid = vs.get("step_id", "?")
                note = str(vs.get("visual_note") or "visual difference detected").strip()
                lines.append(f"Step {sid} (score={sc:.2f}): {note}")
                if len(lines) >= 3:
                    break
        return "\n".join(lines[:5])

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
                root_id = None
                root_score = -1.0
                for step in steps_ready:
                    parsed = self._analyze_one_step(step)
                    for vs in parsed.get("visual_scores") or []:
                        all_scores.append(vs)
                        sc = float(vs.get("visual_score") or 0)
                        if sc > root_score:
                            root_score, root_id = sc, vs.get("step_id")

                result = {
                    "visual_scores": all_scores,
                    "visual_root_cause_step_id": root_id,
                    "visual_summary": self._compile_visual_summary(all_scores),
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

    @staticmethod
    def _pixel_signal(step: dict) -> float:
        """Best available pass/fail pixel diff for Hit@k ranking (global boost or scan)."""
        return best_pixel_signal(step)

    def ensemble_rankings(
        self,
        llm_ranked: list[dict],
        vlm_output: dict,
        vlm_weight: float = 0.4,
        vlm_only: bool = False,
        scored_steps: list[dict] | None = None,
        top_k: int = 5,
        pixel_boost_ranked: list[dict] | None = None,
        text_only_ranked: list[dict] | None = None,
    ) -> tuple[list[dict], str, list[str]]:
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
            return llm_ranked[:top_k], ("vlm" if vlm_only else "llm"), []

        step_map = {s["step_id"]: s for s in (scored_steps or llm_ranked)}
        if pixel_boost_ranked:
            for row in pixel_boost_ranked:
                sid = row["step_id"]
                merged = {
                    **step_map.get(sid, row),
                    **row,
                }
                step_map[sid] = merged

        consequence_of = {
            s["visual_causal_next_step"]: s["step_id"]
            for s in (scored_steps or [])
            if s.get("visual_causal_next_step") is not None
        }

        if vlm_only:
            return self._vlm_only_ranking(
                llm_ranked, visual_map, step_map, consequence_of, top_k,
                vlm_output=vlm_output,
                pixel_boost_ranked=pixel_boost_ranked,
                text_only_ranked=text_only_ranked,
            )

        return self._hybrid_vlm_ranking(
            llm_ranked, visual_map, step_map, consequence_of,
            vlm_output, vlm_weight, top_k,
            pixel_boost_ranked=pixel_boost_ranked,
            text_only_ranked=text_only_ranked,
        )

    def _vlm_step_score(
        self,
        step: dict,
        vis: float,
        *,
        dampen_text: bool = True,
    ) -> float:
        """
        Hit@k-oriented blend: VLM + global pixel + text.
        Strong pixel on the visible fault step can outrank an earlier causal root.
        """
        vc = float(step.get("visual_causal_score") or 0)
        text = float(step.get("rank_score") or step.get("combined_score") or 0)
        px = self._pixel_signal(step)
        if vis >= 0.5:
            score = vis + px * 0.08 + min(text, 0.6) * 0.02
            if vc > 0:
                score += vc * 0.05
            return score
        if vc > 0:
            return vc + px * 0.05 + (text * 0.2 if dampen_text else text * 0.5)
        return text * (0.5 if dampen_text else 1.0) + px * 0.05

    def _is_vlm_consequence(
        self,
        sid: int,
        consequence_of: dict[int, int],
        visual_map: dict[int, float],
        primary_root: int | None,
        visual_roots: dict[int, float],
        step_map: dict[int, dict],
    ) -> bool:
        """Symptom deprioritization — skip when VLM/pixel favor the visible step."""
        if sid in consequence_of:
            parent = consequence_of[sid]
            if float(visual_map.get(sid, 0)) >= float(visual_map.get(parent, 0)):
                return False
            child_step = step_map.get(sid, {})
            parent_step = step_map.get(parent, {})
            if self._pixel_signal(child_step) >= self._pixel_signal(parent_step) + 0.15:
                return False
        if (
            primary_root is not None
            and sid > primary_root
            and sid not in visual_roots
        ):
            if float(visual_map.get(sid, 0)) >= 0.5:
                return False
            return float(visual_map.get(sid, 0)) < 0.5
        return sid in consequence_of

    def _guard_hit_at_k(
        self,
        final: list[dict],
        scored: list[tuple[float, int, dict]],
        visual_map: dict[int, float],
        vlm_output: dict,
        top_k: int,
        *,
        pre_vlm_ranked: list[dict] | None = None,
        pixel_boost_ranked: list[dict] | None = None,
        step_map: dict[int, dict] | None = None,
        text_only_ranked: list[dict] | None = None,
    ) -> tuple[list[dict], list[str]]:
        """Hit@1 arbitration (shared policy — see ranking_arbitrator.py)."""
        if not final or not scored:
            return final, []
        full_map = step_map or {step["step_id"]: step for _, _, step in scored}
        scored_steps = list(full_map.values())
        result = arbitrate_hit_at_k(
            final,
            scored_steps,
            top_k,
            pixel_boost_ranked=pixel_boost_ranked,
            text_only_ranked=text_only_ranked,
            pre_vlm_ranked=pre_vlm_ranked,
            visual_map=visual_map,
            vlm_output=vlm_output,
            path_label="Hit@k guard",
            vlm_path=True,
        )
        return result.ranked, result.notes

    def _guard_pre_vlm_anchor(
        self,
        final: list[dict],
        scored: list[tuple[float, int, dict]],
        pre_vlm_ranked: list[dict],
        visual_map: dict[int, float],
        top_k: int,
    ) -> list[dict]:
        """
        Keep the pre-VLM #1 step at rank #1 when VLM confirms it visually.
        Prevents text/causal boosts on neighbor steps from demoting a correct top-1.
        """
        if not pre_vlm_ranked or not final:
            return final
        anchor_id = pre_vlm_ranked[0]["step_id"]
        if final[0]["step_id"] == anchor_id:
            return final
        anchor_vis = float(visual_map.get(anchor_id, 0))
        if anchor_vis < 0.5:
            return final
        if not any(s["step_id"] == anchor_id for s in final):
            return final
        by_id = {s: c for c, s, _ in scored}
        anchor_combined = by_id.get(anchor_id, 0)
        top_combined = scored[0][0] if scored else 0
        if anchor_combined < top_combined - 0.1:
            return final
        anchor_step = next(step for _, _, step in scored if step["step_id"] == anchor_id)
        rest = [s for s in final if s["step_id"] != anchor_id]
        return [anchor_step] + rest[: top_k - 1]

    def _visual_ensemble_ranking(
        self,
        pre_ranked: list[dict],
        visual_map: dict[int, float],
        step_map: dict[int, dict],
        consequence_of: dict[int, int],
        top_k: int,
        *,
        vlm_output: dict | None = None,
        pixel_boost_ranked: list[dict] | None = None,
        mode_label: str = "vlm",
        vlm_weight: float = 1.0,
        llm_ranked: list[dict] | None = None,
        text_only_ranked: list[dict] | None = None,
    ) -> tuple[list[dict], str, list[str]]:
        """Shared VLM / hybrid ranking via _vlm_step_score + Hit@k guards."""
        if vlm_weight >= 1.0 or not llm_ranked:
            ranking_notes: list[str] = [
                "Final VLM score = VLM API + pixel boost + visual-causal (when present). "
                "Downstream symptom steps rank below attributed cause when scores tie. "
                "Steps without screenshots skip pixel (score treated as 0)."
            ]
        else:
            ranking_notes = [
                f"Hybrid final = {vlm_weight*100:.0f}% unified visual score "
                f"(VLM + pixel + visual-causal) + {(1-vlm_weight)*100:.0f}% LLM rank position. "
                "Steps without screenshots skip pixel (score treated as 0)."
            ]

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

        pool: list[dict] = list(pre_ranked)
        in_pool = {s["step_id"] for s in pool}
        if text_only_ranked:
            for row in text_only_ranked[:2]:
                sid = row["step_id"]
                if sid not in in_pool and sid in step_map:
                    pool.append(step_map[sid])
                    in_pool.add(sid)
        causal_id = find_causal_root_step(list(step_map.values()))
        if causal_id is not None and causal_id in step_map and causal_id not in in_pool:
            pool.append(step_map[causal_id])
            in_pool.add(causal_id)
        for sid in visual_map:
            if sid not in in_pool and sid in step_map:
                pool.append(step_map[sid])
                in_pool.add(sid)

        position_source = llm_ranked or pre_ranked
        n = len(position_source) or 1
        rank_pos = {s["step_id"]: i for i, s in enumerate(position_source)}

        scored: list[tuple[float, int, dict]] = []
        for step in pool:
            sid = step["step_id"]
            vis = float(visual_map.get(sid, 0))
            visual_part = self._vlm_step_score(step, vis)
            if llm_ranked is not None and vlm_weight < 1.0:
                pos = rank_pos.get(sid)
                llm_prior = (1.0 - (pos / n)) if pos is not None else 0.0
                combined = vlm_weight * visual_part + (1 - vlm_weight) * llm_prior
            else:
                combined = visual_part
            if (
                primary_root is not None
                and sid > primary_root
                and sid not in visual_roots
                and float(step.get("visual_causal_score") or 0) == 0
            ):
                combined -= 0.04
            if (
                primary_root is not None
                and sid > primary_root
                and sid not in visual_roots
                and vis < 0.5
            ):
                combined = min(combined, root_floor)
            scored.append((
                combined,
                sid,
                {**step, "vlm_visual_score": round(vis, 3)},
            ))

        def sort_key(item: tuple[float, int, dict]) -> tuple:
            combined, sid, step = item
            is_consequence = self._is_vlm_consequence(
                sid, consequence_of, visual_map, primary_root, visual_roots, step_map
            )
            vc = float(step.get("visual_causal_score") or 0)
            return (-combined, 1 if is_consequence else 0, -vc, sid)

        scored.sort(key=sort_key)
        final = [step for _, _, step in scored[:top_k]]
        is_hybrid = llm_ranked is not None and vlm_weight < 1.0
        if not is_hybrid:
            final = self._guard_pre_vlm_anchor(
                final, scored, pre_ranked, visual_map, top_k
            )
        final, hit_notes = self._guard_hit_at_k(
            final, scored, visual_map, vlm_output or {}, top_k,
            pre_vlm_ranked=pre_ranked,
            pixel_boost_ranked=pixel_boost_ranked,
            step_map=step_map,
            text_only_ranked=text_only_ranked,
        )
        ranking_notes.extend(hit_notes)
        return final, mode_label, ranking_notes

    def _vlm_only_ranking(
        self,
        llm_ranked: list[dict],
        visual_map: dict[int, float],
        step_map: dict[int, dict],
        consequence_of: dict[int, int],
        top_k: int,
        *,
        vlm_output: dict | None = None,
        pixel_boost_ranked: list[dict] | None = None,
        text_only_ranked: list[dict] | None = None,
    ) -> tuple[list[dict], str, list[str]]:
        """Rank by unified visual score (VLM + pixel + visual-causal)."""
        return self._visual_ensemble_ranking(
            llm_ranked, visual_map, step_map, consequence_of, top_k,
            vlm_output=vlm_output,
            pixel_boost_ranked=pixel_boost_ranked,
            mode_label="vlm",
            text_only_ranked=text_only_ranked,
        )

    def _hybrid_vlm_ranking(
        self,
        llm_ranked: list[dict],
        visual_map: dict[int, float],
        step_map: dict[int, dict],
        consequence_of: dict[int, int],
        vlm_output: dict,
        vlm_weight: float,
        top_k: int,
        *,
        pixel_boost_ranked: list[dict] | None = None,
        text_only_ranked: list[dict] | None = None,
    ) -> tuple[list[dict], str, list[str]]:
        """Hybrid: unified visual score blended with LLM rank position."""
        return self._visual_ensemble_ranking(
            llm_ranked, visual_map, step_map, consequence_of, top_k,
            vlm_output=vlm_output,
            pixel_boost_ranked=pixel_boost_ranked,
            mode_label="llm+vlm",
            vlm_weight=vlm_weight,
            llm_ranked=llm_ranked,
            text_only_ranked=text_only_ranked,
        )
