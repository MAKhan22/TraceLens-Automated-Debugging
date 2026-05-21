"""
llm_reasoner.py
---------------
Two-stage LLM pipeline using the Groq API (OpenAI-compatible).

Stage 1 — Re-ranking:
  Takes top-10 heuristic candidates.
  LLM re-orders them by root-cause likelihood.
  Returns ranked step_id list.

Stage 2 — Root cause analysis:
  Takes top-5 re-ranked steps.
  LLM produces structured technical diagnosis (JSON).

Stage 3 — Stakeholder summary:
  Takes the diagnosis JSON.
  LLM produces plain non-technical explanation.

All prompts are loaded from files so they can be edited without touching code.
"""

import json
import re
import os
import time
from pathlib import Path

from openai import OpenAI


PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


def _slim_step_for_llm(step: dict) -> dict:
    """Strip heavy fields, keep only what the LLM needs."""
    fail = step.get("fail_step") or {}
    pass_ = step.get("pass_step") or {}

    # Network diff: only show entries that changed or are new errors
    def net_summary(logs: list) -> list:
        return [
            {"url": e.get("url", ""), "status": e.get("status"), "error": e.get("error")}
            for e in logs
            if e.get("status") and (str(e["status"]).startswith(("4", "5")) or e.get("error"))
        ]

    def con_summary(logs: list) -> list:
        return [
            {"type": e.get("type"), "text": e.get("text", "")[:200]}
            for e in logs
            if e.get("type") in ("error", "severe", "warning")
        ]

    return {
        "step_id":        step.get("step_id"),
        "rank":           step.get("rank"),
        "heuristic_score": step.get("combined_score"),
        "action":         fail.get("action") or pass_.get("action", ""),
        "action_type":    fail.get("action_type") or pass_.get("action_type", ""),
        "intent":         fail.get("intent"),
        "pass_network":   net_summary(pass_.get("network_logs", [])),
        "fail_network":   net_summary(fail.get("network_logs", [])),
        "pass_console":   con_summary(pass_.get("console_logs", [])),
        "fail_console":   con_summary(fail.get("console_logs", [])),
        "network_score":  step.get("network_score"),
        "console_score":  step.get("console_score"),
        "action_score":   step.get("action_score"),
        "intent_score":   step.get("intent_score"),
    }


class LlmReasoner:
    def __init__(self, api_key: str | None = None, model: str = "llama-3.3-70b-versatile",
                 temperature: float = 0.1, base_url: str = "https://api.groq.com/openai/v1"):
        key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    def _call(self, prompt: str, max_retries: int = 5) -> str:
        """Call LLM with exponential backoff on 429 rate limit errors."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                err = str(e)
                if "429" in err and attempt < max_retries - 1:
                    wait = 2 ** attempt * 10  # 10s, 20s, 40s, 80s
                    print(f"  [Groq 429 rate limit] waiting {wait}s before retry {attempt + 1}/{max_retries - 1}...")
                    time.sleep(wait)
                else:
                    raise

    def _extract_json(self, text: str) -> dict | list:
        """Pull JSON from LLM response even if wrapped in markdown."""
        # strip ```json ... ``` fences
        clean = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            # fallback: find first { or [ and parse from there
            m = re.search(r"(\{|\[)", clean)
            if m:
                try:
                    return json.loads(clean[m.start():])
                except Exception:
                    pass
        return {}

    # ── Stage 1: Re-rank ──────────────────────────────────────────────────────

    def rerank(self, candidates: list[dict]) -> list[int]:
        """
        Ask LLM to re-rank the heuristic top-10 candidates.

        Returns: list of step_ids in preferred order.
        """
        slim = [_slim_step_for_llm(s) for s in candidates]
        prompt_template = _load_prompt("rerank_steps.txt")
        prompt = prompt_template.replace("{steps_json}", json.dumps(slim, indent=2))

        raw = self._call(prompt)
        parsed = self._extract_json(raw)

        if isinstance(parsed, dict) and "ranked_step_ids" in parsed:
            return [int(x) for x in parsed["ranked_step_ids"]]
        if isinstance(parsed, list):
            return [int(x) for x in parsed]

        # fallback: return heuristic order
        return [s.get("step_id") for s in candidates]

    # ── Stage 2: Root cause analysis ─────────────────────────────────────────

    def diagnose(self, top_steps: list[dict]) -> dict:
        """
        Produce structured technical root cause analysis from top-5 steps.

        Returns dict with keys: root_cause_step_id, root_cause_summary,
                                downstream_steps, failure_chain
        """
        slim = [_slim_step_for_llm(s) for s in top_steps]
        prompt_template = _load_prompt("root_cause.txt")
        prompt = prompt_template.replace("{steps_json}", json.dumps(slim, indent=2))

        raw = self._call(prompt)
        parsed = self._extract_json(raw)
        if isinstance(parsed, dict):
            return parsed
        return {"root_cause_summary": raw, "raw_response": raw}

    # ── Stage 3: Stakeholder summary ─────────────────────────────────────────

    def stakeholder_summary(self, diagnosis: dict) -> str:
        """
        Convert technical diagnosis into plain non-technical language.

        Returns plain-text string.
        """
        prompt_template = _load_prompt("stakeholder_summary.txt")
        prompt = prompt_template.replace("{diagnosis_json}", json.dumps(diagnosis, indent=2))
        return self._call(prompt)

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def run(self, candidates: list[dict], top_steps: list[dict]) -> dict:
        """
        Run the full two-stage LLM pipeline.

        Stage 1: Re-rank candidates
        Stage 2: Diagnose root cause from re-ranked top steps
        Stage 3: Stakeholder summary

        Returns dict with all LLM outputs plus the final adjusted ranking.
        """
        reranked_ids = self.rerank(candidates)
        time.sleep(2)  # avoid back-to-back 429s on free tier

        diagnosis = self.diagnose(top_steps)
        time.sleep(2)

        summary = self.stakeholder_summary(diagnosis)

        return {
            "llm_reranked_step_ids": reranked_ids,
            "diagnosis":             diagnosis,
            "stakeholder_summary":   summary,
        }
