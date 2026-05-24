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

from src.navigation_signals import compute_navigation_signals
from src.causal_signals import is_observer_step, is_noise_console


PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


def _slim_step_for_llm(step: dict) -> dict:
    """Strip heavy fields, keep only what the LLM needs for diagnosis."""
    fail  = step.get("fail_step") or {}
    pass_ = step.get("pass_step") or {}

    def error_entries(logs: list) -> list:
        return [
            {"url": e.get("url", ""), "status": e.get("status"), "error": e.get("error")}
            for e in logs
            if e.get("status") and (str(e["status"]).startswith(("4", "5")) or e.get("error"))
        ]

    def con_errors(logs: list) -> list:
        return [
            {"type": e.get("type"), "text": e.get("text", "")[:200]}
            for e in logs
            if e.get("type") in ("error", "severe", "warning")
        ]

    pass_net_logs  = pass_.get("network_logs", [])
    fail_net_logs  = fail.get("network_logs", [])
    pass_errors    = error_entries(pass_net_logs)
    fail_errors    = error_entries(fail_net_logs)

    pass_error_urls = {e["url"] for e in pass_errors}
    fail_error_urls = {e["url"] for e in fail_errors}

    # Errors new in fail only (genuine signal)
    new_net_errors   = [e for e in fail_errors if e["url"] not in pass_error_urls]
    # Errors in BOTH pass and fail (background noise — LLM should discount these)
    shared_net_noise = [e for e in fail_errors if e["url"] in pass_error_urls]

    # URLs fetched in pass but absent in fail → "absence of activity"
    pass_all_urls = {e.get("url", "") for e in pass_net_logs if e.get("url")}
    fail_all_urls = {e.get("url", "") for e in fail_net_logs if e.get("url")}
    missing_urls  = list(pass_all_urls - fail_all_urls)[:5]  # cap at 5 for token budget

    pass_con_errors = con_errors(pass_.get("console_logs", []))
    fail_con_errors = con_errors(fail.get("console_logs", []))
    pass_con_texts  = {e["text"] for e in pass_con_errors}
    new_con_errors   = [e for e in fail_con_errors if e["text"] not in pass_con_texts]
    new_con_errors   = [e for e in new_con_errors if not is_noise_console(e.get("text", ""))]
    shared_con_noise = [e for e in fail_con_errors if e["text"] in pass_con_texts]

    pass_action = pass_.get("action", "")
    fail_action = fail.get("action", "")
    action_for_nav = fail_action or pass_action

    nav = compute_navigation_signals(pass_net_logs, fail_net_logs, action_for_nav)

    # Asset-only missing URLs (telemetry/fonts) — less diagnostic than page URLs
    missing_pages = [u for u in missing_urls if u.endswith(".html") or "/topics/" in u]
    missing_assets = [u for u in missing_urls if u not in missing_pages]

    slim = {
        "step_id":              step.get("step_id"),
        "heuristic_score":      step.get("combined_score"),
        # Action diff — key for detecting wrong input / navigation / element clicked
        "pass_action":          pass_action,
        "fail_action":          fail_action,
        "action_changed":       pass_action != fail_action,
        "action_type":          fail.get("action_type") or pass_.get("action_type", ""),
        "intent":               fail.get("intent"),
        # Navigation — strongest signal for wrong-element-click faults
        "missing_expected_pages": nav["missing_expected_pages"],
        "wrong_pages_loaded":     nav["wrong_pages_loaded"],
        "wrong_navigation":       nav["wrong_navigation"],
        # Network signals — separated by whether errors are NEW or pre-existing noise
        "new_network_errors":   new_net_errors,    # ONLY in fail → real fault signal
        "shared_network_noise": shared_net_noise,  # in BOTH pass+fail → background noise
        "missing_requests":     missing_urls,       # in pass but gone in fail
        "missing_asset_requests": missing_assets[:3],
        "pass_request_count":   len(pass_net_logs),
        "fail_request_count":   len(fail_net_logs),
        # Console signals — same split
        "new_console_errors":   new_con_errors,    # ONLY in fail → real fault signal
        "shared_console_noise": shared_con_noise,  # in BOTH → background noise
        # Raw scores (for context only — LLM should reason from content above)
        "network_score":        step.get("network_score"),
        "console_score":        step.get("console_score"),
        "action_score":         step.get("action_score"),
        "intent_score":         step.get("intent_score"),
    }
    if float(step.get("visual_causal_score") or 0) > 0:
        slim["visual_causal_score"] = step["visual_causal_score"]
        if step.get("visual_causal_next_step") is not None:
            slim["visual_causal_next_step"] = step["visual_causal_next_step"]
        if step.get("visual_causal_reason"):
            slim["visual_causal_reason"] = step["visual_causal_reason"]
    return slim


def _compact_slim(step: dict) -> dict:
    """Drop empty/low-value fields to reduce rerank token usage on long traces."""
    drop_if_empty = (
        "shared_network_noise", "shared_console_noise", "missing_asset_requests",
        "missing_requests", "new_network_errors", "new_console_errors",
        "missing_expected_pages", "wrong_pages_loaded", "intent",
    )
    out = {}
    for k, v in step.items():
        if k in drop_if_empty and not v:
            continue
        if k.endswith("_score") or k == "heuristic_score":
            if k == "visual_causal_score" and v:
                out[k] = v
            continue
        if v is False and k in ("action_changed", "wrong_navigation"):
            continue
        if k == "pass_action" and v == step.get("fail_action"):
            continue
        out[k] = v
    return out


def _enrich_slim_causal_links(slims: list[dict]) -> list[dict]:
    """
    Link errors logged on step N+1 back to step N when N+1 is a verify/wait step.
    Fixes silent root causes (e.g. scroll/click at 14, API 500 logged at verify step 15).
    """
    by_id = {s["step_id"]: s for s in slims}
    enriched = []
    for sl in slims:
        copy = dict(sl)
        nxt = by_id.get(sl["step_id"] + 1)
        if nxt and is_observer_step(nxt.get("fail_action", ""), nxt.get("action_type", "")):
            con = nxt.get("new_console_errors") or []
            net = nxt.get("new_network_errors") or []
            if con or net:
                copy["errors_observed_on_next_step"] = {
                    "next_step_id":   sl["step_id"] + 1,
                    "next_action":    nxt.get("fail_action"),
                    "console_errors": con,
                    "network_errors": net,
                }
        enriched.append(copy)
    return enriched


def slim_steps_for_llm(candidates: list[dict], *, compact: bool = False) -> list[dict]:
    """Slim steps and add causal cross-step links for LLM consumption."""
    slims = _enrich_slim_causal_links([_slim_step_for_llm(s) for s in candidates])
    if compact:
        return [_compact_slim(s) for s in slims]
    return slims


class LlmReasoner:
    def __init__(self, api_key: str | None = None, model: str = "meta-llama/llama-3.3-70b-instruct:free",
                 temperature: float = 0.1, base_url: str = "https://openrouter.ai/api/v1"):
        key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        # OpenRouter requires HTTP-Referer + X-Title; other providers ignore unknown headers
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

    def _call(self, prompt: str) -> str:
        """Call LLM; retry indefinitely on 429/empty-response, give up on other errors."""
        attempt = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                choices = response.choices if response else None
                if not choices or choices[0].message.content is None:
                    # Provider returned 200 but empty body — treat like a soft rate limit
                    raise ValueError("empty-response")
                return choices[0].message.content.strip()
            except Exception as e:
                err = str(e)
                retriable = "429" in err or "empty-response" in err or "rate" in err.lower()
                if retriable:
                    wait = min(2 ** attempt * 10, 120)  # 10 → 20 → 40 → 80 → 120s cap
                    attempt += 1
                    print(f"  [rate limit] waiting {wait}s before retry {attempt}...")
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
        slim = slim_steps_for_llm(candidates, compact=True)
        prompt_template = _load_prompt("rerank_steps.txt")
        prompt = prompt_template.replace("{steps_json}", json.dumps(slim, separators=(",", ":")))

        raw = self._call(prompt)
        parsed = self._extract_json(raw)

        if not parsed:
            # Truncated/malformed JSON on long traces — retry once with compact payload
            print("  [json parse] malformed rerank response, retrying...")
            time.sleep(2)
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
        slim = slim_steps_for_llm(top_steps)
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

    def run(self, top_steps: list[dict], reranked_ids: list[int]) -> dict:
        """
        Run stages 2 and 3 of the LLM pipeline (re-ranking is done externally).

        Stage 2: Diagnose root cause from re-ranked top steps
        Stage 3: Stakeholder summary

        Args:
            top_steps:    Top-k re-ranked steps (with rank metadata) for diagnosis.
            reranked_ids: The step_id ordering returned by stage-1 rerank (stored in output).

        Returns dict with all LLM outputs.
        """
        diagnosis = self.diagnose(top_steps)
        time.sleep(2)  # avoid back-to-back 429s on free tier

        summary = self.stakeholder_summary(diagnosis)

        return {
            "llm_reranked_step_ids": reranked_ids,
            "diagnosis":             diagnosis,
            "stakeholder_summary":   summary,
        }
