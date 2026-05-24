"""
anomaly_detector.py
-------------------
Computes a per-step suspicion score for each aligned (pass, fail) step pair.

Score = w_net * network_score
      + w_con * console_score
      + w_act * action_score
      + w_int * intent_score        (ersel only)

All component scores are in [0, 1].

Network scoring rules (ordered by severity):
  - New 5xx status               → 1.0  (server crash / internal error)
  - New 4xx status (404/403/405) → 0.9
  - Status change 2xx → non-2xx  → 0.85
  - New ERR_* / net:: error      → 0.85
  - Any status change            → 0.5
  - Missing requests (pass had some, fail has none)  → 0.7
  - New requests not in pass     → 0.3

Console scoring rules:
  - New "error" / "severe" entry → 0.9 per entry (capped at 1.0)
  - New "warning" entry          → 0.4 per entry (capped at 1.0)
  - Total console entries drop to 0 (was > 0) → 0.5  (silence = suspicious)

Action scoring:
  - Same action text             → 0.0
  - Same type, different content → 0.3
  - Different action type        → 0.6
  - Action present in pass but absent in fail → 0.8

Intent scoring (ersel):
  - "Verification failed" in intent → 1.0
  - "Type failed" / "not found"     → 0.9
  - "Verification passed"           → 0.0
  - None                            → 0.0
"""

import re
from difflib import SequenceMatcher

from src.navigation_signals import compute_navigation_signals
from src.causal_signals import errors_on_next_observer_step


# ── constants ─────────────────────────────────────────────────────────────────

_HTTP_5XX = re.compile(r"^5\d{2}$")
_HTTP_4XX = re.compile(r"^4\d{2}$")
_NET_ERR   = re.compile(r"net::|ERR_|FAILED|err_", re.IGNORECASE)


# ── network scoring ───────────────────────────────────────────────────────────

def _status_severity(status) -> float:
    if status is None:
        return 0.0
    s = str(status)
    if _HTTP_5XX.match(s):
        return 1.0
    if s in ("404", "403", "405", "410", "429"):
        return 0.9
    if _HTTP_4XX.match(s):
        return 0.7
    return 0.0


def _score_network(pass_logs: list, fail_logs: list) -> float:
    if not pass_logs and not fail_logs:
        return 0.0

    # Build url → status maps
    pass_map = {e.get("url", ""): e for e in pass_logs}
    fail_map = {e.get("url", ""): e for e in fail_logs}

    scores = []

    # Check each fail request
    for url, f_entry in fail_map.items():
        f_status = f_entry.get("status")
        f_error  = f_entry.get("error") or ""

        # New network errors
        if _NET_ERR.search(f_error):
            scores.append(0.85)

        if url in pass_map:
            p_status = pass_map[url].get("status")
            if p_status == f_status:
                continue
            # status changed
            if f_status and str(f_status).startswith(("4", "5")):
                scores.append(_status_severity(f_status))
            else:
                scores.append(0.5)
        else:
            # new request not in pass — usually noise, but high-error new requests are suspicious
            if f_status and str(f_status).startswith(("4", "5")):
                scores.append(_status_severity(f_status) * 0.8)

    # Requests present in pass but missing in fail (absence = suspicious)
    missing = [u for u in pass_map if u not in fail_map]
    if missing and len(missing) >= len(pass_map) * 0.5:
        scores.append(0.7)
    elif missing:
        scores.append(0.3)

    # No requests at all in fail but pass had some
    if pass_logs and not fail_logs:
        scores.append(0.7)

    if not scores:
        return 0.0
    return min(1.0, max(scores))  # take worst anomaly score


# ── console scoring ───────────────────────────────────────────────────────────

def _normalise_console_text(text: str) -> str:
    # strip timestamps like "1774953911363: "
    return re.sub(r"^\d{10,}:\s*", "", text).strip()


def _score_console(pass_logs: list, fail_logs: list) -> float:
    pass_texts = {_normalise_console_text(e.get("text", "")) for e in pass_logs}
    fail_texts = {_normalise_console_text(e.get("text", "")) for e in fail_logs}

    new_in_fail = [e for e in fail_logs
                   if _normalise_console_text(e.get("text", "")) not in pass_texts]

    score = 0.0
    for entry in new_in_fail:
        t = entry.get("type", "info").lower()
        if t in ("error", "severe"):
            score += 0.9
        elif t == "warning":
            score += 0.4
        else:
            score += 0.1

    # Silence: pass had console activity, fail has none
    if pass_logs and not fail_logs:
        score = max(score, 0.5)

    return min(1.0, score)


# ── action scoring ────────────────────────────────────────────────────────────

def _score_action(pass_step: dict | None, fail_step: dict | None) -> float:
    if pass_step is None and fail_step is not None:
        return 0.8
    if fail_step is None and pass_step is not None:
        return 0.8
    if pass_step is None and fail_step is None:
        return 0.0

    p_action = pass_step.get("action", "").strip().lower()
    f_action = fail_step.get("action", "").strip().lower()
    p_type   = pass_step.get("action_type", "").strip().lower()
    f_type   = fail_step.get("action_type", "").strip().lower()

    if p_action == f_action:
        return 0.0

    if p_type != f_type:
        return 0.6

    # same type, different content (e.g. typed different text)
    sim = SequenceMatcher(None, p_action, f_action).ratio()
    if sim < 0.5:
        return 0.5
    return 0.3


# ── intent scoring (ersel) ───────────────────────────────────────────────────

def _score_intent(fail_step: dict | None) -> float:
    if not fail_step:
        return 0.0
    intent = (fail_step.get("intent") or "").lower()
    if not intent:
        return 0.0
    if "verification failed" in intent:
        return 1.0
    if "type failed" in intent or "not found" in intent or "not interactable" in intent:
        return 0.9
    if "verification passed" in intent:
        return 0.0
    return 0.0


# ── main scorer ───────────────────────────────────────────────────────────────

class AnomalyDetector:
    def __init__(self, weights: dict | None = None):
        self.base_weights = weights or {
            "network": 0.35,
            "console": 0.25,
            "action":  0.15,
            "intent":  0.25,
        }

    def _effective_weights(self, has_intent: bool) -> dict:
        """
        If no steps carry an intent field, the intent weight is dead weight
        (always multiplied by 0.0). Redistribute it proportionally to the
        other three signals so combined scores still span [0, 1].
        """
        w = dict(self.base_weights)
        if not has_intent:
            freed = w.pop("intent", 0.0)
            others_sum = sum(w.values()) or 1.0
            for k in w:
                w[k] += freed * (w[k] / others_sum)
            w["intent"] = 0.0
        return w

    def score_pair(self, pair: dict, weights: dict) -> dict:
        p = pair.get("pass_step") or {}
        f = pair.get("fail_step") or {}

        step_id = (f or p).get("step_id", -1)
        action  = (f or p).get("action", "")

        net_score    = _score_network(p.get("network_logs", []), f.get("network_logs", []))
        nav          = compute_navigation_signals(
            p.get("network_logs", []), f.get("network_logs", []), action
        )
        if nav["wrong_navigation"]:
            net_score = max(net_score, 0.95)
        con_score    = _score_console(p.get("console_logs", []), f.get("console_logs", []))
        act_score    = _score_action(pair.get("pass_step"), pair.get("fail_step"))
        intent_score = _score_intent(pair.get("fail_step"))

        combined = (
            weights["network"] * net_score +
            weights["console"] * con_score +
            weights["action"]  * act_score +
            weights["intent"]  * intent_score
        )

        return {
            "step_id":        step_id,
            "action":         action,
            "network_score":  round(net_score,    4),
            "console_score":  round(con_score,    4),
            "action_score":   round(act_score,    4),
            "intent_score":   round(intent_score, 4),
            "combined_score": round(combined,     4),
            "pass_step":  pair.get("pass_step"),
            "fail_step":  pair.get("fail_step"),
        }

    def compute_scores(self, aligned_pairs: list[dict]) -> list[dict]:
        has_intent = any(
            (pair.get("fail_step") or {}).get("intent")
            for pair in aligned_pairs
        )
        weights = self._effective_weights(has_intent)
        results = [self.score_pair(pair, weights) for pair in aligned_pairs]
        # Boost action steps whose next verify/wait step logged substantive errors
        for i, res in enumerate(results):
            if i + 1 < len(results) and errors_on_next_observer_step(res, results[i + 1]):
                res["combined_score"] = max(res["combined_score"], 0.72)
                res["network_score"]  = max(res["network_score"], 0.85)
        return results
