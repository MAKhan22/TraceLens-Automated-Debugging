"""
causal_signals.py
-----------------
Detect silent root-cause steps where errors appear on the next verify/wait step.
Complements navigation_signals (wrong page) for API-timeout / scroll-trigger faults.
"""

import re

_OBSERVER_ACTION = re.compile(
    r"\b(verify|check|wait for|confirm|ensure)\b", re.I
)

# Console messages that look like application failures (generic patterns)
_SUBSTANTIVE = re.compile(
    r"500|404|403|502|503|Failed to load resource|net::ERR|API returned empty|"
    r"timeout|timed out|filmosections|firebaseio",
    re.I,
)
# Console noise that should not drive root-cause selection (generic web telemetry)
_NOISE = re.compile(
    r"Track&Report|uedata=|allow-scripts and allow-same-origin|analytics\.js|"
    r"cookie-consent|beacon|telemetry",
    re.I,
)


def is_observer_step(action: str, action_type: str) -> bool:
    """True for verification/wait steps that observe errors but rarely cause them."""
    if action_type in ("verify", "wait_for_load", "wait"):
        return True
    return bool(_OBSERVER_ACTION.search(action or ""))


def is_substantive_error(text: str) -> bool:
    if not text or _NOISE.search(text):
        return False
    return bool(_SUBSTANTIVE.search(text))


def is_noise_console(text: str) -> bool:
    return bool(text and _NOISE.search(text) and not _SUBSTANTIVE.search(text))


def _normalise(text: str) -> str:
    return re.sub(r"^\d{10,}:\s*", "", text).strip()


def _new_console_errors(pass_logs: list, fail_logs: list) -> list[dict]:
    def con_errors(logs: list) -> list:
        return [
            {"type": e.get("type"), "text": (e.get("text") or "")[:120]}
            for e in logs
            if e.get("type") in ("error", "severe", "warning")
        ]

    pass_texts = {_normalise(e["text"]) for e in con_errors(pass_logs)}
    return [e for e in con_errors(fail_logs) if _normalise(e["text"]) not in pass_texts]


def errors_on_next_observer_step(step: dict, next_step: dict | None) -> dict | None:
    """Return causal error bundle if next step is verify/wait and logged new substantive errors."""
    if not next_step:
        return None
    nxt_fail = next_step.get("fail_step") or {}
    nxt_action = nxt_fail.get("action") or ""
    nxt_type = nxt_fail.get("action_type") or ""

    if not is_observer_step(nxt_action, nxt_type):
        return None

    pass_logs = (next_step.get("pass_step") or {}).get("console_logs", [])
    fail_logs = nxt_fail.get("console_logs", [])
    new_con = _new_console_errors(pass_logs, fail_logs)
    substantive = [e for e in new_con if is_substantive_error(e.get("text", ""))]

    if not substantive:
        return None

    return {
        "next_step_id":   nxt_fail.get("step_id"),
        "next_action":    nxt_action,
        "console_errors": substantive,
    }


def find_causal_root_step(scored_steps: list[dict]) -> int | None:
    """
    Return step_id of the action step whose immediate next verify/wait step
    logged substantive new errors — e.g. imdb 14, hackernews 12.
    """
    by_id = {s["step_id"]: s for s in scored_steps}
    for step in scored_steps:
        sid = step["step_id"]
        action = (step.get("fail_step") or {}).get("action") or step.get("action", "")
        action_type = (step.get("fail_step") or {}).get("action_type") or ""
        if is_observer_step(action, action_type):
            continue
        if errors_on_next_observer_step(step, by_id.get(sid + 1)):
            return sid
    return None
