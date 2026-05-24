"""
navigation_signals.py
---------------------
Detect wrong-page navigation from pass/fail network URL diffs.

Used by both anomaly_detector (heuristic boost) and llm_reasoner (LLM input).
"""

import re

_ASSET_EXT = re.compile(r"\.(js|css|png|jpg|jpeg|gif|svg|woff2?|ico|map)(\?|$)", re.I)
_NOISE_PATH = re.compile(
    r"/(analytics|consent|fonts|common/javascript|stripe|tracking|wal/)", re.I
)
_ACTION_STOP = frozenset({
    "click", "wait", "for", "the", "page", "hit", "enter", "type", "verify",
    "check", "on", "to", "and", "with", "from", "into", "topic", "card",
    "button", "link", "through", "scroll", "press", "select", "open",
})


def _is_page_url(url: str) -> bool:
    """True for HTML routes / topic pages, false for assets and telemetry."""
    if not url:
        return False
    if _ASSET_EXT.search(url):
        return False
    if _NOISE_PATH.search(url):
        return False
    if url.endswith(".html") or "/topics/" in url:
        return True
    path = url.split("?")[0].split("#")[0]
    return bool(re.search(r"/[A-Za-z][A-Za-z0-9_-]+\.html", path))


def _action_tokens(action: str) -> list[str]:
    if not action:
        return []
    words = re.findall(r"[A-Za-z]{3,}", action)
    return [w for w in words if w.lower() not in _ACTION_STOP]


def _url_matches_tokens(url: str, tokens: list[str]) -> bool:
    ul = url.lower()
    for token in tokens:
        tl = token.lower()
        if tl in ul:
            return True
        parts = re.findall(r"[A-Z]?[a-z]+", token)
        if len(parts) >= 2 and all(p.lower() in ul for p in parts):
            return True
    return False


def compute_navigation_signals(
    pass_net_logs: list,
    fail_net_logs: list,
    action: str,
) -> dict:
    """
    Compare pass vs fail network URLs to detect wrong navigation.

    Returns:
        missing_expected_pages: page URLs expected from action, present in pass, absent in fail
        wrong_pages_loaded:     page URLs loaded in fail but not pass, NOT matching the action
        navigation_mismatch:    expected page missing AND a different page loaded instead
        wrong_navigation:       same as navigation_mismatch (alias for clarity in prompts)
    """
    pass_urls = {e.get("url", "") for e in pass_net_logs if e.get("url")}
    fail_urls = {e.get("url", "") for e in fail_net_logs if e.get("url")}

    missing = pass_urls - fail_urls
    new_in_fail = fail_urls - pass_urls

    missing_pages = [u for u in missing if _is_page_url(u)]
    new_pages = [u for u in new_in_fail if _is_page_url(u)]

    tokens = _action_tokens(action)
    expected_missing = [u for u in missing_pages if _url_matches_tokens(u, tokens)]
    unexpected_loaded = [u for u in new_pages if not _url_matches_tokens(u, tokens)]

    # Fallback: if action tokens didn't match, still flag clear page swaps
    if not expected_missing and missing_pages and new_pages:
        expected_missing = missing_pages[:1]
    if not unexpected_loaded and new_pages:
        unexpected_loaded = [u for u in new_pages if u not in missing_pages]

    mismatch = bool(expected_missing and unexpected_loaded)

    return {
        "missing_expected_pages": expected_missing[:3],
        "wrong_pages_loaded":     unexpected_loaded[:3],
        "navigation_mismatch":    mismatch,
        "wrong_navigation":       mismatch,
    }
