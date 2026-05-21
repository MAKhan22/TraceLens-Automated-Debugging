"""
trace_aligner.py
----------------
Aligns a passing trace with a failing trace into a list of step pairs.

Primary strategy: align by step index (both traces share the same step count
by design across all three data sources).

Fallback: when counts differ, use SequenceMatcher on action strings to find
the best alignment and flag unmatched steps.

Output per pair:
{
    "pass_step": {...unified step...},
    "fail_step": {...unified step...},
    "aligned":   True | False   # False = only one side has this step
}
"""

from difflib import SequenceMatcher


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def align_by_index(passing: list[dict], failing: list[dict]) -> list[dict]:
    """Simple zip — works when both traces have the same number of steps."""
    pairs = []
    min_len = min(len(passing), len(failing))
    for i in range(min_len):
        pairs.append({
            "pass_step": passing[i],
            "fail_step": failing[i],
            "aligned": True,
        })
    # append unmatched tail steps (if lengths differ)
    for step in passing[min_len:]:
        pairs.append({"pass_step": step, "fail_step": None, "aligned": False})
    for step in failing[min_len:]:
        pairs.append({"pass_step": None, "fail_step": step, "aligned": False})
    return pairs


def align_by_similarity(passing: list[dict], failing: list[dict],
                         threshold: float = 0.6) -> list[dict]:
    """
    Greedy similarity alignment — used when step counts differ significantly.
    Tries to match each failing step to the closest unmatched passing step.
    """
    used_pass = set()
    pairs = []

    for f_step in failing:
        best_idx, best_score = -1, 0.0
        for j, p_step in enumerate(passing):
            if j in used_pass:
                continue
            score = _similarity(
                f_step.get("action", ""),
                p_step.get("action", ""),
            )
            if score > best_score:
                best_score, best_idx = score, j

        if best_score >= threshold and best_idx >= 0:
            used_pass.add(best_idx)
            pairs.append({
                "pass_step": passing[best_idx],
                "fail_step": f_step,
                "aligned": True,
            })
        else:
            pairs.append({"pass_step": None, "fail_step": f_step, "aligned": False})

    for j, p_step in enumerate(passing):
        if j not in used_pass:
            pairs.append({"pass_step": p_step, "fail_step": None, "aligned": False})

    # sort by fail_step step_id so output is sequential
    pairs.sort(key=lambda p: (p["fail_step"] or p["pass_step"] or {}).get("step_id", 0))
    return pairs


def align(passing: list[dict], failing: list[dict],
          force_similarity: bool = False) -> list[dict]:
    """
    Main alignment entry point.

    Uses index alignment when step counts match (or differ by ≤ 2),
    falls back to similarity alignment otherwise.
    """
    diff = abs(len(passing) - len(failing))
    if not force_similarity and diff <= 2:
        return align_by_index(passing, failing)
    return align_by_similarity(passing, failing)
