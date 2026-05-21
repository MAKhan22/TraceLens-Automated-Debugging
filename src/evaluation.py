"""
evaluation.py
-------------
Computes all evaluation metrics for a ranked suspicious step list vs ground truth.

Metrics per trace:
  hit@k             – was the actual fault step in the top-k?
  rank_position     – what rank did the actual fault step appear at? (-1 = not in top-k)
  rank_distance     – |rank_position - 1|  (how far from the top of the list)
  step_distances    – for each of the top-5 predicted steps: |predicted_step_id - actual_step_id|

Aggregate across all traces:
  hit@1, hit@3, hit@5
  mean_rank_distance    (lower = better)
  mean_step_distance@1  (distance of top prediction from actual fault)
  MAD@5                 (mean absolute step distance across top-5)
"""

import json
from pathlib import Path


class Evaluator:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    # ── per-trace metrics ─────────────────────────────────────────────────────

    def hit_at_k(self, ranked: list[dict], actual: int, k: int | None = None) -> int:
        k = k or self.top_k
        predicted = [s["step_id"] for s in ranked[:k]]
        return int(actual in predicted)

    def rank_position(self, ranked: list[dict], actual: int) -> int:
        """1-based rank of actual fault step. -1 if not in ranked list."""
        for i, s in enumerate(ranked):
            if s["step_id"] == actual:
                return i + 1
        return -1

    def rank_distance(self, ranked: list[dict], actual: int) -> int:
        """
        How far is the actual fault step from rank 1?
        rank_position=1 → distance=0
        rank_position=3 → distance=2
        not found       → top_k (worst case)
        """
        pos = self.rank_position(ranked, actual)
        if pos == -1:
            return self.top_k
        return pos - 1

    def step_distances(self, ranked: list[dict], actual: int) -> list[dict]:
        """
        For each of the top-k predictions: how many steps away from the actual fault?
        Returns list of dicts with rank, step_id, step_distance.
        """
        results = []
        for step in ranked[:self.top_k]:
            results.append({
                "rank":          step.get("rank", ranked.index(step) + 1),
                "step_id":       step["step_id"],
                "step_distance": abs(step["step_id"] - actual),
            })
        return results

    def evaluate_trace(self, ranked: list[dict], actual: int) -> dict:
        """Full per-trace evaluation."""
        return {
            "actual_fault_step": actual,
            "hit@1":              self.hit_at_k(ranked, actual, k=1),
            "hit@3":              self.hit_at_k(ranked, actual, k=3),
            "hit@5":              self.hit_at_k(ranked, actual, k=5),
            "rank_position":      self.rank_position(ranked, actual),
            "rank_distance":      self.rank_distance(ranked, actual),
            "step_distances":     self.step_distances(ranked, actual),
            "mad@5":              self._mad(ranked, actual),
            "top1_step_distance": abs(ranked[0]["step_id"] - actual) if ranked else None,
        }

    def _mad(self, ranked: list[dict], actual: int) -> float:
        """Mean absolute step distance across top-k predictions."""
        if not ranked:
            return float("inf")
        dists = [abs(s["step_id"] - actual) for s in ranked[:self.top_k]]
        return sum(dists) / len(dists)

    # ── aggregate metrics across all traces ───────────────────────────────────

    def aggregate(self, per_trace_results: list[dict]) -> dict:
        """
        Aggregate evaluation results across a collection of traces.

        Args:
            per_trace_results: list of dicts returned by evaluate_trace(),
                               each optionally tagged with "trace_id"
        """
        n = len(per_trace_results)
        if n == 0:
            return {}

        def mean(key):
            vals = [r[key] for r in per_trace_results if r.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        # for rank_position: exclude -1 (not found) from mean
        valid_ranks = [r["rank_position"] for r in per_trace_results if r.get("rank_position", -1) > 0]
        mean_rank = round(sum(valid_ranks) / len(valid_ranks), 4) if valid_ranks else None

        return {
            "n_traces":               n,
            "hit@1":                  mean("hit@1"),
            "hit@3":                  mean("hit@3"),
            "hit@5":                  mean("hit@5"),
            "mean_rank_position":     mean_rank,
            "mean_rank_distance":     mean("rank_distance"),
            "mean_top1_step_distance":mean("top1_step_distance"),
            "mean_mad@5":             mean("mad@5"),
            "found_in_top5_count":    sum(r["hit@5"] for r in per_trace_results),
        }

    # ── I/O helpers ───────────────────────────────────────────────────────────

    def save(self, results: dict | list, output_path: str) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
