"""
ranker.py
---------
Takes scored steps from anomaly_detector and produces a ranked list.

Two ranking modes:
  "heuristic"  – pure score sort (fast, no LLM needed)
  "llm"        – all steps (or heuristic top-K if pre_llm_k > 0) go to LLM for re-ranking

The heuristic rank is always computed first.
LLM re-ranking is done in llm_reasoner.py and passed back here for final output.
"""

import json
from pathlib import Path


class Ranker:
    def __init__(self, top_k: int = 5, pre_llm_k: int = 10):
        self.top_k = top_k          # final output size
        self.pre_llm_k = pre_llm_k  # candidates sent to LLM for re-ranking

    def rank_heuristic(self, scored_steps: list[dict]) -> list[dict]:
        """Sort by combined_score descending, return top_k."""
        ranked = sorted(scored_steps, key=lambda x: x["combined_score"], reverse=True)
        return ranked[:self.top_k]

    def candidates_for_llm(self, scored_steps: list[dict]) -> list[dict]:
        """Return candidates to send to the LLM for re-ranking.
        If pre_llm_k is 0, returns all steps so the LLM is not bottlenecked
        by the heuristic ranking (i.e. can discover faults ranked low by heuristic).
        """
        ranked = sorted(scored_steps, key=lambda x: x["combined_score"], reverse=True)
        if self.pre_llm_k == 0:
            return ranked  # all steps
        return ranked[:self.pre_llm_k]

    def apply_llm_reranking(self, llm_order: list[int],
                             scored_steps: list[dict]) -> list[dict]:
        """
        Apply LLM-provided re-ranking.

        Args:
            llm_order:    list of step_ids in LLM-preferred order
            scored_steps: original scored list

        Returns:
            re-ranked list (top_k)
        """
        step_map = {s["step_id"]: s for s in scored_steps}
        reranked = []
        for step_id in llm_order:
            if step_id in step_map:
                reranked.append(step_map[step_id])
        # append any not mentioned by LLM, sorted by score
        mentioned = set(llm_order)
        rest = sorted(
            [s for s in scored_steps if s["step_id"] not in mentioned],
            key=lambda x: x["combined_score"], reverse=True
        )
        reranked.extend(rest)
        return reranked[:self.top_k]

    def add_rank_metadata(self, ranked: list[dict]) -> list[dict]:
        """
        Return a new list where each entry is a shallow copy with rank added.
        Does NOT mutate the original dicts — important because scored/heuristic/llm
        lists all share the same underlying dict objects.
        """
        return [{**step, "rank": i + 1} for i, step in enumerate(ranked)]

    def save(self, ranked: list[dict], output_path: str) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # strip bulky pass_step/fail_step from saved output
        slim = []
        for s in ranked:
            slim.append({k: v for k, v in s.items()
                         if k not in ("pass_step", "fail_step")})
        with open(out, "w", encoding="utf-8") as f:
            json.dump(slim, f, indent=2)
