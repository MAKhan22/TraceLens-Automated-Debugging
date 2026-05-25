"""
screenshot_resolver.py
----------------------
Resolves screenshot file paths for each step across all three data sources.

Each source uses a different naming convention:
  efe_irem   : {trace_dir}/screenshots/correct/step_001_after.png  (1-indexed, 3-digit zero-padded)
  areeb_salem: {trace_dir}/pass/screenshots/step_00.png             (0-indexed, 2-digit zero-padded)
  ersel      : {trace_dir}/passing/step_0_post.png                  (0-indexed, no padding)
"""

from pathlib import Path


def valid_screenshot_pair(
    pass_path: str | None, fail_path: str | None
) -> bool:
    """True when both paths refer to existing regular image files."""
    if not pass_path or not fail_path:
        return False
    p, f = Path(pass_path), Path(fail_path)
    return p.is_file() and f.is_file()


def filter_steps_with_screenshot_pairs(steps: list[dict]) -> list[dict]:
    """Keep only steps that have both pass and fail screenshot files on disk."""
    return [
        s for s in steps
        if valid_screenshot_pair(s.get("pass_screenshot"), s.get("fail_screenshot"))
    ]


class ScreenshotResolver:
    """Resolves pass/fail screenshot paths for a given source, trace, and step."""

    def __init__(self, raw_base: str):
        self.raw_base = raw_base

    def get_paths(
        self,
        source: str,
        source_base: str,
        trace_cfg: dict,
        step_id: int,
    ) -> tuple[str | None, str | None]:
        """
        Return (pass_screenshot_path, fail_screenshot_path) for a step.
        Returns (None, None) if screenshots don't exist or source is unknown.

        Args:
            source:      'efe_irem', 'areeb_salem', or 'ersel'
            source_base: value of sources.<source>.base from config (e.g. 'efe- irem traces')
            trace_cfg:   the trace entry dict with 'id', 'pass', 'fail' keys
            step_id:     0-indexed step number
        """
        base = Path(self.raw_base) / source_base
        trace_dir = trace_cfg["pass"].split("/")[0]

        if source == "efe_irem":
            # Screenshots are 1-indexed and 3-digit zero-padded.
            # Two sub-directory layouts exist across different traces:
            #   Layout A: screenshots/correct/step_001_after.png   (saucedemo, gutenberg, elinguistics)
            #   Layout B: correct screenshots/step_001_after.png   (dictionary, webmd, wikipedia, wolfram)
            num = step_id + 1
            pass_a = base / trace_dir / "screenshots" / "correct"   / f"step_{num:03d}_after.png"
            fail_a = base / trace_dir / "screenshots" / "incorrect" / f"step_{num:03d}_after.png"
            pass_b = base / trace_dir / "correct screenshots"   / f"step_{num:03d}_after.png"
            fail_b = base / trace_dir / "incorrect screenshots" / f"step_{num:03d}_after.png"
            # Prefer Layout A; fall back to Layout B
            pass_path = pass_a if pass_a.exists() else pass_b
            fail_path = fail_a if fail_a.exists() else fail_b

        elif source == "areeb_salem":
            # Screenshots are 0-indexed and 2-digit zero-padded
            pass_path = base / trace_dir / "pass" / "screenshots" / f"step_{step_id:02d}.png"
            fail_path = base / trace_dir / "fail" / "screenshots" / f"step_{step_id:02d}.png"

        elif source == "ersel":
            # Screenshots are 0-indexed with no padding, named step_X_post.png
            pass_path = base / trace_dir / "passing" / f"step_{step_id}_post.png"
            fail_path = base / trace_dir / "failing" / f"step_{step_id}_post.png"

        else:
            return None, None

        pass_str = str(pass_path) if pass_path.exists() else None
        fail_str = str(fail_path) if fail_path.exists() else None
        return pass_str, fail_str

    def attach_screenshots(
        self,
        scored_steps: list[dict],
        source: str,
        source_base: str,
        trace_cfg: dict,
    ) -> list[dict]:
        """
        Return a new list of step dicts with 'pass_screenshot' and 'fail_screenshot'
        fields populated (paths or None).
        """
        result = []
        for step in scored_steps:
            sid = step.get("step_id", 0)
            pass_path, fail_path = self.get_paths(source, source_base, trace_cfg, sid)
            result.append({
                **step,
                "pass_screenshot": pass_path,
                "fail_screenshot":  fail_path,
            })
        return result
