"""Local pass/fail screenshot difference scores (0–1)."""

import hashlib
from pathlib import Path


def _load_pair(pass_path: Path, fail_path: Path):
    from PIL import Image
    import numpy as np

    img_a = Image.open(pass_path).convert("RGB")
    img_b = Image.open(fail_path).convert("RGB")
    w, h = min(img_a.size[0], img_b.size[0]), min(img_a.size[1], img_b.size[1])
    a = np.array(img_a)
    b = np.array(img_b)
    return a, b, w, h


def pixel_diff_score(pass_path: str | None, fail_path: str | None) -> float:
    """Global center-crop mean diff (legacy heuristic pixel boost)."""
    details = pixel_diff_details(pass_path, fail_path)
    return details["global"]


def pixel_diff_details(pass_path: str | None, fail_path: str | None) -> dict:
    """
    Rich pass/fail screenshot comparison.

    Returns:
        identical: byte-identical files
        global:    center-crop mean diff (0–1)
        localized: max channel diff in form/content band (0–1) — catches single-field changes
        effective: max(global, localized) — use for earliest-divergence detection
    """
    empty = {"identical": False, "global": 0.0, "localized": 0.0, "effective": 0.0}
    if not pass_path or not fail_path:
        return empty
    p, f = Path(pass_path), Path(fail_path)
    if not p.exists() or not f.exists():
        return empty

    pb, fb = p.read_bytes(), f.read_bytes()
    if pb == fb:
        return {"identical": True, "global": 0.0, "localized": 0.0, "effective": 0.0}

    try:
        from PIL import Image
        import numpy as np

        a, b, w, h = _load_pair(p, f)

        # Global: center crop mean (existing behaviour)
        box = (int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85))
        ga = np.array(Image.fromarray(a).crop(box).resize((300, 300)))
        gb = np.array(Image.fromarray(b).crop(box).resize((300, 300)))
        global_score = min(1.0, float(np.abs(ga.astype(float) - gb.astype(float)).mean()) / 18.0)

        # Localized: max diff in form/content band (single empty field, button state)
        form_box = (int(w * 0.15), int(h * 0.25), int(w * 0.85), int(h * 0.55))
        fa = a[form_box[1]:form_box[3], form_box[0]:form_box[2]].astype(float)
        fb_ = b[form_box[1]:form_box[3], form_box[0]:form_box[2]].astype(float)
        localized = float(np.abs(fa - fb_).max()) / 255.0

        effective = max(global_score, localized)
        return {
            "identical": False,
            "global": round(global_score, 4),
            "localized": round(localized, 4),
            "effective": round(effective, 4),
        }
    except ImportError:
        ha, hb = hashlib.md5(pb).hexdigest(), hashlib.md5(fb).hexdigest()
        if ha != hb:
            ratio = abs(len(pb) - len(fb)) / max(len(pb), len(fb), 1)
            score = min(1.0, 0.4 + ratio * 0.6)
            return {"identical": False, "global": score, "localized": score, "effective": score}
        return empty
