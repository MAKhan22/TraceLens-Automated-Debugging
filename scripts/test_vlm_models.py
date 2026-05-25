#!/usr/bin/env python3
"""
Test VLM models on a single pass/fail screenshot pair (one API call each).

Usage:
    # List free vision models on OpenRouter
    python scripts/test_vlm_models.py --list-free

    # Test default free OpenRouter VLMs on wolfram step 12
    python scripts/test_vlm_models.py

    # Test specific models
    python scripts/test_vlm_models.py --models google/gemma-4-26b-a4b-it:free,qwen/qwen2.5-vl-72b-instruct

    # Groq (fast, free tier — uses GROQ_API_KEY)
    python scripts/test_vlm_models.py --provider groq
    python scripts/test_vlm_models.py --list-groq

    # Google AI Studio (Gemini) direct — uses GEMINI_API_KEY from .env
    python scripts/test_vlm_models.py --provider gemini --models gemini-2.0-flash

    # Custom trace/step
    python scripts/test_vlm_models.py --source efe_irem --trace saucedemo_2 --step 27

Notes:
    - qwen/qwen2.5-vl-7b-instruct is NOT on OpenRouter (only 72B paid).
    - qwen/qwen3-next-80b-a3b-instruct:free is text-only (no vision).
    - Each model gets ONE attempt (no retry loop) to avoid burning quota.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from src.screenshot_resolver import ScreenshotResolver

SINGLE_STEP_PROMPT = """You compare PASS vs FAIL browser screenshots for ONE test step.

Return ONLY valid JSON:
{{
  "visual_scores": [
    {{"step_id": {step_id}, "visual_score": <0.0-1.0>, "visual_note": "<short reason>"}}
  ],
  "visual_root_cause_step_id": {step_id},
  "visual_summary": "<one sentence>"
}}

Rules:
- visual_score 0.0 = identical, 1.0 = completely different UI state
- If FAIL shows wrong page, missing element, or different content → score >= 0.7

Step {step_id}: {action}
"""

PROVIDERS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "default_models": [
            "google/gemma-4-26b-a4b-it:free",
            "google/gemma-4-31b-it:free",
            "nvidia/nemotron-nano-12b-v2-vl:free",
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        ],
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "default_models": [
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
        ],
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "default_models": [
            "meta-llama/llama-4-scout-17b-16e-instruct",
        ],
    },
}

# Groq vision models (Groq /models endpoint may 403; maintain known list)
GROQ_VISION_MODELS = [
    {
        "id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "name": "Llama 4 Scout 17B — multimodal, free tier, fast",
    },
]

# Known unavailable / misleading IDs (documented for quick reference)
NOT_VLM = {
    "qwen/qwen2.5-vl-7b-instruct": "Not listed on OpenRouter (only qwen/qwen2.5-vl-72b-instruct)",
    "qwen/qwen3-next-80b-a3b-instruct:free": "Text-only on OpenRouter (no image input)",
}


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_free_vision_models(api_key: str) -> list[dict]:
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    data = json.loads(urllib.request.urlopen(req, timeout=30).read())
    out = []
    for m in data.get("data", []):
        mid = m.get("id", "")
        arch = m.get("architecture") or {}
        if "image" not in (arch.get("input_modalities") or []):
            continue
        if ":free" not in mid:
            continue
        out.append({"id": mid, "name": m.get("name", ""), "prompt": m.get("pricing", {}).get("prompt")})
    return sorted(out, key=lambda x: x["id"])


def encode_image(path: str) -> str:
    p = Path(path)
    data = p.read_bytes()
    ext = p.suffix.lower().lstrip(".") or "png"
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    return f"data:{mime};base64,{base64.standard_b64encode(data).decode()}"


def parse_json(text: str) -> dict:
    clean = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    return json.loads(clean)


def call_vlm_once(
    *,
    api_key: str,
    base_url: str,
    model: str,
    pass_path: str,
    fail_path: str,
    step_id: int,
    action: str,
    timeout: float = 120.0,
) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    prompt = SINGLE_STEP_PROMPT.format(step_id=step_id, action=action[:120])
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "text", "text": "Pass screenshot:"},
            {"type": "image_url", "image_url": {"url": encode_image(pass_path)}},
            {"type": "text", "text": "Fail screenshot:"},
            {"type": "image_url", "image_url": {"url": encode_image(fail_path)}},
        ],
    }]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )
    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        raise RuntimeError("empty response")
    return parse_json(raw)


def resolve_step(cfg: dict, source: str, trace: str, step_id: int) -> tuple[dict, str, str]:
    raw_base = cfg["data"]["raw_base"]
    src = cfg["data"]["sources"][source]
    trace_cfg = next(t for t in src["traces"] if t["id"] == trace)
    resolver = ScreenshotResolver(raw_base=raw_base)
    p, f = resolver.get_paths(source, src["base"], trace_cfg, step_id)
    if not p or not f:
        raise FileNotFoundError(f"No screenshots for {source}/{trace} step {step_id}")
    return trace_cfg, p, f


def test_model(
    provider: str,
    model: str,
    pass_path: str,
    fail_path: str,
    step_id: int,
    action: str,
) -> dict:
    if model in NOT_VLM:
        return {"model": model, "provider": provider, "status": "skip", "error": NOT_VLM[model]}

    prof = PROVIDERS[provider]
    api_key = os.environ.get(prof["api_key_env"], "")
    if not api_key:
        return {
            "model": model,
            "provider": provider,
            "status": "skip",
            "error": f"{prof['api_key_env']} not set in .env",
        }

    t0 = time.time()
    try:
        parsed = call_vlm_once(
            api_key=api_key,
            base_url=prof["base_url"],
            model=model,
            pass_path=pass_path,
            fail_path=fail_path,
            step_id=step_id,
            action=action,
        )
        scores = parsed.get("visual_scores") or []
        sc = float(scores[0].get("visual_score", 0)) if scores else 0.0
        note = scores[0].get("visual_note", "") if scores else ""
        return {
            "model": model,
            "provider": provider,
            "status": "ok",
            "latency_s": round(time.time() - t0, 2),
            "visual_score": sc,
            "visual_note": note[:120],
            "raw_summary": parsed.get("visual_summary", "")[:120],
        }
    except Exception as e:
        err = str(e)
        hint = ""
        if "429" in err and "openrouter" in provider:
            hint = "OpenRouter free daily quota — add credits or wait for reset"
        elif "429" in err and provider == "gemini":
            hint = "Gemini free tier quota — check https://ai.dev/rate-limit or enable billing"
        elif "429" in err and provider == "groq":
            hint = "Groq rate limit — check console.groq.com usage (100k tokens/day on free tier)"
        elif "402" in err:
            hint = "OpenRouter credits required for this model"
        return {
            "model": model,
            "provider": provider,
            "status": "fail",
            "latency_s": round(time.time() - t0, 2),
            "error": err[:400],
            "hint": hint,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test VLM models on one screenshot pair")
    parser.add_argument("--list-free", action="store_true", help="List free vision models on OpenRouter")
    parser.add_argument("--list-groq", action="store_true", help="List known Groq vision models")
    parser.add_argument("--provider", choices=list(PROVIDERS.keys()), default="openrouter")
    parser.add_argument("--models", default=None, help="Comma-separated model IDs")
    parser.add_argument("--source", default="efe_irem")
    parser.add_argument("--trace", default="wolfram")
    parser.add_argument("--step", type=int, default=12)
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between model tests")
    args = parser.parse_args()

    if args.list_groq:
        print("Known Groq vision models:\n")
        for m in GROQ_VISION_MODELS:
            print(f"  {m['id']:<55} {m['name']}")
        print("\nRequires GROQ_API_KEY in .env — get one at https://console.groq.com/keys")
        print("Test: python scripts/test_vlm_models.py --provider groq")
        return

    if args.list_free:
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            print("ERROR: OPENROUTER_API_KEY not set")
            sys.exit(1)
        print("Free vision models on OpenRouter:\n")
        for m in fetch_free_vision_models(key):
            print(f"  {m['id']:<55} {m['name']}")
        print("\nNot on OpenRouter:")
        for mid, reason in NOT_VLM.items():
            print(f"  {mid:<55} {reason}")
        return

    cfg = load_config()
    trace_cfg, pass_path, fail_path = resolve_step(cfg, args.source, args.trace, args.step)
    action = f"Test step {args.step} ({args.trace})"

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = list(PROVIDERS[args.provider]["default_models"])

    print(f"Testing {len(models)} model(s) on {args.source}/{args.trace} step {args.step}")
    print(f"Provider: {args.provider}")
    print(f"Pass: {pass_path}")
    print(f"Fail: {fail_path}\n")

    results = []
    for i, model in enumerate(models):
        if i > 0 and args.delay > 0:
            time.sleep(args.delay)
        print(f"--- {model} ---")
        r = test_model(args.provider, model, pass_path, fail_path, args.step, action)
        results.append(r)
        if r["status"] == "ok":
            print(f"  OK  {r['latency_s']}s  score={r['visual_score']:.2f}  {r['visual_note']}")
        elif r["status"] == "skip":
            print(f"  SKIP  {r['error']}")
        else:
            print(f"  FAIL  {r['latency_s']}s  {r['error'][:200]}")
            if r.get("hint"):
                print(f"        Hint: {r['hint']}")

    ok = [r for r in results if r["status"] == "ok"]
    print("\n" + "=" * 60)
    print(f"Summary: {len(ok)}/{len(results)} succeeded")
    if ok:
        best = max(ok, key=lambda x: x["visual_score"])
        print(f"Best score: {best['model']} → {best['visual_score']:.2f}")
        print("\nTo use in config.yaml (vlm section):")
        prof = PROVIDERS[args.provider]
        print(f"  base_url:    \"{prof['base_url']}\"")
        print(f"  vlm_model:   \"{best['model']}\"")
        print(f"  api_key_env: \"{prof['api_key_env']}\"")


if __name__ == "__main__":
    main()
