"""
trace_parser.py
---------------
Parses the three raw data formats into a single unified step schema:

Unified step:
{
    "step_id":      int,            # 0-indexed
    "action":       str,            # human-readable instruction
    "action_type":  str,            # e.g. "click", "navigate", "type", "verify"
    "network_logs": [               # list of request objects
        {"url": str, "status": int, "method": str, "error": str|None}
    ],
    "console_logs": [               # list of log entries
        {"type": str, "text": str}  # type: "error"|"warning"|"info"|"severe"
    ],
    "intent":       str|None,       # ersel only: "Verification passed/failed: ..."
    "screenshot_before": str|None,  # relative path
    "screenshot_after":  str|None,
}

Supported source formats:
  "efe_irem"    – JSON with embedded per-step logs
  "areeb_salem" – JSON with path references + separate .txt log files
  "ersel"       – steps.json + global_console/network_logs.json
"""

import json
import re
from pathlib import Path


# ── unified schema helper ─────────────────────────────────────────────────────

def _make_step(
    step_id: int,
    action: str,
    action_type: str = "unknown",
    network_logs: list | None = None,
    console_logs: list | None = None,
    intent: str | None = None,
    screenshot_before: str | None = None,
    screenshot_after: str | None = None,
) -> dict:
    return {
        "step_id": step_id,
        "action": action,
        "action_type": action_type,
        "network_logs": network_logs or [],
        "console_logs": console_logs or [],
        "intent": intent,
        "screenshot_before": screenshot_before,
        "screenshot_after": screenshot_after,
    }


# ── FORMAT 1: Efe-Irem ─────────────────────────────────────────────────────────
# wikipedia_correct.json / wikipedia_incorrect.json
# Each step already has console_log and network_log arrays embedded.

def _parse_network_log_efe(entry: dict) -> dict:
    return {
        "url":    entry.get("url", ""),
        "status": entry.get("status"),
        "method": entry.get("method", "GET"),
        "error":  entry.get("error"),
    }


def _parse_console_log_efe(entry: dict) -> dict:
    raw_type = entry.get("type", "info").lower()
    # normalise to: error / warning / info / severe
    if raw_type in ("error", "severe"):
        t = "error"
    elif raw_type == "warning":
        t = "warning"
    else:
        t = "info"
    return {"type": t, "text": entry.get("text", "")}


def parse_efe_irem(json_path: str) -> list[dict]:
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    steps = []
    for i, raw_step in enumerate(raw):
        step_id_raw = raw_step.get("step_id", f"step_{i:03d}")
        # convert "step_001" → 0, "step_002" → 1, etc.
        if isinstance(step_id_raw, str):
            digits = re.findall(r"\d+", step_id_raw)
            step_id = int(digits[0]) - 1 if digits else i
        else:
            step_id = int(step_id_raw)

        steps.append(_make_step(
            step_id=step_id,
            action=raw_step.get("step_description", ""),
            action_type=raw_step.get("action_type", "unknown"),
            network_logs=[_parse_network_log_efe(e) for e in raw_step.get("network_log", [])],
            console_logs=[_parse_console_log_efe(e) for e in raw_step.get("console_log", [])],
            screenshot_before=raw_step.get("screenshot_before"),
            screenshot_after=raw_step.get("screenshot_after"),
        ))
    return steps


# ── FORMAT 2: Areeb & Salem ───────────────────────────────────────────────────
# trace_fail.json / trace_pass.json
# stepDetails[i] has paths to console .txt and screenshots.
# Network logs live in network/network_step_NN.txt (not referenced in JSON).

def _parse_console_txt(txt_path: str) -> list[dict]:
    """Parse a console_step_NN.txt file into a list of log dicts."""
    path = Path(txt_path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    logs = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # match patterns like [SEVERE] ..., [WARNING] ..., [INFO] ...
        m = re.match(r"^\[(SEVERE|WARNING|INFO|ERROR)\]\s*(.*)", line, re.IGNORECASE)
        if m:
            raw_type = m.group(1).upper()
            text_body = m.group(2)
            t = "error" if raw_type in ("SEVERE", "ERROR") else raw_type.lower()
            logs.append({"type": t, "text": text_body})
        elif line.startswith("=== ") or line.startswith("Timestamp:"):
            continue  # header lines
        else:
            # generic line — treat as info
            logs.append({"type": "info", "text": line})
    return logs


def _parse_network_txt(txt_path: str) -> list[dict]:
    """Parse a network_step_NN.txt file into a list of request dicts."""
    path = Path(txt_path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    logs = []
    # each entry looks like:
    # [GET] https://...
    #   Status: 200
    #   Time: ...
    #   ────────────
    current: dict | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("===") or line.startswith("Timestamp") or set(line) == {"─"}:
            if current:
                logs.append(current)
                current = None
            continue
        m_req = re.match(r"^\[(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\]\s+(.+)", line)
        if m_req:
            if current:
                logs.append(current)
            current = {"url": m_req.group(2), "status": None, "method": m_req.group(1), "error": None}
            continue
        if current:
            m_status = re.match(r"^Status:\s*(\d+)", line)
            if m_status:
                current["status"] = int(m_status.group(1))
            m_err = re.match(r"^Error:\s*(.+)", line, re.IGNORECASE)
            if m_err:
                current["error"] = m_err.group(1)
    if current:
        logs.append(current)
    return logs


def parse_areeb_salem(json_path: str) -> list[dict]:
    json_path = Path(json_path)
    base_dir = json_path.parent

    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    step_details = raw.get("stepDetails", raw) if isinstance(raw, dict) else raw
    steps = []

    for i, entry in enumerate(step_details):
        step_info = entry.get("step", {})
        action = step_info.get("stepInstruction", "")
        action_type = step_info.get("action", "unknown")
        screenshot = entry.get("screenshot")
        console_ref = entry.get("consoleLog")

        # resolve console logs
        console_logs = []
        if console_ref:
            console_txt = base_dir / console_ref
            console_logs = _parse_console_txt(str(console_txt))

        # network logs: infer path from console path pattern
        network_logs = []
        if console_ref:
            net_ref = console_ref.replace("console/console_", "network/network_")
            network_txt = base_dir / net_ref
            network_logs = _parse_network_txt(str(network_txt))

        # screenshot_after only (areeb traces only have one screenshot per step)
        ss = str(base_dir / screenshot) if screenshot else None

        steps.append(_make_step(
            step_id=i,
            action=action,
            action_type=action_type,
            network_logs=network_logs,
            console_logs=console_logs,
            screenshot_after=ss,
        ))
    return steps


# ── FORMAT 3: Ersel-Ikra-Merve ────────────────────────────────────────────────
# steps.json  – per-step actions + optional intent field
# global_console_logs.json – flat list of strings (NOT per step)
# global_network_logs.json – flat list of strings (NOT per step)
# Screenshots: step_N_prev.png / step_N_post.png

def _parse_global_console_ersel(entries: list[str]) -> list[dict]:
    logs = []
    for line in entries:
        line = str(line).strip()
        if not line:
            continue
        upper = line.upper()
        if "SEVERE" in upper or "FAILED TO LOAD" in upper or "ERROR" in upper:
            t = "error"
        elif "WARNING" in upper or "WARN" in upper:
            t = "warning"
        else:
            t = "info"
        logs.append({"type": t, "text": line})
    return logs


def _parse_global_network_ersel(entries: list[str]) -> list[dict]:
    """
    Format: "GET /en-gb/catalog/desktops - 200 OK"
             "POST /en-gb?route=checkout/cart.add - 200 OK"
    """
    logs = []
    for line in entries:
        line = str(line).strip()
        if not line:
            continue
        m = re.match(r"^(GET|POST|PUT|DELETE|PATCH)\s+(\S+)\s+-\s+(\d+)", line)
        if m:
            logs.append({
                "url":    m.group(2),
                "status": int(m.group(3)),
                "method": m.group(1),
                "error":  None,
            })
        else:
            logs.append({"url": line, "status": None, "method": "UNKNOWN", "error": None})
    return logs


def parse_ersel(trace_dir: str) -> list[dict]:
    """
    trace_dir should be either the 'passing' or 'failing' subfolder,
    e.g. ersel-ikra-merve traces/opencart_purchase_40/failing/
    """
    trace_dir = Path(trace_dir)

    with open(trace_dir / "steps.json", encoding="utf-8") as f:
        raw_steps = json.load(f)

    # global logs (not per-step) — attach to step 0 as a summary marker
    global_console: list[dict] = []
    global_network: list[dict] = []

    console_path = trace_dir / "global_console_logs.json"
    network_path = trace_dir / "global_network_logs.json"

    if console_path.exists():
        with open(console_path, encoding="utf-8") as f:
            global_console = _parse_global_console_ersel(json.load(f))

    if network_path.exists():
        with open(network_path, encoding="utf-8") as f:
            global_network = _parse_global_network_ersel(json.load(f))

    steps = []
    for raw_step in raw_steps:
        idx = raw_step.get("step_idx", len(steps))
        action = raw_step.get("stepInstruction", "")
        action_type = raw_step.get("action", "unknown")
        intent = raw_step.get("intent")

        # screenshots
        ss_before = str(trace_dir / f"step_{idx}_prev.png")
        ss_after  = str(trace_dir / f"step_{idx}_post.png")
        ss_before = ss_before if Path(ss_before).exists() else None
        ss_after  = ss_after  if Path(ss_after).exists()  else None

        # Global logs are session-wide and NOT per-step in ersel traces.
        # Attaching them to individual steps would inflate scores incorrectly.
        # They are stored on the returned list as a metadata attribute below.
        steps.append(_make_step(
            step_id=idx,
            action=action,
            action_type=action_type,
            network_logs=[],
            console_logs=[],
            intent=intent,
            screenshot_before=ss_before,
            screenshot_after=ss_after,
        ))

    return steps


# ── Public API ────────────────────────────────────────────────────────────────

def parse_trace(source: str, path: str) -> list[dict]:
    """
    Parse a trace from any supported format.

    Args:
        source: one of "efe_irem", "areeb_salem", "ersel"
        path:   path to the JSON file (efe_irem / areeb_salem)
                OR path to the trace folder (ersel)

    Returns:
        list of unified step dicts
    """
    if source == "efe_irem":
        return parse_efe_irem(path)
    elif source == "areeb_salem":
        return parse_areeb_salem(path)
    elif source == "ersel":
        return parse_ersel(path)
    else:
        raise ValueError(f"Unknown source format: {source!r}. Use 'efe_irem', 'areeb_salem', or 'ersel'.")


def save_processed(steps: list[dict], output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(steps, f, indent=2, ensure_ascii=False)
