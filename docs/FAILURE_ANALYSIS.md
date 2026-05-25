# TraceLens — Failure Case Analysis

This document categorizes the 7 traces that scored Hit@5=0 in the heuristic-only baseline run, explains why each fails, and documents the fixes applied (or limitations acknowledged).

**Recommended commands:** Text fixes → `python main.py --llm`. VLM-dependent traces → `python main.py --llm --vlm`. Heuristic baseline → `python main.py` (no flags).

---

## Summary Table

| Trace | Fault Type | Signal | Fixable w/ Text? | Root Problem |
|-------|-----------|--------|-----------------|--------------|
| `efe_irem/saucedemo_2` | missing_input_zipcode | screenshot_only | No — Phase 2 (VLM) | No network/console/action text signal exists |
| `efe_irem/wolfram` | wrong_element_clicked | absence_of_activity | **Yes** | Heuristic ignores *missing* requests; LLM can reason about it |
| `efe_irem/wikipedia` | broken_anchor_toc | network_noise_distractor | No — text-invisible | Step 13 has zero signal; early steps have synthetic cross-domain noise |
| `areeb_salem/hackernews` | firebase_timeout | console_and_network | **Yes** | Heuristic ranks symptom (step 13) over root cause (step 12) |
| `areeb_salem/imdb` | api_500_filmography | network | **Yes** | Adjacent noisy steps outscore the actual fault step |
| `areeb_salem/npm` | network_error_registry | network | **Yes** | Same as imdb — nearby noise outscores the fault |
| `areeb_salem/pypi` | captcha_bot_challenge | screenshot | No — Phase 2 (VLM) | CAPTCHA page loads silently; no text signal |

**4 out of 7 failures are text-fixable. 3 are text-invisible (VLM or environmental noise).**

---

## Case-by-Case Breakdown

### 1. `efe_irem/saucedemo_2` — VLM only ❌

**Fault**: The zipcode field in the checkout form was left empty in the failing trace.  
**Why heuristic fails**: Step 27 has no HTTP errors, no console errors. The action text may be identical between pass and fail (both say "fill checkout form"). Score = 0.0.  
**Why LLM can't fix it**: The LLM only sees text. If the action description doesn't differ and no network/console signal exists, there is simply no information available.  
**Fix**: Phase 2 VLM — compare screenshots of the form to detect the missing field visually.

---

### 2. `efe_irem/wolfram` — Fixable ✅

**Fault**: The wrong math element was clicked at step 12, causing navigation to the wrong page. The correct page never loaded.  
**Why heuristic fails**: Step 12 has `action_score > 0` (action divergence) but scores `network_score = 0` and `console_score = 0`. Steps 4, 14, 15, 16, 17 have moderate network scores because they all made requests to wolfram CDN resources. Step 12 gets buried.  
**Root problem**: The fail run loaded a **different page** (`RecreationalMathematics.html`) instead of the expected `Geometry.html`. The old pipeline only surfaced `missing_requests` (expected page absent) but not `wrong_pages_loaded` (wrong page present). The LLM then misread step 4's `fail_request_count=0` + missing font/analytics URLs as the root cause.  
**Fix applied**:
- `navigation_signals.py` detects `wrong_navigation=true` when expected page missing AND wrong page loaded
- `_slim_step_for_llm` exposes `missing_expected_pages`, `wrong_pages_loaded`, `wrong_navigation`
- Heuristic network score boosted to 0.95 for wrong-navigation steps
- `diagnosis_candidates()` injects wrong-navigation steps into LLM diagnosis even if reranker missed them
- Prompts prioritize `wrong_navigation` over `fail_request_count=0` asset gaps

---

### 3. `efe_irem/wikipedia` — Text-invisible ❌

**Fault**: Broken anchor link at step 13. The `#Etymology` TOC anchor was renamed in the failing page version, so clicking it silently does nothing.  
**Why text-only fails**:
- Step 13: `action_changed=False` (same action text both runs), `pass_request_count=0`, `fail_request_count=0`, zero console errors — **completely invisible to text analysis**. Anchor navigation is purely browser-side JS with no network request and no error thrown.
- Steps 0 and 2: Have `new_network_errors` pointing to `https://example.com/assets/logo.png` (500) and `https://example.com/api/tracking` (404). These are **synthetic placeholder URLs** in the trace dataset from a completely different domain than the test subject (Wikipedia). The LLM correctly flags them as suspicious because they are new errors, but they are unrelated environmental noise.
**Root problem**: The actual fault produces zero observable signal in any text dimension. The synthetic errors from unrelated domains mislead both heuristic and LLM.  
**Fix possible?**: Only with one of:
1. VLM — screenshot would show the page didn't scroll to the expected section
2. Explicit domain filtering — if we tell the LLM "errors on domains other than the test target are noise" it could ignore steps 0/2 (but still can't find step 13 with no signal)
3. Sequence context — if subsequent steps fail to find expected elements after the TOC click, that contextual chain would implicate step 13 — **this is the most promising text-based approach**

---

### 4. `areeb_salem/hackernews` — Fixable ✅

**Fault**: Firebase async call times out at step 12, causing the comments section to never load.  
**Why heuristic fails**: Step 13 ("Verify comments page loaded") has `console_score = 1.0` because it produces the visible JS error when the timeout is detected. Step 12 (where the Firebase request itself timed out) has a lower score. Heuristic ranks the verification symptom above the timeout root cause.  
**Root problem**: Heuristic is score-based, not causality-aware. It can't distinguish "this step caused the error" from "this step revealed the error."  
**Fix applied**: The LLM is now explicitly prompted to prefer the **earlier** step when two steps have similar suspicion, and to reason about causality (timeout at 12 → verification failure at 13). With `pass_action`/`fail_action` diff, the LLM can see that step 12's action itself may have diverged (e.g., the Firebase call took longer / returned different data).

---

### 5. `areeb_salem/imdb` — Fixable ✅

**Fault**: The filmography API returns a 500 error at step 14 when loading a lead actor's credits.  
**Why heuristic fails**: Multiple adjacent steps (15, 20, 5) also have high network scores due to CDN errors and redirects during page load. Step 14's `network_score` is not the highest — it gets beaten by the general noise on the movie detail page.  
**Root problem**: Same pattern as wikipedia — background network noise from a media-heavy page swamps the actual fault signal.  
**Fix applied**: Same as wikipedia — `new_errors` vs `shared_noise` split ensures the LLM sees that step 14's 500 error is **new** in the fail trace, while steps 15/20's CDN errors are shared noise present in both.

---

### 6. `areeb_salem/npm` — Fixable ✅

**Fault**: The npm registry API call fails at step 16 with a network error when fetching package dependency data.  
**Why heuristic fails**: Step 17 ("Verify dependencies list visible") scores highly on console errors because it logs the failure of the verification. Step 16 (where the actual registry request errored) is ranked below it.  
**Root problem**: Same symptom-vs-cause confusion as hackernews.  
**Fix applied**: Same as hackernews — LLM causality reasoning + new fields.

---

### 7. `areeb_salem/pypi` — VLM only ❌

**Fault**: PyPI served a CAPTCHA/bot challenge page instead of the search results page at step 7.  
**Why heuristic fails**: The CAPTCHA page loads successfully (HTTP 200). No console errors. The action may say "verify search results loaded" and both pass and fail reach this step, but the page content is completely different.  
**Why LLM can't fix it**: HTTP 200 means "page loaded successfully" — no text signal indicates anything is wrong. The difference is purely visual (real search results vs CAPTCHA form).  
**Fix**: Phase 2 VLM — screenshot comparison at step 7 would immediately detect the wrong page.

---

## What Changed in the Code

### `_slim_step_for_llm` (in `src/llm_reasoner.py`)

Before:
```python
"pass_network": [errors in pass trace],   # showed pass errors (noise)
"fail_network": [errors in fail trace],   # showed all fail errors indiscriminately
```

After:
```python
"new_network_errors":  [...],  # errors ONLY in fail — real signal
"shared_network_noise": [...], # errors in BOTH — background noise, low diagnostic value
"missing_requests":    [...],  # URLs present in pass but absent in fail — absence of activity
"pass_request_count":  int,    # total requests pass made at this step
"fail_request_count":  int,    # total requests fail made at this step
"pass_action":  "...",         # what was done in the passing run
"fail_action":  "...",         # what was done in the failing run
"action_changed": bool,        # True if pass_action != fail_action
```

### `prompts/rerank_steps.txt`

Updated to:
- Explain `new_network_errors` vs `shared_network_noise` distinction
- Tell LLM to treat `missing_requests` and large `pass_request_count` vs low `fail_request_count` as "absence of activity" signals
- Instruct LLM to compare `pass_action` vs `fail_action` for input/navigation divergence
- Ask for ALL step IDs in the ranked output (not just top-5)

---

## Expected Impact on Metrics

After these fixes are applied and a full LLM run completes:

| Trace | Before (heuristic) | Expected after LLM fixes |
|-------|-------------------|--------------------------|
| `wolfram` | Hit@5=0, RankDist=5 | LLM should promote step 12 using `missing_requests` signal |
| `hackernews` | Hit@5=0, RankDist=5 | LLM should rank step 12 above step 13 using causality reasoning |
| `imdb` | Hit@5=0, RankDist=5 | LLM should promote step 14 — fewer steps have `new_network_errors` now |
| `npm` | Hit@5=0, RankDist=5 | LLM should promote step 16 over step 17 (cause vs symptom) |
| `efe_irem/wikipedia` | Hit@5=0 | No change — step 13 has zero text signal; early steps have synthetic off-domain noise |
| `saucedemo_2` | Hit@5=0 | No change — needs `--llm --vlm` |
| `pypi` | Hit@5=0 | No change — needs `--llm --vlm` |

If the 4 fixable cases improve to Hit@5=1, the overall Hit@5 rate would rise from **68.2% → 86.4%** (19/22 traces).  
The remaining 3 text-invisible failures (`efe_irem/wikipedia`, `saucedemo_2`, `pypi`) require Phase 2 VLM or sequence-context reasoning to address.
