# Trace Selection Rationale

This document explains which traces were selected for evaluation, which were excluded, and why.

---

## Selected Traces (22 total)

### Source: efe_irem (8 traces)

| Trace ID | Fault Step | Fault Type | Primary Signal | Notes |
|----------|-----------|------------|----------------|-------|
| `saucedemo_1` | 7 | Wrong password attempt | Console (login error) | Clean fault — single bad credential causes auth failure and downstream errors |
| `saucedemo_2` | 27 | Missing zipcode input | Screenshot only | Fault is only visible in screenshot; console/network silent. Included as a hard case for text-only diagnosis. |
| `gutenberg` | 8 | HTTP 403 Forbidden | Network | Clear 403 on ebook resource; good network signal example |
| `dictionary` | 25 | Network connection lost | Console (`net::ERR_*`) | Mid-trace disconnection; tests detection of late-appearing console errors |
| `webmd` | 2 | HTTP 404 page not found | Network | Early fault at step 2; high-score step should be obvious for heuristic |
| `wolfram` | 12 | Wrong element clicked | Absence of activity | No network errors, no console errors — fault detectable only by absence of expected requests. Hard case. |
| `elinguistics` | 11 | HTTP 404 missing resource | Network + console | Combined signal; step is mid-trace |
| `wikipedia` | 13 | Broken anchor / TOC navigation | Network noise (distractor) | Fault is subtle; many steps have incidental network noise, making heuristic ranking harder |

### Source: areeb_salem (10 traces)

| Trace ID | Fault Step | Fault Type | Primary Signal | Notes |
|----------|-----------|------------|----------------|-------|
| `amazon` | 20 | Auth-gated action (not logged in) | Console + network | Action divergence signal present; tests downstream-vs-root-cause separation |
| `bbc` | 18 | Stale CDN cache (visual regression) | Screenshot only | Text-invisible fault; included as a known hard case for text-only diagnosis |
| `github` | 23 | UI rendering crash (JS exception) | Console | Pure JS crash in fail trace; console error is the only signal |
| `hackernews` | 12 | Firebase timeout | Console + network | Async timeout; fault step followed by many quiet steps |
| `imdb` | 14 | HTTP 500 filmography API | Network | Server error on a secondary API call during page load |
| `npm` | 16 | Network error on registry fetch | Network | Package registry request fails; good isolated network fault |
| `producthunt` | 16 | Auth-gated upvote action | Console + network | Similar to amazon; tests whether LLM distinguishes auth failure from downstream UI errors |
| `pypi` | 7 | CAPTCHA / bot challenge | Screenshot only | Challenge page appears; no textual error — another known hard case |
| `wikipedia` | 17 | Broken anchor / TOC navigation | Console + network | Parallel to efe_irem/wikipedia, but different tooling schema |
| `youtube` | 13 | Metadata pipeline bug | Screenshot only | Comment count missing; error visible only in screenshot |

### Source: ersel (4 traces)

| Trace ID | Fault Step | Fault Type | Primary Signal | Notes |
|----------|-----------|------------|----------------|-------|
| `opencart_purchase_40` | 21 | Invalid search input (no results) | Intent field | `intent` field logs "Verification failed"; only ersel traces have this signal |
| `opencart_wishlist_35` | 22 | HTTP 404 nonexistent product URL | Intent + network | Both intent signal and network 404 present; strongest combined signal |
| `opencart_manufacturer_404_35` | 23 | HTTP 404 invalid manufacturer route | Network | Clean network fault; good counterpart to wishlist case |
| `opencart_contact_35` | 16 | Invalid email format submitted | Action divergence | Form accepts invalid email; fault detected via action content diff, not network/console |

---

## Fault Type Coverage Summary

| Category | Count | Trace IDs |
|----------|-------|-----------|
| HTTP 4xx errors | 6 | gutenberg (403), webmd (404), elinguistics (404), opencart_wishlist (404), opencart_manufacturer (404), hackernews (timeout→4xx) |
| HTTP 5xx errors | 1 | imdb (500) |
| Authentication / auth-gated | 3 | saucedemo_1, amazon, producthunt |
| Network disconnection | 2 | dictionary, npm |
| JavaScript / UI crash | 1 | github |
| Invalid user input | 2 | opencart_purchase_40, opencart_contact_35 |
| Screenshot-only (text-invisible) | 3 | saucedemo_2, bbc, pypi, youtube *(4 total)* |
| Subtle / absence of activity | 2 | wolfram, wikipedia (×2) |

---

## Excluded Traces

### From efe_irem

No efe_irem traces were excluded. All 8 available traces were included. The original data contained exactly these 8 test cases.

### From areeb_salem

No areeb_salem traces were excluded. All 10 available traces were included.

### From ersel

The ersel dataset contained additional traces beyond the 4 selected. Exclusion criteria:

| Reason | Description |
|--------|-------------|
| **Fail-only, no pass trace** | Some cases only had a failing `steps.json` with no corresponding passing trace, making alignment and diff-based scoring impossible without fabricating a synthetic pass trace. |
| **Language barrier** | At least one trace was recorded in Turkish (step descriptions, intent fields). While technically parseable, non-English intent/action text degrades LLM reasoning quality since the prompts and output format assume English. It was excluded rather than included as a noisy data point. |
| **Duplicate fault type** | Some additional ersel traces covered the same fault type (e.g., another invalid-search-input variant) already well-represented in the selected set. Including them would over-weight that fault type in aggregate metrics. |

---

## Key Design Decisions

**Why 22 traces?**  
This is the full set of available traces that met all inclusion criteria: (1) both a pass and a fail trace exist, (2) at least one of the four scored signals (network, console, action, intent) theoretically captures the fault, and (3) no duplicate fault-type overrepresentation.

**Why keep screenshot-only faults?**  
Traces like `saucedemo_2`, `bbc`, `pypi`, and `youtube` have faults invisible to the text-only Phase 1 pipeline. They are included deliberately to establish a lower-bound on heuristic/LLM performance and to motivate Phase 2 (VLM screenshot analysis). They are expected to score poorly and are documented as such.

**How was `fault_step` determined?**  
Ground truth was assigned manually by inspecting the trace diff: the first step where the injected fault (wrong password, wrong URL, missing input, etc.) becomes observable. It is the **root cause step**, not the last step that errored.
