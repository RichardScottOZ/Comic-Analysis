# Gemini Flash 2.5 Lite — JSON Response Defence

**Pipeline:** `src/version2/batch_vlm_analysis_lithops_v2.py`  
**Model:** `google/gemini-2.5-flash-lite` via OpenRouter  
**Infrastructure:** AWS Lambda (Lithops), S3  
**Date:** 2026-03-15  

---

## Overview

The production VLM pipeline processes comic-book page images through Gemini Flash 2.5 Lite and
expects a structured JSON response for each page (overall summary + per-panel bounding boxes,
descriptions, dialogue, etc.).  Gemini frequently returns *technically invalid* JSON — unescaped
inner quotation marks, truncated responses, markdown formatting artefacts, invalid Unicode escapes,
and more.  This document records every error class encountered during iterative batch testing
(~5 000 pages), the root cause of each, and the repair strategy implemented in `inner_repair_json`.

---

## Batch Testing Summary

| Batch | Records | Success | Fail | Fail % | Notes |
|-------|---------|---------|------|--------|-------|
| 1 | 100 | 93 | 7 | 7.0% | Baseline |
| 2 | 100 | 98 | 2 | 2.0% | After fix round 1 |
| 3 | 200 | 185 | 15 | 7.5% | New pages |
| 4 | 15 | 14 | 1 | 6.7% | Retry of 15 |
| 5 | 500 | 483 | 17 | 3.4% | New pages |
| 6 | 17 | 15 | 2 | 11.8% | Retry of 17 |
| 7 | 1 000 | 984 | 16 | 1.6% | After fix round 2 |
| 8 | 16 | 15 | 1 | 6.3% | Retry of 16 |
| 9 | 2 000 | 1 979 | 21 | 1.0% | After fix round 3 |
| 10 | 21 | 20 | 1 | 4.8% | Retry of 21 |
| 11 | 1 000 | 992 | 8 | 0.8% | After fix round 4 |
| 12 | 8 | 8 | 0 | 0.0% | Retry of 8 — all clear |

**Effective production failure rate: ~0%** when a single automatic retry pass is added (all
observed failures resolved on retry with a fresh API call).

---

## Token Distribution (representative 1 000-record batch)

```
Token stats (n=1000): min=351, mean=2385, p95=4071, max=8223, truncated(>=8192)=19
```

- ~1.9% of pages hit the `max_tokens=8192` ceiling and are truncated mid-response.
- Truncation is reliably identified by `completion_tokens >= 8192` in the usage dict.
- Tokens slightly above 8192 (e.g. 8199, 8217, 8223) are returned by the API due to
  special-token accounting; these are still effectively truncated responses.
- The p95 of ~4 000 tokens means most pages complete well under the limit.
- Prompt tokens are ~3 657 for the current production prompt (image + text).

---

## Error Classes & Root Causes

### 1. Unescaped Inner Quotes — `Expecting ',' delimiter`

**Frequency:** Most common non-truncation failure (~60% of non-truncation errors)

**Cause:** The LLM embeds quoted words directly inside JSON string values without escaping:
```
"description": "She is about to "school" him on the matter"
```
The parser closes the outer string at `"school`, making `him on the matter"` outside the string.

**Sub-patterns:**

| `json_str[pos]` char | Meaning | Fix |
|---|---|---|
| `"` | Opening quote of the inner word (e.g. `"school`) | Escape directly: `\"` |
| Letter/punctuation | Outer string closed early; error lands after closing `"` | Backwards search for premature closer |
| `.`, `?`, `!` (was missed) | Period/punctuation after quoted word (e.g. `"Whatever it takes."`) | Covered by `not in ':{}[],\\"'` condition |

**Backwards search mechanic (`_last_unescaped_quote`):**  
Scan backwards from the error position up to 200 characters to find the last unescaped `"` and
escape it.  The 200-char lookback cap prevents accidentally escaping structural quotes far from the
error site.

---

### 2. Unescaped Quoted Word Followed By Comma — `Expecting property name enclosed in double quotes`

**Frequency:** ~15% of non-truncation errors

**Cause:** When the unescaped word is followed by `,` in the text (e.g. `"Doc", is speaking`):
1. Outer string closes at `"` before `Doc`.
2. The literal `,` in `"Doc",` is consumed as a key-value separator.
3. Parser looks for the next key but finds a plain letter → `Expecting property name`.

This is a **two-step unescaped-quote fix** — the first backward-search pass (triggered by an
earlier `,` error) escapes the opening `\"Doc`, but the string then closes at the second `"` of
`Doc`.  The `Expecting property name` handler applies a further backwards search to escape that
second `"` as well.

---

### 3. Response Truncation — `Unterminated string` / `Expecting value` / `Expecting ':'`

**Frequency:** ~35% of all failures; reliably identified by `completion_tokens >= 8192`

**Cause:** Response hits the `max_tokens=8192` ceiling mid-JSON, typically mid-string inside a
panel's `description` field.

**Fix (stack-based closer):**  
A stack-based scanner tracks every `{` and `[` encountered outside strings.  On truncation:
1. If `_in_s` is True (truncated inside a string), append `"` to close it.
2. Emit `''.join(reversed(_stk))` to close all open brackets in the correct nesting order.

The old naive approach (`']'*N + '}'*M`) produced wrong nesting order for structures like
`{"panels": [{"description": "...` which needs `"}]}` not `}]}`.

**Step 6 — depth-based truncation:**  
After the stack closer, scan for the first position where bracket depth returns to 0 and return
only up to that point, discarding any duplicate/extra JSON objects.

---

### 4. Unescaped Quote in a JSON Key Name — `Expecting ':' delimiter`

**Frequency:** ~10% of failures in deeper panel content

**Cause:** The LLM writes a character name as a key:
```json
"character "Lex" Luthor": "description"
```
The key string closes at `"` before `Lex`, and the parser finds `L` where it expects `:`.

**Fix:**  
`Expecting ':' delimiter` handler: if `json_str[pos]` is a letter/non-structural char AND the
error is more than 200 chars from EOF, apply backwards search to escape the premature closer.

**Structural quote guard (`_is_structural_quote`):**  
A `':'` error near a structural key opener (a `"` at the start of an indented line, preceded by
newline + whitespace) must **not** be escaped — it is a legitimate key opener.  The guard checks
whether the quote found by backwards search is preceded (ignoring spaces/tabs) by `\n`, `{`, `[`,
or `,`.  If so, break and let the stack closer handle it.

This guard also applies to `Expecting property name` backwards searches but NOT to
`Expecting ','` backwards searches (where over-escaping eventually hits a break-char and stops,
and the stack closer recovers).

---

### 5. Invalid Unicode Escape — `Invalid \uXXXX escape`

**Frequency:** Rare (~1–2 per 1 000 pages)

**Cause:** The model outputs `\u` followed by non-hex characters, e.g. `\uROSC` or `\u ` (space).
Python's json module raises `Invalid \uXXXX escape` for these.

**Fix (step 2.5):**  
```python
json_str = re.sub(r'\\u(?![0-9a-fA-F]{4})', r'\\\\u', json_str)
```
Negative lookahead: if `\u` is NOT followed by exactly 4 hex digits, double-escape the backslash
so it becomes a literal `\u` string rather than a Unicode escape sequence.

---

### 6. Invalid Single-Character Escape — `Invalid \\X escape`

**Frequency:** Rare

**Cause:** The model outputs invalid escapes like `\x`, `\p`, `\d` inside strings.

**Fix (step 2.5):**  
```python
def fix_escapes(match):
    esc = match.group(1)
    valid = {'"', '\\', '/', 'b', 'f', 'n', 'r', 't'}
    if esc in valid or esc == 'u':
        return match.group(0)
    return '\\\\' + esc
json_str = re.sub(r'\\([^"\\\/bfnrtu])', fix_escapes, json_str)
```

---

### 7. Markdown Code Fences — `Expecting value`

**Frequency:** Occasional (~1–3%)

**Cause:** The model wraps its JSON response in ` ```json ... ``` ` fences despite being asked
for raw JSON via `response_format: {type: json_object}`.

**Fix (step 1):** Strip leading ` ```json ` or ` ``` ` and trailing ` ``` `.

---

### 8. Markdown Bold Markers Before Keys — `Expecting ',' delimiter`

**Frequency:** Rare but distinctive

**Cause:** The model uses markdown bold formatting before a JSON key:
```
  ** "panels": [
```
The `**` before `"panels"` breaks the JSON structure — the parser sees `*` where it expects a
property name.

**Fix (step 1.5):**  
```python
json_str = re.sub(r'(?m)^(\s*)\*+\s*(?=")', r'\1', json_str)
```
Strips any `*` sequence at the start of an indented line when immediately before a `"`.

---

### 9. `None` / `null` API Response — `AttributeError: 'NoneType'`

**Frequency:** Rare, when OpenRouter returns `"message": null` in a choice

**Cause:** `choice['message']['content']` raises `AttributeError` when `message` is `null`.

**Fix:**
```python
message = choice.get('message') or {}
content = message.get('content') or ''
```

---

### 10. Zero-Token Response (HTTP 200 with empty content)

**Frequency:** Very rare (~2 per 2 000 pages)

**Cause:** OpenRouter occasionally returns HTTP 200 with empty `content` and all-zero `usage`
counts.  Possibly a cached error response or provider timeout served as success.

**Fix:** Detect `not content and completion_tokens == 0` and retry up to 2 times with 5s/10s
backoff before failing.

---

### 11. Transient HTTP Errors (429, 5xx)

**Cause:** Rate limits (429) and transient Lambda/OpenRouter server errors (5xx).

**Fix:** Retry loop with:
- 429 → `sleep(15 * attempt)` (generous back-off for rate limits)
- 5xx → `sleep(5 * attempt)` (brief back-off for transient errors)
- 4xx other → no retry (auth failure, bad request)

---

## Repair Pipeline Step Order (`inner_repair_json`)

```
1.    Strip markdown code fences (``` ... ```)
1.5.  Strip markdown bold/italic markers before keys (** "key" → "key")
2.    Find first { and discard prefix
2.5.  Fix invalid escape sequences (\x → \\x, \uINVALID → \\uINVALID)
3-4.  Fix missing commas between properties (regex)
4.5.  Iterative unescaped-quote fixer (up to 100 parse-fix-reparse iterations):
        - Expecting ',' delimiter:
            pos == '"'         → escape directly (inner word opening quote)
            pos is letter/etc  → backwards search for premature closer
        - Expecting property name:
            pos is letter/etc  → backwards search (with structural-quote guard)
        - Expecting ':' delimiter:
            pos == '"'         → escape directly
            pos is letter/etc AND >200 chars from EOF AND not structural quote
                               → backwards search
        - Expecting value:
            trailing comma → strip comma
            None/undefined/NaN → replace with null
5.    Stack-based closer (close open strings and {[ in correct nesting order)
6.    Depth-based truncation (return first complete JSON object only)
```

---

## Key Design Decisions

### Why `strict=False` in step 4.5?
Using `json.loads(json_str, strict=False)` allows the parser to report errors deep inside
multi-line strings (real-world panel descriptions span many lines).  In strict mode, the parser
would abort at the first newline inside a string, giving an unhelpful `Unterminated string` error
at the wrong position.

### Why 200-char lookback limit on backwards search?
Prevents the backwards search from finding structural quotes (key openers, array element quotes)
that are far from the error site and should not be escaped.

### Why no `_is_structural_quote` guard on `','` backwards search?
Adding the structural guard to `','` caused regressions (Abbott-001 p002 style cases where a
subsequent retry would have succeeded but the guard caused early termination and step 5/6 produced
invalid output).  For `','` errors, over-escaping a structural quote eventually hits `:` or another
break-character and stops naturally; step 5/6 then recovers.  The guard is only applied where
it was confirmed necessary: `':'` errors and `Expecting property name` errors.

### Why the `_is_structural_quote` guard IS needed for `':'` errors?
The specific failure pattern (p063): a truncated response where a cross-line `strict=False` parse
caused the backwards search to find a key opener (`"panel_number"` at start of indented line),
and escaping it produced `\"panel_number"` — corrupting the key structure.  The error position
was ~11 000 chars from EOF (clearly not truncation-at-EOF) but the `"` found was still structural.

### Stack-based closer vs. naive bracket counter
The naive approach counted `{` and `[` and appended `']'*open_brackets + '}'*open_braces`.  This
produced wrong nesting order for `{"panels": [{"desc": "...` → needs `"}]}` but naive produced
`}]}`.  The stack-based approach tracks open tokens in order and reverses them, always producing
correct nesting.

---

## Token Budget Considerations

| max_tokens | Truncation rate | Notes |
|---|---|---|
| 4 096 | ~8–10% estimated | Would trigger stack-closer much more frequently |
| 8 192 | ~1.9% observed | Current production setting |
| 16 384 | ~0.1% estimated | Would require larger Lambda memory |

The current 8 192 limit is a reasonable trade-off.  Increasing to 16 384 would nearly eliminate
truncation failures but at higher per-call cost and potentially requiring a Lambda memory increase.

---

## Recommended Next Step: Auto-Retry Pass

All 8 remaining failures in the final batch resolved on a simple re-run.  Adding an automatic
retry pass inside `main()` would bring the effective failure rate to ~0%:

```python
# After chunk results: re-submit failures for one retry pass
failed_ids = {r['canonical_id'] for r in results if r['status'] == 'error'}
if failed_ids:
    retry_tasks = [t for t in tasks if t['task_data']['canonical_id'] in failed_ids]
    retry_futures = fexec.map(process_page_vlm, retry_tasks)
    retry_results = fexec.get_result(retry_futures)
    # merge retry_results back into results
```

This handles:
- Nondeterministic LLM responses that happen to be cleaner on the second call
- Truncated responses where a shorter retry response avoids truncation
- Transient API issues that the inner retry loop didn't catch

---

## Throughput & Rate Limiting

### Observed chunk timing anomaly (2026-03-15, 3 000-record batch)

| Chunk | Duration | Truncated | Fail | Note |
|-------|----------|-----------|------|------|
| 1 (records 5001–6000) | ~1 min | 21 (2.1%) | 11 | Normal |
| 2 (records 6001–7000) | ~3.5 min | **48 (4.8%)** | 20 | Rate-limit degradation |
| 3 (records 7001–8000) | hung at 2/1000 | — | — | Full rate-limit saturation |

**Root cause:** OpenRouter has per-API-key rate limits (RPM and/or concurrent-request limits).
At `--workers 100` with a 1 000-record chunk, Lithops can launch all 1 000 Lambdas nearly
simultaneously.  After 2 000 API calls in ~4 minutes, OpenRouter throttles heavily.

- Chunk 2 degradation signals: 3.5× slower duration AND 2.3× more truncations (the model
  truncates under load, serving shorter responses more quickly).
- Chunk 3 hit the wall: all 1 000 Lambdas received 429s and entered their retry sleep loops
  simultaneously.  With backoff of 15 → 30 → 45 seconds (90 s total), plus Lambda timeout
  constraints, most functions timed out before succeeding.

### Fixes applied

1. **`--jitter` flag (default 30 s):** Each Lambda sleeps `random.uniform(0, jitter_max)` seconds
   before its first API call.  This spreads 1 000 concurrent API calls over a 30-second window
   (~33 req/s) instead of a sub-second burst.

2. **Jittered 429 backoff:** Retry sleeps are now `15*(attempt+1) + random(0,10)` and
   `5*(attempt+1) + random(0,5)` so all rate-limited Lambdas do not retry at the same moment.

### Recommended settings for large batches

```
--workers 50 --jitter 30    # ~33 req/s peak, well under typical OpenRouter limits
--workers 100 --jitter 60   # same ~17 req/s, more cautious
```

Disable jitter only for small batches where burst is not a concern:
```
--limit 50 --workers 10 --jitter 0
```

---

## Empirical Cost Analysis (2026-03-15, production run)

### Dataset
- **8 341 Comic Analysis API calls** processed on this date (~0.7% of the 1.2M corpus)
- **Single model:** `google/gemini-2.5-flash-lite` via OpenRouter — 100% of calls
- **Source:** `openrouter_activity_2026-03-15.csv`
- **Analysis script:** `testcopilot/comic_report.py` → `comic_report_results.json`

### Derived pricing rates (from non-cached production rows)
| Token type | Google listed rate | OpenRouter actual rate | Markup |
|---|---|---|---|
| Input | $0.075 / M | **$0.10 / M** | ~33% |
| Output | $0.30 / M | **$0.40 / M** | ~33% |

OpenRouter's ~33% provider markup was absent from the March 1 benchmark doc, which used Google's listed rates.

### Token distribution (exact, n=8 341)
| Metric | Value |
|---|---|
| Avg prompt tokens | **3 457** (mix of 3 657 Amazon + 2 109 Calibre) |
| Avg completion tokens | **2 499** |
| Truncated responses (completion ≥ 8 192) | **227 / 8 341 = 2.72%** |
| finish_reason = `stop` | 8 108 (97.2%) |
| finish_reason = `length` | 229 (2.7%) |
| finish_reason = `` (empty) | 4 (0.05%) |

### Per-call cost statistics (exact, n=8 341)
| Metric | Value |
|---|---|
| Total spend | **$11.19** |
| Mean | **$0.001342** |
| Median (p50) | $0.001278 |
| p75 | $0.001553 |
| p90 | $0.001870 |
| p95 | $0.002204 |
| p99 | $0.003642 |
| Max | $0.003668 |

### Projection to 1.2 million pages (exact mean extrapolation)

| Component | Cost |
|---|---|
| Model API ($0.001342 × 1 200 000) | **$1 610** |
| AWS S3 egress + Lambda (fixed) | $608 |
| **Total all-in** | **$2 218** |

vs. March 1 doc estimate of **$1 988** — a **+$230 (+11.6%) underestimate**, almost entirely due to the OpenRouter ~33% provider markup not being in the original doc.

### Calibre / lower-res tranche (estimated 300–400K pages)
- Avg prompt drops 3 657 → 2 109 tokens (~42% reduction)
- Estimated cost: ~$0.00093/page vs $0.00134 for Amazon
- Savings: ~$0.00041/page × 400K pages ≈ **~$164 saving**
- Blended 1.2M projection with Calibre mix: model ~$1 500, all-in ~**$2 108**

### Additional cost overheads (not in the $2 218 figure)

| Item | Estimate | Notes |
|---|---|---|
| **Failure reruns** | +~$16 | ~1% of pages make 2 API calls; 1.2M × 1% × $0.001342 |
| **Internal retries** | +~$5–10 | 429/5xx/zero-token retries inside Lambda — billed but invisible in success count |
| **AWS egress + Lambda** | $608 *(estimated)* | Pending verification via AWS Cost Explorer in coming days |
| **Calibre tranche saving** | −~$164 | ~400K pages at $0.00093/page vs $0.00134 |

**Likely true all-in range: ~$2,075–$2,250** once Calibre savings and retry overhead are netted off and AWS actuals confirmed.
