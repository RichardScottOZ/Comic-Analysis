# VLM Benchmark: Cost vs. Speed Efficiency Analysis
**Date:** 2025-12-31  
**Dataset:** 1.22 Million Comic Pages (Projected)  
**Task:** Structured JSON Panel & Character Analysis

## 🏆 The "Golden" Selection (Best Overall Value)

These models offer 100% reliability, high speed, and reasonable costs.

| Model | Speed (s/page) | Cost/Page | 1.22M Project Cost | Success Rate | Verdict |
|:---|:---|:---|:---|:---|:---|
| **google/gemma-3-12b-it** | 4.52 s | **$0.000133** | **$162.02** | 100% | 🥇 **Best Value.** Cheap & 100% Reliable. |
| **amazon/nova-lite-v1** | 1.71 s | **$0.000314** | **$382.93** | 96% | 💎 **Value & Speed.** Excellent middle ground. |
| **google/gemini-2.0-flash-001** | **1.15 s** | $0.000636 | $775.38 | 100% | 🚀 **Speed King.** Fastest processing. |
| **openai/gpt-4.1-nano** | 1.40 s | $0.000476 | $580.62 | 100% | ⚖️ **Excellent Balance.** Fast & Affordable. |
| **google/gemini-2.0-flash-lite-001** | 1.73 s | $0.000433 | $528.11 | 100% | 💡 **Lite Champion.** Low cost, high speed. |
| **meta-llama/llama-3.2-90b-vision-instruct** | 2.63 s | $0.000748 | $912.76 | 100% | 🧠 **Smartest Fast Model.** High reasoning/size. |

---

## 📊 Detailed Efficiency Table (Sorted by Cost)

| Model | Success | Speed (s/it) | Cost/Page | 1.22M Project | Efficiency Verdict |
|:---|:---|:---|:---|:---|:---|
| mistralai/pixtral-12b | 16% | 0.73* | $0.000081 | $99.31 | ❌ **Broken.** Too many failures. |
| **google/gemma-3-12b-it** | 100% | 4.52 | $0.000133 | $162.02 | 💎 **Incredible Value.** |
| **amazon/nova-lite-v1** | 96% | 1.71 | **$0.000314** | **$382.93** | 💎 **Fast & Cheap.** High value. |
| mistralai/ministral-14b-2512 | 96% | 1.59 | $0.000428 | $521.87 | ✅ Very Fast. |
| **google/gemini-2.0-flash-lite-001** | 100% | 1.73 | $0.000433 | $528.11 | ✅ High Value. |
| **openai/gpt-4.1-nano** | 100% | 1.40 | $0.000476 | $580.62 | ✅ High Value. |
| **google/gemini-2.0-flash-001** | 100% | 1.15 | $0.000636 | $775.38 | ⚡ **Top Choice for Time.** |
| **meta-llama/llama-3.2-90b-vision-instruct** | 100% | 2.63 | $0.000748 | $912.76 | ✅ Solid Performance. |
| qwen/qwen3-vl-30b-a3b-instruct | 100% | 2.88 | $0.000860 | $1,049.69 | ✅ Good. |
| qwen/qwen2.5-vl-72b-instruct | 100% | 4.69 | $0.000945 | $1,152.56 | ✅ Solid. |
| amazon/nova-2-lite-v1 | 100% | 1.65 | $0.002484 | $3,030.82 | 💸 Expensive for the speed. |
| google/gemini-2.5-flash | 100% | 1.62 | $0.003674 | $4,482.33 | 💸 **Avoid.** Gemini 2.0 is better & cheaper. |
| openai/gpt-4o-mini | 100% | 2.70 | $0.004238 | $5,169.92 | 💸 **Avoid.** Overpriced for this task. |

---

## 🔍 Key Insights

### 1. The Gemini Paradox
**Gemini 2.0 Flash** is both **faster** and **~6x cheaper** than Gemini 2.5 Flash for this specific workflow. There is zero reason to use Gemini 2.5 Flash for this project right now.

### 2. The GPT-4o-Mini Trap
While often praised for value, **GPT-4o-Mini** is actually one of the most expensive options for this high-volume task ($5,169 vs $162 for Gemma 3 12B). Its speed is also mediocre compared to the newer Nano and Flash models.

### 3. High-Volume Strategy
If time is not the primary constraint, **Gemma 3 12B** is the absolute winner ($162 total cost). 

However, **Amazon Nova Lite v1** ($382 total cost) offers a compelling middle ground: it is **2.6x faster** than Gemma 3 12B while still being **$4,700+ cheaper** than GPT-4o-Mini.

### 4. Throughput Calculation
At **Gemini 2.0 Flash** speeds (1.15s/page):
- 8 Workers: ~25,000 pages / hour.
- 1.22M pages: **~48 hours total wall time.**
- Total Cost: **$775.**

At **Amazon Nova Lite v1** speeds (1.71s/page):
- 8 Workers: ~16,800 pages / hour.
- 1.22M pages: **~72 hours (3 days).**
- Total Cost: **$382.**

At **Gemma 3 12B** speeds (4.52s/page):
- 8 Workers: ~6,300 pages / hour.
- 1.22M pages: **~193 hours (8 days).**
- Total Cost: **$162.**

---

## 🛠️ Handling the Unicode Issue
The `UnicodeEncodeError` in `batch_comic_analysis_vlm.py` was caused by non-standard characters in VLM outputs being printed to the Windows terminal. The fix ensures that the log writing uses `utf-8` encoding and terminal output handles broad character sets safely.

```python
# Fixed logging with UTF-8
with open('vlm_failures.log', 'a', encoding='utf-8') as log:
    log.write(f"{res['canonical_id']}: {res.get('error')}\n")
```

