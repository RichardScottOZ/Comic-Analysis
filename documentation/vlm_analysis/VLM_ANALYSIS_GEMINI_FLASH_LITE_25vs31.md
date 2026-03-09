# VLM Analysis: Gemini 2.5 Flash Lite vs. Gemini 3.1 Flash Lite
**Date:** March 1, 2026
**Subject:** Comparative Cost and Performance Analysis for 1.2 Million Page Extraction

## 1. Executive Summary
This report evaluates the transition from the "Gold Standard" Gemini 2.5 Flash Lite to the newly released Gemini 3.1 Flash Lite. While Gemini 3.1 is approximately **1.9x more expensive** per page, it offers a **35% reduction in input token usage** and superior spatial grounding capabilities. Given the high-resolution nature of the source archive (2.85 MB average), Gemini 3.1 is the technically superior choice, while Gemini 2.5 remains the value leader for the remaining 4-month lifecycle.

---

## 2. Empirical Benchmark Data
Data based on a 30-page tiered sample (Amazon, Calibre, and High-Res Stress Test).

### 2.1 Resolution & Storage Profile
*   **Average File Size:** 2.85 MB per page
*   **Maximum File Size:** 9.00 MB per page
*   **Total Projected Data (1.2M Pages):** 3.34 Terabytes

### 2.2 Token Efficiency (Vision Encoder)
| Metric | Gemini 2.5 Flash Lite | Gemini 3.1 Flash Lite | Improvement |
| :--- | :--- | :--- | :--- |
| **Input Tokens (Avg)** | 2,883 | **1,384** | **~52% Reduction** |
| **Input Tokens (High-Res)** | 3,657 | **1,411** | **~61% Reduction** |
| **Completion Tokens** | 1,500 - 4,000 | 1,000 - 2,500 | Variable |

**Analysis:** Gemini 3.1 features a vastly more efficient visual vocabulary. It does not "tile" high-resolution images as aggressively as 2.5, resulting in significantly lower and more stable input token counts regardless of megapixels.

---

## 3. Financial Comparison (All-In Cost)
Calculated for 1.2 million pages, including AWS infrastructure taxes.

| Cost Component | Gemini 2.5 Flash Lite | Gemini 3.1 Flash Lite |
| :--- | :--- | :--- |
| **Model Cost (API)** | ~$1,380.00 | ~$2,520.00 |
| **AWS S3 Egress (3.34 TB)** | $300.60 | $300.60 |
| **S3 API Requests** | $7.00 | $7.00 |
| **Lithops Compute (Lambda)** | ~$300.00 | ~$300.00 |
| **TOTAL PROJECT COST** | **~$1,987.60** | **~$3,127.60** |
| **Cost Per Page** | **$0.0016** | **$0.0026** |

---

## 4. Technical Trade-offs

### 4.1 Vision & Grounding
*   **Gemini 3.1 Advantage:** The technical report indicates significant gains in the **MMMU** and **FACTS** benchmarks. In local tests, 3.1 delivered "excellent" zero-shot bounding boxes that closely align with Faster R-CNN hulls.
*   **The "Horny Dog" Test:** Gemini 3.1 maintains the high-fidelity OCR capability required to transcribe complex onomatopoeia and stylized text that other affordable VLMs (like Qwen or Llama) consistently fail.

### 4.2 Lifecycle & Risk
*   **Gemini 2.5:** Scheduled for permanent shutdown on **July 22, 2026**. Starting a 1.2M run now leaves only a 4-month window for completion and error correction.
*   **Gemini 3.1:** The current "Frontier" model. Future-proof for the duration of the project.

### 4.3 Throughput
*   Because Gemini 3.1 uses fewer input tokens, it is less likely to hit "Context Window" limits or "Rate Limit" quotas on high-resolution pages, leading to higher overall pipeline stability in Lithops.

---

## 5. Infrastructure Breakdown (The Fixed Tax)
Regardless of the model chosen, the infrastructure tax for processing high-resolution images via Lithops is **~$608.00**. 
*   **3.34 TB Egress:** This is the non-negotiable cost of moving the raw pixels from S3 to the OpenRouter API.
*   **Compute:** 1.2 million Lambda invocations (approx. 10s each).

---

## 6. Final Recommendation
1.  **For Maximum Value:** Use **Gemini 2.5 Flash Lite**. It is $1,140 cheaper and provides "Gold Standard" quality. **Risk:** Must finish the full run before the July shutdown.
2.  **For Technical Excellence:** Use **Gemini 3.1 Flash Lite**. It is significantly more efficient at the architectural level, future-proof, and less likely to fail on outlier high-res pages.

**Recommended Action:** Proceed with **Gemini 3.1 Flash Lite** if the budget allows for the $3,100 all-in cost. The 50% token efficiency and improved grounding factuality make it the safer choice for a foundational narrative dataset of this magnitude.
