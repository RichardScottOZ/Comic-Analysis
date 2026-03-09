# VLM Qualitative Comparison Report: #Guardian 001 Test Pages
**Date:** 2025-12-31  
**Test Subjects:** #Guardian 001 Page 2 (The Arrival) and Page 3 (The Buried Secret)  
**Models Evaluated:** ~20 Premium and Budget VLMs (OpenRouter)

## Overview
This report provides a comparative analysis of VLM performance on two specific comic pages. These pages were selected for their complexity, including bilingual dialogue, internal monologues, and dramatic action sequences.

---

## Page 2 Analysis: The Arrival (#Guardian 001 - p002)
**Scene:** Ana arrives in Fort Davis, Texas, via Greyhound bus. This page tests the model's ability to handle heavy text, distinct speech/thought bubbles, and bilingual (English/Spanish) content.

| Model | Dialogue Accuracy | Scene Understanding | Verdict |
| :--- | :--- | :--- | :--- |
| **Google Gemini 2.5 Flash** | **100%** (Perfect) | Identified **Greyhound bus**, specific **Texas setting**, and **Ana/Mama dynamic**. | ✅ **Winner** |
| **Amazon Nova Pro v1** | **95%** (High) | Very literal and accurate. Recognized the **gas station/parking lot** transition. | ✅ Excellent |
| **Claude Haiku 4.5** | **90%** (Good) | Understood the **climate/heat** theme and character frustration well. | ✅ Strong |
| **OpenAI GPT-4o** | **80%** (Partial) | Tended to truncate dialogue strings. Safety filtered "horny dog" to "angry dog". | ⚠️ Moody |
| **Gemma 3 12B** | **40%** (Poor) | **Hallucinated** "police officers" and "drug deals" not present on the page. | ❌ Unreliable |

---

## Page 3 Analysis: The Buried Secret (#Guardian 001 - p003)
**Scene:** A journey through a desolate landscape ending at a ruined church where Ana reveals a dark secret while digging. This page tests narrative flow and visual action interpretation.

| Model | Narrative Flow | Visual Detail (The Church) | Verdict |
| :--- | :--- | :--- | :--- |
| **Google Gemini 3 Flash** | **Excellent** | Perfect link between "swear word" dialogue and the "buried him" reveal. | ✅ **Winner** |
| **Amazon Nova Lite v1** | **High** | Great at capturing the **Father Francis** narration boxes accurately. | ✅ Best Value |
| **GPT-5-Image** (OR) | **High** | Strong understanding of **sunset lighting** and atmospheric tension. | ✅ Capable |
| **Claude 3.5 Haiku** | **Good** | Solid transcription of "dying town" text, though missed final panel intensity. | ✅ Reliable |
| **Gemma 3 4B** | **Weak** | Mixed up characters; described "flying" instead of a woman digging. | ❌ Poor |

---

## Key Findings & Qualitative Verdict

### 1. The "Dataset Advantage" Hypothesis
Both **Google** and **Amazon** models demonstrated a superior "native" understanding of comic layout conventions (distinguishing between speech, thought, and narration boxes). This strongly supports the theory that their training data included vast comic repositories (Google Play Books and Amazon ComiXology).

### 2. Dialogue & Bilingual Handling
*   **Gemini** models were the only ones to transcribe Spanish lines without "correcting" them or introducing syntax errors.
*   **OpenAI** models showed a tendency to hallucinate "summaries" of dialogue rather than literal transcriptions when the text was dense.

### 3. Failure Modes
*   **Reasoning Models (o1/o3):** Frequently failed with "Response ended prematurely" or truncated JSON. They are too verbose for structured extraction.
*   **Search Models (Perplexity/Sonar):** Failed instantly by trying to "search" for the comic instead of analyzing the pixels.

## Final Recommendation

*   **For the Bulk Run (1.22M Pages):** Use **Amazon Nova Lite v1**. It provides ~95% of the quality of premium models at ~30% of the cost.
*   **For the Final Cleanup/Premium Run (~100k Pages):** Use **Google Gemini 2.0 Flash (Standard)**. It is the gold standard for comic transcription and layout understanding.
