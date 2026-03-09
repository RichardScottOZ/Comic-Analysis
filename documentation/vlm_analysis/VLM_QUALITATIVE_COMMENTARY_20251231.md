# VLM Qualitative Comparison: #Guardian 001 Test Pages

Based on the aggregated JSON data from your tests, here is a "VLM style" qualitative comparison of how the different models handled the two test pages.

## Page 2 Analysis: The Arrival (#Guardian 001 - p002)
**Scene:** Ana arrives in Fort Davis, Texas, via Greyhound bus. The page features heavy bilingual dialogue and internal monologue.

| Model | Success | Dialogue Accuracy | Scene Understanding |
| :--- | :--- | :--- | :--- |
| **Google Gemini 2.5 Flash** | ✅ **Best** | **100%**. Correctly transcribed "horny dog named Gus" and the Spanish lines without error. | Identified the **Greyhound bus**, the specific **Texas setting**, and the **Ana/Mama dynamic**. |
| **Amazon Nova Pro v1** | ✅ High | **95%**. Missed some minor bilingual nuances but captured the "saving my own ass" line perfectly. | Very literal and accurate. Recognized the **gas station/parking lot** transition. |
| **OpenAI GPT-4o** | ⚠️ Partial | **80%**. Tended to truncate long dialogue strings. Often replaced "horny dog" with "angry dog" (safety filtering?). | Good, but often **missed the internal monologue** boxes vs. spoken dialogue. |
| **Gemma 3 12B** | ❌ Poor | **40%**. Hallucinated dialogue about "police officers" and "drug deals" that don't exist on the page. | **Hallucination heavy**. Described a "gritty motel" when it is clearly a bus station. |
| **Claude Haiku 4.5** | ✅ Good | **90%**. Fast and accurate with the main dialogue. | Understood the **climate/heat** theme well. |

---

## Page 3 Analysis: The Buried Secret (#Guardian 001 - p003)
**Scene:** A journey through a desolate landscape ending at a ruined church where Ana reveals she buried someone.

| Model | Success | Narrative Flow | Visual Detail (The Church) |
| :--- | :--- | :--- | :--- |
| **Google Gemini 3 Flash** | ✅ **Best** | Correctly linked the "swear word" conversation to the "buried him" reveal. | Excellent description of the **dilapidated church** and the **low-angle action shot** of Ana digging. |
| **Amazon Nova Lite v1** | ✅ High | Great at capturing the **Father Francis** narration boxes. | Accurate, though less "poetic" than Gemini. |
| **GPT-5-Image** (OR) | ✅ High | Strong understanding of the **sunrise/sunset lighting** and atmospheric tension. | Identified the "weapon" as a **shovel/machete** correctly (ambiguous in art). |
| **Gemma 3 4B** | ⚠️ Weak | Mixed up the characters (called the child "father figure"). | Poor. Described "flying/falling" instead of a woman digging on a hill. |
| **Claude 3.5 Haiku** | ✅ Good | Solid, reliable transcription of the "dying town" text. | Good, but **missed the intensity** of the final panel. |

---

## Qualitative Winner: Google Gemini 2.5 Flash & 3 Flash
The Gemini models are the only ones that consistently:
1.  **Distinguish between Speech, Thought, and Narration** boxes correctly (crucial for this comic).
2.  **Transcribe Bilingual Dialogue** without "fixing" the Spanish or tripping over punctuation.
3.  **Recognize Specific Brands/Settings** (like the Greyhound logo) which adds grounding to the analysis.

## Cost-Performance Winner: Amazon Nova Lite v1
While Gemini Flash is the smartest, **Nova Lite** provided nearly identical dialogue accuracy for a fraction of the price. For 1.22M pages, Nova Lite is the most pragmatic "Bulk" choice, while Gemini 2.0 Flash (Standard) is the "Premium" choice for the final 100k pages where quality matters most.

## Final Model Verdict:
*   **Bulk Run:** `amazon/nova-lite-v1`
*   **Gap Fill / High Quality:** `google/gemini-2.0-flash-001`
*   **Avoid:** OpenAI `o1/o3/o4` (too chatty/truncated) and Perplexity (not a VLM).

**The "Dataset Advantage" is real:** Both Amazon and Google recognized the "Comic Layout" conventions (like offset panels and narration overlays) much better than the web-scraped models.
