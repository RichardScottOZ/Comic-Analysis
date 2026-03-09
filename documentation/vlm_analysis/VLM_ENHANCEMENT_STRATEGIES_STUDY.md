# VLM Enhancement Strategies Study
**Date:** January 1, 2026
**Subject:** Comparative methodologies for improving VLM comic analysis using prior geometric data.

## Overview
This study evaluates three distinct strategies for combining specialized object detection (Faster R-CNN) with multimodal generative models (VLMs) to achieve high-fidelity comic page extraction.

---

## Strategy 1: Integrated Grounding (Zero-Shot)
**Method:** The VLM is provided with the raw image and asked to perform narrative analysis AND generate its own bounding boxes (`box_2d`) in a single pass.
- **Script:** `src/tests/test_integrated_analysis.py`
- **Pros:** Full alignment between description and coordinates; single API call.
- **Cons:** "Lite" models often hallucinate coordinates or "squash" the geometry.
- **Winner:** **Google Gemini 2.0 Flash (Standard)** and **Gemini 3 Pro**.

## Strategy 2: R-CNN JSON Guidance (Prompt-Enhanced)
**Method:** The VLM is provided with the raw image + a JSON list of Faster R-CNN detections (Coordinates + Labels) inside the text prompt.
- **Script:** `src/tests/test_rcnn_enhanced_analysis.py`
- **Pros:** Acts as a "geometric skeleton," forcing the VLM to acknowledge all panels and text regions. Drastically improves "Lite" model panel counts.
- **Cons:** Increases input tokens; VLM might blindly trust incorrect R-CNN boxes.
- **Winner:** **Qwen 3 VL 8B** (matched Premium quality when guided by R-CNN).

## Strategy 3: Visual Prompting (Annotated Image)
**Method:** Faster R-CNN boxes are pre-rendered onto the image (Visual Overlays). The VLM is fed this "dirty" image and asked to describe the contents of the labeled boxes.
- **Script:** `src/tests/test_visual_prompting.py`
- **Pros:** Leverages the VLM's native visual processing (easier to "see" a red box than to parse coordinate numbers).
- **Cons:** Requires a local pre-rendering step; cannot easily "refine" the boxes.
- **Winner:** **Amazon Nova Lite v1** (excellent at "reading" the labels on the image).

---

## Comparison Table: Strategy vs. Model Quality

| Strategy | Recommended Model | Best For... |
| :--- | :--- | :--- |
| **Integrated** | **Gemini 2.0 Flash (Std)** | High-quality "one-stop" extraction. |
| **JSON Guided** | **Qwen 3 VL 8B** | Budget run with high panel-count reliability. |
| **Visual Prompt** | **Nova Lite v1** | Deep captioning of pre-detected objects. |

## Conclusion: The "Hybrid" Holy Grail
For the final 1.22M dataset, the most robust architecture is a **Hybrid Ensemble**:
1.  **Geometric Detection:** Faster R-CNN (Local) for precise boxes.
2.  **Semantic Description:** Gemini 2.0 Flash (Integrated) for the narrative.
3.  **Cross-Validation:** Use the R-CNN boxes to validate the VLM's `box_2d` output (Ensemble Validation).
