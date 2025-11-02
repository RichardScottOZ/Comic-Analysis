# Panel Alignment Plan

This document outlines a plan to analyze, validate, and improve the alignment between the two sources of panel detections in the CoMiX project: the Faster R-CNN model and the Vision-Language Model (VLM).

## 1. The Goal: Improving Panel Alignment

The current pipeline generates two sets of panel data for each page: one from a computer vision object detection model (Faster R-CNN) and one from the VLM's own internal analysis. The agreement between these two sources is low (approx. 25%), and there is a hypothesis that the R-CNN model is overly aggressive, particularly on covers.

The goal is to determine which detector is more reliable and implement a strategy to improve the overall quality and consistency of panel data.

---

## 2. Key Terminology

To ensure clarity, here are definitions for the core concepts discussed in this plan.

*   **Panel Count Disagreement**: This is the most basic level of mismatch, identified by the `find_disagreements.py` script. It occurs whenever the number of panels reported by the R-CNN does not equal the number of panels reported by the VLM for the same page (`rcnn_panels != vlm_panels`). This indicates that the VLM failed to process one or more of the panels provided by the R-CNN.

*   **Panel Count Agreement / Perfect Match**: These two terms are synonymous in this project. This state occurs when `rcnn_panels == vlm_panels`. It signifies that the VLM has successfully processed and returned an analysis for every single panel that the R-CNN detected on that page. This is the ideal state for generating high-quality data, as it indicates no data was lost during the VLM analysis stage.

---

## 3. Prerequisite: Human-in-the-Loop Validation

Before we can fix the problem, we must first establish a "ground truth" by determining which detector is more accurate on your specific dataset. This requires human evaluation.

### Proposed Tool: The Panel Validator UI

I propose creating a new, simple Flask-based tool called `tools/panel_validator_ui.py`. This tool would:

1.  **Find Disagreements:** Scan the dataset to find pages where the R-CNN and VLM disagree on the number of panels.
2.  **Display Side-by-Side:** For each disagreement, it would display the page image twice:
    *   On the left, it would render the bounding boxes detected by **Faster R-CNN**.
    *   On the right, it would render the bounding boxes detected by the **VLM**.
3.  **Collect Votes:** Underneath the images, it would have three simple buttons: "R-CNN is Better", "VLM is Better", and "Both are Bad".
4.  **Generate a Report:** Each vote would be logged to a `validation_results.csv`. After a validation session, this report will provide a quantitative answer to the question: **Which detector is right more often?**

## 4. Experimental Paths

The results from the validation tool will determine the next steps.

### Path A: If the R-CNN Model is Weaker

If your validation shows that the VLM is consistently better at identifying panels, we should pursue your idea of finding a better object detection model.

*   **The Plan:** Replace the current Faster R-CNN model with a state-of-the-art alternative.
*   **Recommendation:** Based on the research report below, the most promising approach would be to **fine-tune a YOLO or DETR model**. These architectures (specifically recent versions like YOLOv9 or RT-DETR) are the current industry standard and consistently outperform older models like Faster R-CNN in both speed and accuracy. You would fine-tune one of these models on your existing panel coordinate data to create a much more accurate panel detector.

### Path B: If the VLM is the Weaker Detector

If your validation shows that the Faster R-CNN is more reliable, we should treat its output as the ground truth and force the VLM to conform to it. This aligns with your idea of "giving the R-CNN output to the VLM."

*   **The Plan:** Implement **Visual Prompting**.
*   **Implementation:** We would modify the `batch_comic_analysis_multi.py` script. Instead of asking the VLM to find the panels, the prompt would be changed to:

    > "Here is a comic page. I have already detected N panels at these bounding box coordinates: `[...]`. For each of these boxes, please analyze its content and provide the character, dialogue, and action descriptions."

*   **Benefit:** This forces the VLM to act as a pure **analyzer** and not a detector, leveraging the strength of the R-CNN model. This is a powerful technique to improve the quality and consistency of the VLM's output.

---

## 5. Deep Research Report (Gemini CLI)

### State-of-the-Art in Object Detection (Alternatives to Faster R-CNN)

The field of object detection has advanced significantly beyond the classic Faster R-CNN architecture. The current state-of-the-art is dominated by two main families of models:

1.  **YOLO (You Only Look Once):** The YOLO family is the industry standard for real-time object detection, offering an excellent balance of speed and accuracy. Recent versions like **YOLOv9** and **YOLO-World** have pushed performance even further. YOLO-World is particularly interesting as it is an "open-vocabulary" model that can detect objects based on text descriptions, which aligns well with the multi-modal nature of this project.

2.  **DETR (DEtection TRansformer):** This family of models uses a Transformer-based architecture, which has revolutionized many areas of deep learning. Models like **RT-DETR** (Real-Time DETR) have become competitive with YOLO in terms of performance while using a different underlying approach. Fine-tuning a DETR model on your panel data could yield very high accuracy.

**Recommendation:** Fine-tuning a pre-trained **YOLOv9** or **RT-DETR** model on your existing panel dataset is the most promising path to creating a more accurate panel detector.

### Visual Prompting and Open-Vocabulary Detection

Your idea to "give rcnn output to vlm with boxes drawn" is a form of **Visual Prompting**. This is an emerging area of research where a model's behavior is guided by inputs beyond just the image itself.

*   **General VLMs:** For a model like the one you are using, providing the bounding boxes in the prompt forces it to focus its analysis on specific regions, turning it from a generalist into a specialist analyzer. This is a valid and powerful technique.

*   **Open-Vocabulary Detectors:** More advanced models like **GroundingDINO** are built specifically for this. You can give them an image and a text prompt like "find the panel where a character is yelling" and they will attempt to draw the bounding box around that specific panel. While this may be too slow for your current pipeline, it represents the future of this technology and is worth keeping in mind for future experiments.

---

## 6. How Object Detection Models Work

Any model that "does boxes"—whether it's the classic Faster R-CNN or a newer model like YOLO or DETR—will give you the panel coordinates by its very nature. You do not need a separate tool to find the coordinates after detection.

The fundamental output of an object detection model is a list of predictions. For each object it finds, it returns several pieces of information:

1.  **The Bounding Box:** A set of four numbers that defines the location and size of the box, typically as `[x_min, y_min, x_max, y_max]` or `[x_center, y_center, width, height]`.
2.  **The Class Label:** The type of object it found (e.g., "panel", "character").
3.  **The Confidence Score:** A number (e.g., 0.0 to 1.0) indicating the model's confidence in its prediction.

In your pipeline, the `predictions.json` file is the direct output of this process. The `bbox` field found in the `annotations` section of that file *is* the set of panel coordinates detected by the `faster_rcnn_calibre.py` script. Therefore, any upgraded or alternative detection model you choose will also provide these coordinates as its primary output.

---

## 7. VLM Cost-Capability Analysis Plan

As you noted, the choice of VLM is a critical trade-off between cost, speed, and the quality of the analysis, especially when processing over a million pages for a hobby project. A systematic benchmark is needed to make an informed decision.

### The Goal

To quantitatively measure the performance and cost of various VLMs available via OpenRouter on a representative sample of your most difficult comic pages.

### The Method: A VLM Benchmark Script

This experiment would require a new script, `tools/benchmark_vlms.py`.

1.  **Create a Test Set:** Using the results from the `Panel Validator UI` (proposed in Section 2), we would create a small, challenging test set of 50-100 pages where we know both the R-CNN and the initial VLM performed poorly.

2.  **Define a VLM List:** The script would contain a list of VLM model identifiers to test, focusing on the lower end of the cost spectrum. Based on your notes, this would include:
    *   `google/gemma-7b-it:free` (Good, cheap baseline)
    *   `meta-llama/llama-3-8b-instruct` (e.g., Llama Maverick was likely a fine-tune of this family)
    *   `google/gemini-flash-1.5`
    *   A cheaper Amazon Titan VLM (when available).
    *   Other high-performers like `Qwen 2.5 VL Instruct` for comparison.
    *   `Qwen2.5-Omni-7B` (7B parameter model from Alibaba's Qwen family)
    *   `SmolVLM-2.2B`, `SmolVLM-500M`, `SmolVLM-256M` (Small, efficient models from Hugging Face)
    *   `Kimi-VL-A3B-Thinking` (Efficient model from Moonshot AI with 2.8B active parameters)
    *   `Gemma 3 (12B, 4B, 1B)` (New family of models from Google)
    *   `MiniCPM-o-2_6` (9B parameter open-source model)

3.  **Run the Benchmark:** The script would iterate through each page in the test set and send it to **each VLM** in the list. It would record several key metrics for each run:
    *   **Success Rate:** Did the model return a valid, parsable JSON?
    *   **Latency:** How long did the API call take?
    *   **Cost:** The script would use OpenRouter's pricing information to calculate the cost of the request based on input/output tokens and any per-image fees (noting that models like Gemma have favorable pricing here).
    *   **Quality (Heuristic):** It would record the number of panels detected and the total amount of text generated as a simple proxy for the richness of the output.

4.  **Generate a Report:** The script would output a `vlm_benchmark_report.csv` file. This report would have one row per VLM, with columns for average success rate, average latency, estimated cost per 1,000 pages, and average panel/text counts. This would provide a clear, data-driven comparison to help you select the most cost-effective model for your needs.

---

## 8. Researcher's Notes

*   **Gemini 4B - can do basics**
    *   fails on some perpetually?
    *   yet to be understood why
    *   perhaps look at fourier analysis?

*   **Gemma 12B - can handle some failures of 4B**
    *   performance of 'free version' via google provider is better than the other providers which keep failing

*   **Mistral 3.1**
    *   character A and character B only generally but can handle gemini failures

*   **Qwen 2.5 VL Instruct failed on same that gemini 4B did**

*   **Gemini Flash 1.5 - can do missing - some null captions and characters**

*   **Gemini Flash 2.5 flash lite - can do missing**
    *   good, but output is 8 times more expensive than gemma 4B - which could run locally
    *   quite a few connection errors with google as the provider
    *   on the last hardest 660 had 1/3 errors and 1/5 json errors

*   **GPT Nano 4.1 says unsupported image type?**
    *   so not as good as google which can handle

*   **Meta Llama 4 Scout**
    *   much better, success on 300 out of 500 images left at the end 
    *   also 0.08/0.3 compared to 0.10/0.40 for Gemini Flash Lite 2.5 - so way better
    *   GMI Cloud provider big problems

---

## 9. First Steps on What to Do

This section outlines a practical workflow based on the latest understanding of the project's data and goals.

### Phase 1: Root Cause Analysis

The primary goal is to understand *why* the VLM and R-CNN panel counts disagree. This requires a diagnostic approach.

1.  **Identify Failures:** The first step is to create a script that reads an analysis CSV (e.g., `calibre_rcnn_vlm_analysis_v2_split0_with_page_types.csv`) and filters it to find all rows where `rcnn_panels` does not equal `vlm_panels`. This produces a definitive list of error cases.

2.  **Adapt the Validator UI:** The `panel_validator_ui.py` tool should be modified for diagnostics. For each failure case, it should:
    *   Display the page image.
    *   Render **all** bounding boxes detected by the R-CNN.
    *   Visually highlight the specific boxes that the VLM failed to provide an analysis for (the "missing" panels).

3.  **Categorize Failures:** A human validator should then categorize the *reason* for the VLM's failure, using buttons like:
    *   **"Bad R-CNN Box":** The R-CNN detected something that isn't a real panel. The VLM correctly ignored it.
    *   **"Good Panel, VLM Failed":** The R-CNN correctly identified a panel, but the VLM failed to analyze it. This points to a VLM or prompt issue.
    *   **"Ambiguous/Complex Panel":** The panel is unusual and may be a legitimate challenge for any model.

This process will generate a `validation_results.csv` containing a categorized log of failure types.

### Phase 2: Acting on the Diagnosis

The results from the root cause analysis will guide the next steps.

#### If the R-CNN is the Primary Problem (High "Bad R-CNN Box" Rate)

This confirms the hypothesis that the R-CNN is "overly aggressive."

*   **Action:** This aligns with **Path A** of the original plan. A script should be used to gather all human-validated "good" panel coordinates from the dataset. This clean data will be used to fine-tune a modern object detector like **YOLOv9 or RT-DETR**, which will replace the current R-CNN model and provide a much cleaner input signal to the VLM.

#### If the VLM is the Primary Problem (High "Good Panel, VLM Failed" Rate)

This indicates the R-CNN is performing adequately, but the VLM is not robust.

*   **Action:** This aligns with the **"VLM Cost-Capability Analysis Plan."** The clear next step is to find a better VLM or a better prompt. The `tools/benchmark_vlms.py` script should be implemented as outlined. By testing various models on a test set of these known failure cases, we can quantitatively determine which model provides the most reliable analysis for the lowest cost.
