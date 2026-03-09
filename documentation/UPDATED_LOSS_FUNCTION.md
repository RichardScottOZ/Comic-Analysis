# Updated Loss Function: Stage 3 Training

This document details the corrected loss function architecture for Stage 3 (Domain-Adapted Panel Feature Generation). It addresses the "Mode Collapse" issue encountered in early experiments and explains the theoretical justification for the three-part objective.

## 1. The Core Problem: Mode Collapse

In the initial implementation, the Contrastive Loss was formulated as:
> *Maximize the similarity between all panels on the same page.*

**Mathematical Flaw:**
Without negative pairs (panels from *other* pages), the optimal solution to this objective is trivial:
*   Output a constant vector `[1, 1, 1...]` for every single panel.
*   Similarity becomes `1.0` (Perfect).
*   Loss becomes minimum.

**Result:** The model "collapsed," producing identical embeddings for every input image and text.

## 2. The Solution: InfoNCE Contrastive Loss

We have replaced the naive "Positive Only" loss with a standard **InfoNCE (Noise Contrastive Estimation)** formulation.

### The Objective
> *Maximize similarity between panels on the SAME page (Positives), while MINIMIZING similarity to panels from DIFFERENT pages (Negatives).*

### Formula
```math
L_{contrastive} = - \log \left( \frac{\exp(\text{sim}(p_i, p_j) / \tau)}{\sum_{k  \text{Batch}} \exp(\text{sim}(p_i, p_k) / \tau)} \right)
```
Where:
*   $p_i, p_j$: Panels from the same page.
*   $p_k$: All panels in the batch (including those from other pages).
*   $	au$: Temperature scaling (0.07).

### Why this works
To minimize this loss, the model **cannot** output a constant vector. If it did, the similarity to negatives ($p_k$) would be 1.0, making the denominator large and the loss high. It is forced to learn features that make Page A distinct from Page B.

## 3. Balancing Objectives (The "Why")

A common critique is: *"Why force panels on the same page to be similar? They represent different moments!"*

We use a **Multi-Objective Strategy** to balance Global Context with Local Detail.

### A. Contrastive Loss (Global Context)
*   **Goal:** Group panels by "Scene/Page".
*   **Intuition:** Panels on the same page share art style, color palette, lighting, and narrative context.
*   **Role:** Acts as the "Glue" that organizes the embedding space.

### B. Reconstruction Loss (Local Detail)
*   **Goal:** Predict a masked panel's features from its neighbors.
*   **Formula:** MSE between `Predicted_Embedding` and `Actual_Embedding`.
*   **Intuition:** To predict Panel 2 from Panel 1 & 3, the model must understand the *sequence* and the *specifics* of the missing moment.
*   **Role:** Prevents the "blurring" effect of the contrastive loss. It forces embeddings to retain fine-grained details unique to that specific panel.

### C. Modality Alignment Loss (Semantic Bridge)
*   **Goal:** Ensure the Image Embedding matches the Text Embedding for the same panel.
*   **Role:** Grounds the visual features in semantic meaning (dialogue/narration).

## 4. Final Loss Equation

```python
Total_Loss = (1.0 * L_Contrastive) + (0.5 * L_Reconstruction) + (0.3 * L_Alignment)
```

*   **1.0 Contrastive:** Strongest signal, sets the global structure.
*   **0.5 Reconstruction:** Ensures individual panels remain distinct within their page cluster.
*   **0.3 Alignment:** Tries to align text/vision, but weighted lower initially due to noisy OCR data.

---
*Updated: Jan 2026*
