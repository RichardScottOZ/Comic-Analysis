## Plan: Developing a Multi-Modal Comic Page Classifier

**Overall Goal:** To develop a robust classifier capable of categorizing individual comic book pages into distinct content types (e.g., Story Page, Cover, Advertising, Backmatter), thereby improving data quality, refining search, and enabling more nuanced analysis within the `CoMix` framework.

**Core Strategy:**
1.  **Data Definition & Annotation:** Establish clear page categories and create a diverse, human-annotated ground-truth dataset.
2.  **Multi-Modal Feature Extraction:** Leverage `ClosureLiteSimple` to extract rich visual, textual, and compositional features for each page.
3.  **Supervised Classifier Training:** Train a classifier on these extracted features to predict page types.

---

**Detailed Planning Steps:**

**Phase 1: Defining Page Types & Annotation Strategy**

*   **Objective:** Establish a comprehensive, well-defined set of page categories and create a ground-truth dataset for training and evaluation.
*   **Steps:**
    1.  **Define Exhaustive Page Type Taxonomy:**
        *   **Covers:** Front Cover (primary), Back Cover, Variant Cover.
        *   **Story Pages:** Multi-Panel Story Page (typical sequential art), Splash Page (single large narrative image).
        *   **Backmatter/Frontmatter (Non-Story, Non-Ad):** Creator Info/Credits/Indicia, Letters Column/Editorials, Interviews/Articles, Sketchbook/Character Designs/Art Process, Table of Contents/Title Page.
        *   **Advertising:** Product Ads (external goods/services), House Ads/Other Comic Ads (internal comic promotion), Previews (short story segments from other comics).
        *   **Blank / Near-Blank Pages.**
        *   **Miscellaneous / Other:** Catch-all for undefined types.
    2.  **Develop Annotation Guidelines:** Create clear, unambiguous instructions, definitions, and visual examples for human annotators to ensure consistent and high-quality labeling.
    3.  **Annotate Dataset:** Manually label a large, representative sample of comic pages from various sources and periods according to the defined taxonomy. This will be a labor-intensive but critical step.

**Phase 2: Multi-Modal Feature Extraction per Page**

*   **Objective:** Extract rich, discriminative feature vectors for each annotated page using the `ClosureLiteSimple` model and associated tools.
*   **Steps:**
    1.  **Generate `E_page`:** For each page, obtain its multi-modal page embedding (`E_page`) using `model.han.panels_to_page` (which integrates `P_flat` from `model.atom`). This will serve as a primary, holistic page feature.
    2.  **Extract Aggregated Panel-Level Features:**
        *   **From `P_flat`:** If a page contains multiple panels, compute summary statistics (e.g., mean, variance, max-pool) across the individual multi-modal fused panel embeddings (`P_flat`).
        *   **From `comp_feats`:** Calculate statistics from compositional features (`comp_feats`) for all panels (e.g., total panel count, average aspect ratio, variance of panel sizes, spatial distribution metrics).
    3.  **Extract Aggregated Textual Features:**
        *   **Keyword Spotting:** Identify presence/absence of indicative keywords (e.g., "Copyright", "Letters", product names, comic titles).
        *   **Text Density:** Metrics like word count, character count, or the ratio of text area to image area.
        *   **Concatenated Text Embeddings:** Average or concatenate the text embeddings of all panels, or generate a single embedding for the entire page's concatenated text.
    4.  **Extract Complementary Visual Features (if needed):** If `E_page` alone doesn't capture certain visual aspects sufficiently, consider direct image features like dominant color palettes, image entropy, or presence of large logos/borders.
    5.  **Integrate External Metadata:** Review `Comic_Database_Research_Report.md` and any available database schemas for features that could aid classification (e.g., "is_first_page", "is_last_page", "story_arc_start", "is_collection" flag).

**Phase 3: Classifier Architecture & Training**

*   **Objective:** Train a robust supervised classifier to predict page types based on the extracted multi-modal features.
*   **Steps:**
    1.  **Construct Feature Vector:** Concatenate all extracted features (multi-modal embeddings, compositional aggregates, textual aggregates, visual specifics, metadata) for each page into a single, comprehensive input vector.
    2.  **Dataset Split & Balancing:** Divide the annotated dataset into training, validation, and test sets. Address potential data imbalance (e.g., many story pages, few letters columns) using techniques like stratified sampling or class weighting/oversampling.
    3.  **Select Classifier Architecture:**
        *   **Baseline Models:** Start with simpler, interpretable models like Logistic Regression, Support Vector Machines, Random Forests, or XGBoost.
        *   **Neural Networks:** For more complex feature interactions, train a Multi-Layer Perceptron (MLP) on the concatenated feature vector.
        *   **Advanced (Optional):** If the sequential arrangement of panels is critical for certain page types (e.g., distinguishing story pages from collections of sketches), consider incorporating RNN/Transformer layers over the sequence of `P_flat` embeddings before feeding into the classifier.
    4.  **Train & Evaluate:** Train the chosen classifier(s) and evaluate performance using appropriate multi-class metrics (accuracy, precision, recall, F1-score, confusion matrix per class) on the test set.

**Phase 4: Addressing Specific Challenges & Refinements**

*   **Objective:** Optimize classification for particularly challenging distinctions and ensure robust, semantically meaningful performance.
*   **Key Challenges & Strategies:**
    *   **Covers/Back Covers:** Feature prominent visual elements, unique compositional attributes (e.g., prominent title/logo, lack of RPP-relevant structure), and textual cues (cover taglines, indicia). Leverage "is_first_page", "is_last_page" metadata.
    *   **Creator Info / Legal Pages:** Prioritize textual features (heavy keyword spotting for "Copyright," "Written by," "Publisher"), high text density, and specific compositional traits (single large text block, minimal art).
    *   **Advertising Pages:** Emphasize visual features (branding, vibrant colors, product images, bold fonts), compositional layouts (splash-like, distinct visual clusters), and commercial textual keywords ("Buy now!").
    *   **Previews (e.g., "Revival" in an "Invincible" comic):** This is highly complex. Strategies could include:
        *   **Cross-Comic Title Matching:** Extracting potential comic titles from text within the preview and comparing against a database of known comic titles.
        *   **Publisher/Creator Consistency:** Detecting a change in common publisher or creator names.
        *   **Subtle Art Style Shifts:** (Highly advanced research area) Detecting a sudden change in primary artistic style.
    *   **Sketches/Character Designs/Art Process Pages:** Focus on visual characteristics (uncolored, line art, rougher textures), specific textual cues ("Sketch," "Concept"), and non-narrative panel arrangements.
    *   **Data Imbalance:** Actively manage class imbalance in training (e.g., oversampling minority classes, undersampling majority classes, using class-weighted loss functions).
    *   **Robustness to VLM/OCR Errors:** Ensure textual features are designed to be somewhat robust to inevitable errors from VLM/OCR processing.

**Expected Outcome:** A highly accurate comic page classifier that can significantly enhance content organization, search specificity, and provide a richer understanding of comic book structure.