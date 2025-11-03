 # Version 2 Framework: Semantic Comic Understanding Pipeline

  Stage 1: Raw Data Ingestion & Foundational Feature Extraction

   * Inputs: Raw Comic Files (CBZ, PDF, EPUB), Page Images, Existing VLM JSONs, R-CNN Detections.
   * Processes:
       * Image Extraction: Convert comic files to individual page images (using create_master_manifest.py).
       * Panel Detection: Run Fast-RCNN on page images to get bounding boxes and compositional features (per page/panel).
       * OCR Text Extraction: Run a VLM (e.g., Qwen2.5-VL-32B) on page images to get raw text. This generates the OCRResult JSONs CoSMo expects.
       * Text Embedding: Convert OCRResult text into textual embeddings (e.g., SentenceTransformer).
   * Outputs:
       * Organized Page Images (root_dir/book_id/image.jpg)
       * OCR JSONs (root_dir/book_id/image.json with OCRResult text)
       * Panel Bounding Boxes & Compositional Features (per page/panel)

  Stage 2: Page Stream Segmentation & Page-Level Embedding (CoSMo Integration)

   * Inputs: Page Images, OCR Text (or embeddings).
   * Model: CoSMo (Multimodal Transformer, trained on your annotated data).
   * Processes: CoSMo processes each page (and its context within the comic stream) to:
       1. Classify Page Type: Assign a semantic label (e.g., narrative, advertisement, back_matter_text, cover_front).
       2. Generate Page Embedding: Produce a rich, multimodal page-level embedding that captures the visual and textual characteristics CoSMo used for
           its classification.
   * Outputs:
       * Page-level Classifications (for all pages).
       * CoSMo Page Embeddings (for all pages): These are valuable queryable embeddings for all page types, including backmatter.
   * Filter for Narrative: Identify narrative pages based on CoSMo's classification. Only these pages proceed to deeper panel-level analysis.

  Stage 3: Domain-Adapted Multimodal Panel Feature Generation

   * Inputs (for `narrative` pages):
       * Panel Image Crops (from R-CNN detections).
       * Panel Text Embeddings (from OCR JSONs).
       * Panel Compositional Features (bounding boxes, relative positions, aspect ratios).
   * Visual Encoder(s):
       * Domain-Adapted Visual Backbone(s): Fine-tune a visual encoder (e.g., SigLIP/ViT, or a ResNet, or a fusion of both) specifically on comic
         book images.
       * Output: Rich Visual Panel Features (e.g., F_vit, F_cnn).
   * Panel-Level Fusion:
       * Fuse Visual Panel Features (potentially from multiple backbones), Textual Embeddings, and Compositional Features into a single,
         comprehensive Multimodal Panel Embedding.
   * Output: A sequence of Multimodal Panel Embeddings for each comic strip.

  Stage 4: Semantic Sequence Modeling (ComicsPAP & Text-Cloze Inspired Transformer)

   * Inputs: Sequence of Multimodal Panel Embeddings for each comic strip.
   * Model: A Transformer encoder (similar to CoSMo's architecture, or the Multimodal-LLM from the Text-Cloze paper, adapted for panel sequences).
   * Tasks/Losses: This model would be trained on tasks that require understanding the relationships between panels:
       * ComicsPAP Tasks: Identifying missing panels, assessing character coherence, visual closure, text closure, caption relevance.
       * Text-Cloze: Predicting missing text within a panel based on context.
       * Reading Order: Refining reading order prediction.
   * Output: Contextualized Semantic Panel Embeddings and a Semantic Strip Embedding (an aggregated representation of the entire comic strip).

  Stage 5: Downstream Queryable Embeddings & Applications

   * Outputs:
       * CoSMo Page Embeddings (for ALL pages): Allows querying for covers, advertisements, back_matter_text (like letters pages!), back_matter_art,
         etc.
       * Semantic Panel Embeddings (for narrative pages): Rich embeddings for individual panels, capturing their content and context within the
         strip.
       * Semantic Strip Embeddings (for narrative pages): Aggregated embeddings for entire comic strips or narrative pages, capturing the overall
         meaning and narrative.
   * Storage: Store these embeddings efficiently in Zarr for fast querying.
   * Applications:
       * Advanced Semantic Search: Query by narrative elements, character actions, plot points.
       * Content Recommendation: Suggest similar comic strips or specific types of backmatter.
       * Automated Summarization: Generate summaries of comic strips.
       * Narrative Analysis: Study pacing, character development, etc.
