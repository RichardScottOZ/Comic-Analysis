# **A Multimodal Methodology for Paratextual and Non-Narrative Content Classification in Digital Comic Archives**

## **Section I: Conceptual Framework: Defining the Paratextual Threshold in Digital Comics**

The automated analysis of large-scale digital comic book archives presents a formidable challenge, extending beyond the mere recognition of panels, characters, and text. A fundamental, yet often overlooked, aspect of this challenge is the systematic identification and classification of non-narrative and paratextual content. This content, far from being extraneous filler, is integral to the comic book as a material and cultural object. A robust methodology for its classification is not merely a technical prerequisite for downstream narrative analysis but a crucial step toward a computational understanding of the medium itself. This report outlines such a methodology, grounded in a multimodal machine learning pipeline that is informed by the theoretical framework of paratextuality.

### **Introduction to Paratextuality**

The theoretical foundation for this work is derived from literary theorist Gérard Genette's concept of the "paratext".1 Genette defines the paratext as the collection of ancillary materials that surround and mediate a primary text, creating what he terms a "threshold" or a "zone of transaction" between the reader and the narrative.2 These materials—which include titles, prefaces, author names, and even the physical design of the book—are not part of the text proper but are indispensable to its presentation and reception. As Genette argues, "the paratext is for us the means by which a text makes a book of itself and proposes itself as such to its readers, and more generally to the public".2 This framework provides a powerful lens for analyzing the complete communicative function of a published work, moving beyond the isolated narrative to consider its packaging and framing.

### **Applying Paratextuality to Comics**

The medium of comic books is exceptionally rich in paratextual content, making it a particularly fertile ground for this mode of analysis. A typical periodical comic book is a complex artifact that integrates the primary narrative with a host of other materials.3 Front covers, creator credits, editorial columns, publisher and third-party advertisements, and letters-to-the-editor pages are all standard components.2 These elements are not incidental; they are intrinsic to the "book-ness" of the comic and play a significant role in shaping its economic, artistic, and cultural value.2 The cover, for instance, functions as a vital marketing tool and an interpretive gateway, which may or may not have a direct representational relationship with the interior narrative content.2 Similarly, letters columns represent a "privileged space of readerly communication," fostering a sense of community and creating a feedback loop between creators and their audience.4 Therefore, classifying these pages is not simply a task of filtering out "non-story" content; it is an act of computational paratextual analysis, enabling researchers to study how these elements frame the narrative and manage public reception.

### **The Digital Paratext and Serial Publications**

While Genette's original formulation focused on printed literary works and explicitly excluded serial publications, subsequent scholarship has argued for the framework's expansion.1 In the context of serialized media like newspaper comic strips or monthly comic books, paratext becomes even more significant.5 It serves to maintain continuity, manage reader expectations between installments, and cultivate a dedicated readership over time. Letters columns, in particular, evolve into a site of "paratextual negotiation," where the meaning and direction of the ongoing narrative are actively discussed and shaped by the fan community.4

The transition to digital archives introduces further complexity. In the context of digitized print comics, the paratextual elements of the original artifact are preserved as scanned pages. However, in born-digital comics like webtoons, the concept of paratext expands to include the technological platform, user interface, recommendation algorithms, and community interaction features like comment sections.7 While this report focuses on digitized print media, the proposed methodology for identifying semantic shifts between page types provides a foundational model that could be extended to these more dynamic digital paratexts.

The concept of the paratext as a "threshold" offers a powerful heuristic for developing a computational model. A threshold is, by definition, a point of transition, implying a discernible change in form or function. In comics, this transition between the narrative (text) and its surrounding materials (paratext) is communicated through deliberate design choices. An advertisement page employs a distinct visual language—typography, color palette, layout, and imagery—that is designed to differentiate it from a story page. A letters column has a characteristic multi-column, text-dense layout. This theoretical framing translates directly into a machine learning objective: the model should be trained to detect high-gradient changes in multimodal feature distributions within the sequential stream of pages. By treating the problem as one of boundary detection or segmentation, informed by paratextual theory, the methodology moves beyond a brute-force, page-by-page classification to a more nuanced, context-aware approach that mirrors the reader's own cognitive process of recognizing a shift from story to non-story content.

## **Section II: Foundational Pre-Classification: Distinguishing Publication Formats**

The architecture of a comic book—the types of paratextual content it contains and where that content is located—is fundamentally dependent on its publication format. A monthly, 22-page "single issue" periodical has a different structure and purpose than a 150-page "trade paperback" collection. Therefore, a critical preliminary step in any automated classification pipeline is to determine the format of the digital comic file as a whole. This section details a robust methodology for this binary classification task, which serves as an essential contextual input for the more granular page-level analysis that follows.

### **Defining the Formats**

The two primary formats in modern American comics publishing are single issues and trade collections (including trade paperbacks and hardcovers).8

* **Single Issues:** These are the episodic, periodical installments of a comic book series, typically ranging from 20 to 40 pages in length.10 They are characterized by their "magaziney" quality, often printed on thinner paper stock, and contain contemporary advertisements relevant to their publication date. Their paratextual content commonly includes a single front cover, a credits/indicia block on an early page, and back matter such as a letters-to-the-editor column.2  
* **Trade Collections:** These volumes collect a series of single issues, typically comprising a complete story arc.9 Consequently, they are significantly longer, often exceeding 100 pages. They are generally printed on higher-quality paper and are less likely to contain contemporary third-party advertisements. Their paratextual structure is different; they often include the covers of the original single issues as chapter dividers and feature unique back matter, such as character designs, script excerpts, and creator interviews or sketchbooks, which are rare in single issues.11

### **Feature Engineering for a Binary Format Classifier**

Given these distinct characteristics, a highly accurate binary classifier can be constructed using a small set of engineered features extracted from the document as a whole. These features can be computed efficiently before initiating the more resource-intensive, page-by-page analysis. An effective model for this task could be a gradient boosting classifier (e.g., XGBoost), a Support Vector Machine (SVM), or a small, fully connected neural network.

The following features are proposed:

1. **Total Page Count:** This is arguably the most powerful single heuristic. The page count distributions for single issues and trade collections are largely distinct. A single issue rarely exceeds 48 pages, while a trade collection is almost always over 90 pages. A simple statistical analysis of page counts from a labeled dataset can establish clear probabilistic thresholds.  
2. **Cover Barcode Analysis:** The barcode on the front cover provides a definitive signal. Single issues use a Universal Product Code (UPC), which is often accompanied by text identifying it as a "Direct Edition" or "Newsstand Edition." In contrast, trade paperbacks and hardcovers, being books, use an International Standard Book Number (ISBN). A targeted computer vision module can be applied to the first page (assumed to be the cover) to locate the barcode region. Subsequently, an Optical Character Recognition (OCR) engine can extract the numbers and surrounding text. The presence of the term "ISBN" or a number format consistent with ISBN-13 is a near-certain indicator of a trade collection.  
3. **Internal Cover Repetition:** Trade collections frequently reproduce the covers of the individual issues they contain, often as chapter breaks.12 This creates a pattern of near-duplicate images within the document stream. This pattern can be detected by generating a perceptual hash (e.g., pHash, dHash) or a more robust deep learning-based image fingerprint for every page. A high count of pages with very low Hamming distance between their hashes is a strong indicator of a trade collection. This technique leverages research in near-duplicate image detection.13  
4. **Statistical Sampling of Page Types:** A lightweight, probabilistic assessment can be made by sampling a small number of pages from the latter portion of the book (where back matter resides). A simplified version of the main page classifier can be run on these samples. The detection of even a single page with features strongly indicative of a sketchbook section (e.g., sparse, uncolored line art, no panel structure) significantly increases the likelihood that the document is a trade collection.12 Conversely, detecting a classic multi-column letters page increases the likelihood of it being a single issue.

This pre-classification stage is more than a simple filtering mechanism; it functions as a crucial source of contextual information for the main page classification pipeline. The output of this binary classifier should not be a hard decision (e.g., "this is a trade") but rather a probability (e.g., $P(\\text{format}=\\text{Trade}) \= 0.95$). This probability can then be passed as an additional feature to the main page-stream segmentation model for every page in the document. This allows the primary model to learn conditional probabilities and resolve ambiguities more effectively. For instance, a page containing sparse line art might be ambiguous on its own. However, if the model also knows there is a 95% chance the document is a trade paperback, the probability of that page being correctly classified as back\_matter\_art increases dramatically. This transforms the pre-classification step from a disconnected preliminary process into an integrated Bayesian prior, creating a more robust and context-aware system.

## **Section III: Core Architecture: A Multimodal Pipeline for Page-Stream Segmentation and Classification**

With the publication format established as a contextual prior, the core task is to classify each page within the digital comic. This requires an architecture capable of understanding the complex interplay of visual layout and textual content, not just on a single page but across the entire sequence of the document. This section surveys the state-of-the-art in Document Layout Analysis (DLA) and proposes a specific, high-performance multimodal Transformer architecture tailored for the comic book domain.

### **A Comparative Survey of Document Layout Analysis Architectures**

The field of Document AI has produced several dominant architectural paradigms for DLA, each with distinct strengths and weaknesses.15

* **Convolutional Neural Networks (CNNs):** Architectures like Faster R-CNN, YOLO, and Mask R-CNN treat DLA as an object detection or instance segmentation problem.17 They excel at identifying and localizing visually distinct regions such as text blocks, images, tables, and, in the comics domain, panels and speech balloons.18 Their strength lies in powerful visual feature extraction. However, they are inherently local in scope; they do not natively model the relationships between detected objects or the overall logical structure of the page.16  
* **Graph Neural Networks (GNNs):** This paradigm explicitly addresses the relational shortcomings of CNNs by representing a document page as a graph.19 Layout elements (detected via a preliminary step, often using a CNN) become nodes, and their spatial or logical relationships (e.g., "is below," "is adjacent to") become edges. GNNs then operate on this graph structure to classify nodes or predict relationships.20 Models like GLAM (Graph-based Layout Analysis Model) have achieved competitive performance while being significantly more lightweight than large Transformer models.19 Their primary advantage is the explicit modeling of page structure. However, they are less naturally suited to modeling the sequential, page-to-page context of a multi-page document.  
* **Transformers:** Vision and multimodal Transformers represent the current state-of-the-art in DLA.21 Models like LayoutLM, DiT (Document Image Transformer), and VGT (Vision Grid Transformer) have demonstrated superior performance by leveraging the self-attention mechanism to capture long-range dependencies among document elements.21 They can process a document as a flattened sequence of image patches and/or text tokens, allowing them to learn complex relationships between visual and textual information simultaneously. Their ability to handle long sequences makes them uniquely suited for modeling not just the layout within a page but also the context across multiple pages.

| Architecture | Key Models | Strengths | Weaknesses | Suitability for Comic Page Stream Segmentation |
| :---- | :---- | :---- | :---- | :---- |
| **CNN-based Object Detection** | YOLOv8, Faster R-CNN | Excellent for detecting visually distinct elements (panels, balloons, figures). Computationally efficient for inference. | Lacks understanding of global page structure and relationships between elements. Not inherently sequential. | **Low.** Suitable for preliminary tasks (e.g., panel extraction) but insufficient for page-level classification that depends on overall layout and sequence. |
| **Graph Neural Network (GNN)** | GLAM, Paragraph2Graph | Explicitly models spatial and logical relationships between layout elements on a page. Computationally efficient. | Less effective at modeling sequential context across multiple pages. Requires a separate step to construct the graph. | **Medium.** Strong for classifying complex single-page layouts but requires modification to incorporate sequential information from the page stream. |
| **Sequential Transformer** | LayoutLMv3, DiT, CoSMo | State-of-the-art performance. Natively handles both visual and textual modalities. Self-attention captures long-range dependencies within and across pages. | High computational cost for training and inference. Requires large datasets. | **High.** The ideal architecture. The problem can be framed as a sequence labeling task, which directly maps to the Transformer's strengths. |

### **The Page Stream Segmentation (PSS) Paradigm**

The most effective way to frame the comic book classification problem is as **Page Stream Segmentation (PSS)**.22 This approach, formalized in recent work on the CoSMo model, reframes the task from classifying independent pages to segmenting the entire sequence of pages (the "stream") into coherent, semantically meaningful units like stories, advertisements, or back matter sections.23 This is fundamentally a multi-class sequence labeling problem, where the classification of a given page is dependent not only on its own content but also on the content of its neighboring pages. This formulation is critical for comics, where narrative and paratextual sections often span multiple pages, and the boundaries between them are the primary objects of interest.

### **Proposed Architecture: A CoSMo-Inspired Multimodal Transformer**

Based on this analysis, the proposed core architecture is a multimodal Transformer encoder modeled after CoSMo.22 This model is designed to integrate visual, textual, and sequential information to produce robust page classifications. The pipeline consists of three main stages:

1. **Visual Feature Extraction:** Each page in the comic book is processed by a powerful pre-trained vision model that acts as a feature extraction backbone. The CoSMo model uses a frozen SigLIP backbone, which is chosen for its strong performance in generating robust general-purpose image embeddings.22 This process converts each page image into a high-dimensional vector that captures its visual layout, style, and content.  
2. **Textual Feature Extraction:** A state-of-the-art Optical Character Recognition (OCR) engine is applied to each page to extract all textual content. For optimal performance, a context-aware model is preferred; the CoSMo paper notes the use of Qwen2.5-VL-32B for this purpose, which can identify both printed and handwritten text within its spatial context.23 The extracted raw text is then tokenized and passed through a pre-trained language model (e.g., a BERT variant) to generate a corresponding textual embedding for the page.  
3. **Multimodal Fusion and Sequential Encoding:** For each page, the visual embedding and the textual embedding are concatenated to form a single, fused multimodal vector. The entire comic book is thus represented as a sequence of these vectors. This sequence is then fed into a Transformer encoder. The self-attention layers of the Transformer allow the model to weigh the importance of every other page in the sequence when making a prediction for a specific page. This enables it to learn the sequential patterns characteristic of comic books—for example, that a credits\_indicia page is typically followed by the start of the narrative, or that a block of advertisement pages often appears in the middle of a book.  
4. **Classification Head:** Finally, a linear classification layer followed by a softmax function is applied to the output embedding for each page from the Transformer. This layer projects the learned representation into the space of possible page labels, outputting a probability distribution over the predefined classes (e.g., narrative, advertisement, cover\_front, etc.) for every page in the stream.

A crucial finding from the development of the CoSMo model is the relative importance of the two modalities. Experiments with both vision-only and multimodal variants demonstrated that visual features are dominant for determining the macro-structure of the page stream (e.g., identifying the large, contiguous block of story pages).22 However, the inclusion of textual features was shown to be critical for resolving challenging ambiguities, such as distinguishing a text-heavy story page from a text-heavy essay in the back matter.24 This indicates that the system learns a hierarchy of cues: vision provides the coarse segmentation, while text provides the fine-grained semantic disambiguation needed for high-accuracy classification.

## **Section IV: Feature Extraction and Classification of Specific Non-Narrative Content**

While a powerful Transformer architecture can learn relevant features automatically, a successful implementation depends on understanding the distinct multimodal characteristics of each page category. This knowledge informs data annotation, model debugging, and the potential for incorporating specialized sub-models. This section details the specific visual and textual features that serve as the strongest indicators for each target page type.

### **A. Covers and Credits/Indicia Pages**

**Front Covers:** The classification of front covers is primarily a visual task. They are defined by a unique layout that is distinct from any interior page.

* **Visual Features:** The most salient feature is a single, dominant piece of artwork that often fills the entire page (a "splash" image). Other key indicators include the presence of a prominent title logo, publisher logos (e.g., Marvel, DC), the Comics Code Authority seal (on older comics), and a barcode block, typically in a lower corner. An object detection model pre-trained on these elements can identify covers with high precision.

**Credits/Indicia Pages:** These pages contain publication and copyright information and are characterized by dense, formulaic text.

* **Visual Features:** The indicia often appear as a small, dense block of text with a small font size, typically located at the bottom of the first interior page or on a dedicated credits page.25 The layout is often non-narrative, lacking panels or large images. Research into detecting small and contextual text blocks is highly relevant here.26  
* **Textual Features:** This is a classic keyword-spotting task. After OCR, the text can be searched for a predefined dictionary of keywords that are almost exclusively found in indicia: "Publisher," "Editor-in-Chief," "Writer," "Artist," "Penciler," "Inker," "Colorist," "Letterer," as well as legal terms like "Copyright ©," "LLC," "Inc.," and mailing address information. Google's Cloud Vision API, specifically its DOCUMENT\_TEXT\_DETECTION mode, is optimized for dense text and can provide the structured output (page, block, paragraph) needed for this analysis.28

### **B. Advertisement Detection**

Detecting advertisements in comics can leverage extensive research on ad detection in magazines, which has shown the efficacy of deep learning approaches.29

* **Visual Features:** Ads often break the established visual grammar of the comic. Key indicators include the use of product photography instead of illustration, different color palettes and saturation levels, unique and often chaotic typographic layouts, and the prominent display of corporate logos and brand names.29 CNN-based models are particularly effective at learning these stylistic differences.31  
* **Textual Features:** The language of advertising is distinct from narrative dialogue. OCR'd text from ads will contain product names, slogans, pricing information ($), website URLs, and explicit calls to action (e.g., "On sale now\!", "Visit us at..."). A combination of keyword spotting and NLP models trained to detect persuasive language can be highly effective. Multimodal models that fuse these visual and textual cues have proven successful in related tasks like fake advertisement detection.32

### **C. Back Matter Analysis: Letters, Interviews, and Process Art**

Back matter is a diverse category of supplemental content, typically found at the end of a comic.

**Letters Columns and Interviews:** These pages are text-centric and can be identified through their distinct layout and conversational structure.

* **Layout Features:** These pages often use a multi-column layout, which is uncommon for narrative pages. A Q\&A format is a strong indicator, identifiable by visual cues like bolded "Q:" and "A:" prefixes or distinct formatting for questions and answers. Letters columns are characterized by salutations ("Dear Marvel," "To the Editor:") and sign-offs.  
* **NLP-Based Features:** The most sophisticated approach is to use **Dialogue Act (DA) recognition**. Research on the COMICORDA dataset provides a direct blueprint for this task.25 The process involves (1) using an object detector like YOLOv8 to segment individual text blocks, (2) applying OCR to extract the text, and (3) using a Transformer-based classifier (e.g., BERT fine-tuned on a dialogue corpus) to classify each sentence or utterance. A page with a high frequency of Inform, Question (specifically Yes\_No\_Question, Wh\_Question), and Answer dialogue acts is very likely an interview or a letters column.25

**Sketches and Process Art:** This content, common in trade collections, is almost entirely visual.

* **Visual Features:** The defining characteristics are stylistic. These pages feature high-frequency, uncolored line art (pencil sketches or inks), a sparse layout often devoid of panel borders or backgrounds, and an absence of speech balloons and captions. The artwork may include construction lines, annotations, or multiple character poses on a single page.  
* **Model Approach:** Techniques from Sketch-Based Image Retrieval (SBIR) are applicable here.34 A CNN can be trained to distinguish sketches from finished, colored artwork. Low-level image features such as high edge density, a bimodal color histogram (peaking at black and white), and texture analysis can serve as strong indicators. The absence of detected panels or speech balloons (from a preliminary segmentation step) further reinforces this classification.

The following table summarizes the key multimodal features that can be engineered or learned by a model to distinguish these paratextual page types.

| Page Category | Key Visual Features | Key Textual / NLP Features |
| :---- | :---- | :---- |
| **Front Cover** | Single dominant "splash" image; prominent title logo; publisher logos; barcode. | Minimal text; title, creator names, issue number. |
| **Credits/Indicia** | Small, dense block of uniform text; small font size; often at bottom of first/second page. | Keyword spotting: "Publisher," "Editor," "Writer," "Artist," "Copyright ©," "LLC," mailing address. |
| **Advertisement** | Use of photography; distinct color palettes and typography; corporate logos; non-panel layouts. | Brand names; product names; pricing ($); slogans; URLs; calls to action ("On sale now\!"). |
| **Letters Column** | Multi-column text layout; repeating salutations and sign-offs. | Dialogue Act recognition: high frequency of Inform and Opinion. Keyword spotting: "Dear Editor," "Next Issue:". |
| **Interview** | Q\&A format (e.g., bolded "Q:"/"A:"); multi-column text layout. | Dialogue Act recognition: high frequency of Question and Answer patterns. |
| **Sketch/Process Art** | Uncolored line art (pencils/inks); sparse layout; no panel borders or backgrounds; construction lines. | Minimal to no text, possibly handwritten annotations. |
| **Preview** | Panel-based comic layout; may be preceded by a "Next Issue" or preview splash page. | Semantic shift: sudden, sustained change in character names, locations, and narrative terms. |

## **Section V: The Preview Problem: Differentiating Embedded Narratives**

Perhaps the most nuanced challenge in comic book page classification is the "preview problem": distinguishing an embedded preview of a different comic series from the main narrative of the book. A preview often mimics the visual grammar of the main story—it is composed of panels, speech balloons, and sequential art—making it difficult for a classifier that relies solely on page-level layout features to differentiate. Solving this problem requires moving beyond static page analysis to a model that understands narrative context and discontinuity.

### **Framing as Narrative Discontinuity and Story Boundary Detection**

The key insight is that while a preview exhibits visual *continuity* with the main story, it represents a complete *narrative discontinuity*. The characters, setting, and plot are abruptly replaced. Therefore, the task can be framed as a problem of **story boundary detection**, a well-studied area in other media. In video analysis, this is analogous to shot boundary detection, where algorithms detect cuts by identifying sudden, large changes in visual features like color histograms and motion vectors between frames.36 In natural language processing, this is known as text segmentation or topic modeling, where shifts in vocabulary and semantic content are used to identify the boundaries between different sections of a document.37 For comics, a successful approach must be multimodal, detecting a simultaneous shift in both visual and textual streams. The Page Stream Segmentation (PSS) framework is perfectly suited for this, as a preview is simply another type of semantic segment whose beginning and end must be identified.22

### **Proposed Techniques for Detecting Previews**

A multi-pronged approach, combining several signals, is necessary to robustly identify these embedded narratives.

1. **Semantic Shift in Textual Content:** This is the most reliable and powerful signal. By applying OCR to every page in the document, the system can analyze the textual content of the narrative as a sequence. A sudden and sustained change in the vocabulary used in dialogue and caption boxes is a strong indicator of a new story. This can be implemented in several ways:  
   * **Named Entity Recognition (NER):** Track the names of characters and locations mentioned on each page. An abrupt shift where the primary cast of the main story disappears and a new set of names appears is a clear boundary marker.  
   * **Topic Modeling/TF-IDF Analysis:** On a sliding window of several pages, compute the term frequency-inverse document frequency (TF-IDF) vectors or run a topic model (like Latent Dirichlet Allocation). A high cosine distance between the vectors of consecutive windows indicates a significant topic shift, and thus a potential story boundary.  
2. **Visual Style and Content Change:** While often more subtle than the textual shift, a change in artistic style can be a valuable signal, especially if the preview is drawn by a different creative team.  
   * **Art Style Analysis:** The same deep visual features used for artist classification can be employed here.39 The feature vector for each page can be extracted from a pre-trained vision model. A clustering algorithm or a change-point detection model can then be run on the sequence of these vectors to identify points where the visual style changes significantly.  
   * **Character Re-identification:** Advanced models can be trained to recognize and track specific characters. The disappearance of the main story's protagonists and the appearance of new, unrecognized characters provides a strong visual cue for a narrative break.  
3. **Explicit Boundary Markers:** Often, previews are not seamlessly integrated. They are frequently preceded by a full-page splash that explicitly functions as a paratextual marker. This page might contain the cover art of the previewed comic, along with text such as "Coming Soon\!", "A Special Preview Of:", or "In the Next Issue...". These pages can be identified by the main page classifier as a distinct class (e.g., preview\_splash), which then signals to the sequence model that a narrative boundary is imminent.  
4. **Cross-Document Content Fingerprinting:** This is the most definitive, albeit computationally intensive, technique, and it is particularly suited for large archival contexts. The core idea is that a preview in one comic is, by definition, a duplicate of the primary narrative content in another comic.  
   * **Methodology:** During an initial indexing phase of the entire archive, a robust perceptual hash or deep feature vector (a "fingerprint") is generated for every single page.41 These fingerprints are stored in a database optimized for fast similarity search.  
   * **Detection:** When analyzing a given book, if the system encounters a potential story boundary, it can query the fingerprint database with the hashes of the subsequent pages. If this sequence of pages returns a strong match to a sequence at the *beginning* of a different comic file in the archive, the system can confirm with very high confidence that it has identified a preview.13  
   * **Implication:** This approach transforms the preview problem from one of intra-document inference to one of inter-document verification. It highlights the limitation of analyzing each digital comic file in isolation. A truly robust archival system must leverage the relationships between documents. By creating a library-scale index of page content, the system can resolve ambiguities within one document by referencing the content of another, effectively using the entire archive as its own ground truth.

## **Section VI: Data and Annotation Strategy: Building a Ground Truth for Comic Archives**

The performance of any supervised machine learning model is fundamentally dependent on the quality and relevance of its training data. While general-purpose datasets for Document Layout Analysis (DLA) are valuable for pre-training, the unique visual language and structural conventions of comic books necessitate the creation of a specialized, domain-specific dataset to achieve state-of-the-art performance. This section surveys existing relevant datasets and proposes a comprehensive strategy for annotating a new corpus for the task of paratextual page classification.

### **Survey of Existing Datasets**

A review of publicly available datasets reveals a landscape of resources that are useful but not perfectly suited for the specific multi-label page stream segmentation task at hand.

* **General DLA Datasets:** These are excellent for pre-training the visual backbone of a model to learn fundamental layout analysis.  
  * **PubLayNet:** A massive dataset with over 360,000 document images derived from scientific articles on PubMed Central.43 Its annotations are automatically generated and limited to five classes: text, title, list, figure, and table.45 Its scale is its primary strength, but its domain (scientific papers) and limited label set represent a significant domain gap.46  
  * **DocLayNet:** A more recent dataset containing over 80,000 pages from diverse sources like financial reports, patents, and manuals.47 Crucially, it is human-annotated with a more detailed set of 11 labels (e.g., Caption, Footnote, Page-header).48 Its detailed annotation guidelines and measurement of inter-annotator agreement provide an excellent model for best practices in creating a new dataset.48  
* **Comics-Specific Datasets:** These datasets are essential for fine-tuning and evaluation, as they capture the specific visual features of the comics medium.  
  * **Manga109:** A foundational dataset for manga analysis, comprising 109 full manga volumes with annotations for panels, text, faces, and characters.18 Recent work has augmented it with pixel-level segmentation masks, making it highly valuable for training segmentation models.50 However, its focus is on Japanese manga, which has different structural conventions and reading order than Western comics.  
  * **eBDtheque:** Another widely used dataset in academic comics research, focusing on Franco-Belgian comics (bandes dessinées).33  
  * **CoMix and ComicsPAP:** These are newer, multi-task benchmarks designed to push research towards deeper narrative and multimodal understanding, such as character re-identification and dialogue generation.51 While not directly targeting page-type classification, they signal the growing need for more richly annotated comics data.  
  * **CoSMo Dataset:** This is the most directly relevant existing resource. Created specifically for the Page Stream Segmentation (PSS) task in American comics, it contains 430 manually annotated books (20,800 pages) with labels aligned to the comprehensive metadata from the Grand Comics Database (comics.org).22 This dataset provides the ideal template for the annotation schema and task definition.

| Dataset | Domain | Size | Annotation Type | Key Labels | Strengths & Weaknesses for This Project |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **PubLayNet** | Scientific Articles | \>360k pages | Bounding Box (auto-generated) | text, title, list, figure, table | **Strength:** Massive scale for visual pre-training. **Weakness:** Significant domain gap; limited and noisy labels. |
| **DocLayNet** | Business/Technical Docs | \~81k pages | Bounding Box (human-annotated) | 11 classes (Caption, Footnote, etc.) | **Strength:** High-quality human annotations; diverse layouts; excellent model for annotation guidelines. **Weakness:** Domain is not comics. |
| **Manga109** | Japanese Manga | 109 volumes | Bounding Box, Segmentation Mask | panel, text, face, character | **Strength:** Large-scale, high-quality, comic-specific annotations. **Weakness:** Focus on manga, not Western comics. |
| **CoSMo Dataset** | American Comics | 430 books (\~21k pages) | Page-level labels | story, advertisement, text insert, etc. | **Strength:** Directly addresses the PSS task; correct domain. **Weakness:** May not be publicly available or may have a label set that needs expansion for this project's specific goals. |

### **Proposed Annotation Schema and Best Practices**

Given the limitations of existing datasets, the creation of a new, custom-annotated corpus is recommended. The following methodology ensures the creation of a high-quality ground truth dataset.

1. **Corpus Selection:** Acquire a diverse corpus of digital comics (.cbz or .cbr files), ensuring a balanced representation of different publishers (e.g., Marvel, DC, Image), eras (e.g., Silver Age, Modern Age), and, most importantly, both single\_issue and trade\_collection formats.  
2. **Annotation Schema:** A two-tiered annotation schema is proposed.  
   * **Tier 1: Document-Level Annotation:** Each file in the corpus receives a single metadata tag.  
     * format: single\_issue or trade\_collection.  
   * **Tier 2: Page-Level Annotation:** Each page within a file receives a single, mutually exclusive label corresponding to its primary function. The proposed label set, derived from the user query and the PSS task, is:  
     * cover\_front: The main front cover of the publication.  
     * narrative: A page containing sequential art that is part of the main story.  
     * advertisement: A page primarily dedicated to a third-party or publisher advertisement.  
     * credits\_indicia: A page containing the main credits and publication information.  
     * back\_matter\_text: Text-heavy back matter, including letters columns, interviews, and essays.  
     * back\_matter\_art: Visually-focused back matter, including sketchbooks, character designs, and process art.  
     * preview: A page containing sequential art that is part of a preview for a different story.  
     * cover\_internal: Covers of single issues reprinted inside a trade collection.  
     * other: Any page that does not fit the above categories (e.g., table of contents, back cover, blank pages).  
3. **Annotation Guidelines:** A detailed annotation guide is crucial for consistency. This document, modeled after the DocLayNet labeling guide 49, must provide clear, explicit rules with visual examples for each label. For example:  
   * *Rule:* "A page is labeled narrative only if it contains at least one panel contributing to the primary story. A full-page text piece, even if it is in-universe, should be labeled back\_matter\_text."  
   * *Rule:* "If a page contains both a small ad and story content, label it narrative. A page should only be labeled advertisement if the ad is the dominant or sole content."  
4. **Tooling and Quality Control:**  
   * **Annotation Tool:** Use a standard annotation tool that supports image classification and allows for efficient labeling of sequences.  
   * **Inter-Annotator Agreement (IAA):** To ensure data quality and the clarity of the guidelines, a subset of the corpus (e.g., 10%) should be annotated by at least two independent annotators. The IAA can then be calculated using a metric like Cohen's Kappa for categorical labels. A high IAA score validates the reliability of the annotation schema. This practice of using redundant annotations to measure an upper bound on model performance is a key feature of robust datasets like DocLayNet.48

## **Section VII: Synthesis and Proposed End-to-End Methodology**

Synthesizing the preceding analysis, this section outlines a complete, end-to-end methodology for developing, training, and deploying a system for the automated classification of paratextual and non-narrative content in digital comic archives. This workflow represents a robust, state-of-the-art approach grounded in academic best practices.

### **Step 1: Data Acquisition and Preparation**

The initial step is to assemble a representative corpus of digital comic books.

* **Acquisition:** Collect a large number of comic book archive files (e.g., .cbz, .cbr) from public domain sources or institutional collections.  
* **Diversity:** Ensure the corpus includes a wide variety of publishers, genres, historical eras, and a balanced mix of single issues and trade collections.  
* **Preprocessing:** Standardize the data by extracting all page images from the archive files into a consistent format (e.g., PNG) and sequence order.

### **Step 2: Document-Level Format Classification**

Before page-level analysis, each comic book file must be classified by format.

* **Feature Extraction:** For each file, compute the global features detailed in Section II: total page count, barcode type (UPC/ISBN) from the first page via OCR, and a metric for internal cover repetition using perceptual hashing.  
* **Model Training:** Train a binary classifier (e.g., XGBoost) on a manually labeled subset of the corpus to distinguish single\_issue from trade\_collection.  
* **Inference:** Apply the trained classifier to the entire corpus to generate a format probability (e.g., $P(\\text{format}=\\text{Trade})$) for each book. This probability will serve as a contextual feature in the main pipeline.

### **Step 3: Data Annotation for Page-Stream Segmentation**

Create the ground-truth dataset required for the primary classification task.

* **Schema Application:** Using the two-tiered annotation schema defined in Section VI, manually label a substantial subset of the prepared corpus. Each file should be tagged with its format, and every page within it should be assigned a label from the set (cover\_front, narrative, advertisement, etc.).  
* **Quality Control:** A portion of the data must be annotated by multiple annotators to calculate Inter-Annotator Agreement (IAA). The annotation guidelines should be refined until a high IAA score (e.g., Kappa \> 0.8) is achieved, ensuring the labels are consistent and reliable.  
* **Data Splitting:** Partition the annotated data into training, validation, and test sets. Critically, this split should be done at the book level, not the page level, to prevent data leakage, as pages from the same book are highly correlated and may share unique layout styles.48

### **Step 4: Model Training and Fine-Tuning**

Train the core multimodal Transformer model.

* **Visual Pre-training (Optional but Recommended):** To improve performance and reduce the amount of comic-specific data needed, pre-train the visual backbone of the model on a large-scale DLA dataset like PubLayNet or DocLayNet. This allows the model to learn general principles of document layout before specializing in comics.  
* **End-to-End Fine-Tuning:** Implement the CoSMo-inspired architecture from Section III. The model will take a sequence of pages as input, where each page is represented by a concatenated visual embedding, textual embedding (from OCR), and the document-level format probability.  
* **Training:** Train the full model on the annotated comic book training set. The task is to minimize the cross-entropy loss between the model's predicted page labels and the ground-truth labels for each sequence.

### **Step 5: Evaluation**

Rigorously evaluate the performance of the trained model.

* **Metrics:** Use a suite of metrics appropriate for multi-class sequence labeling.  
  * **Per-Class Metrics:** Calculate Precision, Recall, and F1-score for each page category to understand model performance on both common classes (like narrative) and rare classes (like credits\_indicia).  
  * **Overall Metrics:** Report overall accuracy and macro-averaged F1-score.  
  * **Segmentation Metrics:** Employ metrics that evaluate the correctness of the segmentation boundaries, such as Panoptic Quality (PQ), as used in the CoSMo evaluation.22 This measures how well the model identifies contiguous segments of the same class.  
* **Error Analysis:** Conduct a thorough qualitative and quantitative error analysis on the validation set. Identify common confusion pairs (e.g., does the model confuse back\_matter\_text with narrative pages that are text-heavy?). Pay special attention to the model's performance on the "preview problem" and other challenging cases identified during development. This analysis will guide further model refinement.

### **Step 6: Deployment and Application in Digital Archives**

The ultimate goal is to apply the trained model to enhance the functionality of a digital comic archive.

* **Automated Metadata Generation:** Deploy the model to process a large, unannotated collection of comics. The model's output—a predicted label for every page—can be stored as structured metadata alongside the comic files.  
* **Enabling New Research and Discovery:** This generated metadata unlocks powerful new capabilities for researchers and users. It enables faceted search queries that were previously impossible, such as:  
  * "Find all letters columns from DC Comics published between 1980 and 1985."  
  * "Analyze the visual style of advertisements in comics from the 1990s."  
  * "Extract all narrative pages, excluding paratextual content, for a large-scale story analysis project."  
* **Preservation and Access:** By automatically structuring these vast and complex digital objects, this methodology is a vital step toward the long-term preservation of cultural heritage and enables more intelligent, scalable access to massive comic book corpora.22

#### **Works cited**

1. “Not merely para”: continuing steps in paratextual research \- ResearchGate, accessed October 31, 2025, [https://www.researchgate.net/publication/316638927\_Not\_merely\_para\_continuing\_steps\_in\_paratextual\_research](https://www.researchgate.net/publication/316638927_Not_merely_para_continuing_steps_in_paratextual_research)  
2. Judging Comics by Their Covers: Comic Books, Text, Paratext and ..., accessed October 31, 2025, [https://www.popmatters.com/190063-judging-comics-by-their-covers-2495567072.html](https://www.popmatters.com/190063-judging-comics-by-their-covers-2495567072.html)  
3. Digital Humanities Quarterly: Comic Book Markup Language: An Introduction and Rationale \- DHQ Static, accessed October 31, 2025, [https://dhq-static.digitalhumanities.org/pdf/000117.pdf](https://dhq-static.digitalhumanities.org/pdf/000117.pdf)  
4. (PDF) Paratextual Negotiations: Fan Forums as Digital Epitexts of Popular Superhero Comic Books and Science Fiction Pulp Novel Series \- ResearchGate, accessed October 31, 2025, [https://www.researchgate.net/publication/369961648\_Paratextual\_Negotiations\_Fan\_Forums\_as\_Digital\_Epitexts\_of\_Popular\_Superhero\_Comic\_Books\_and\_Science\_Fiction\_Pulp\_Novel\_Series](https://www.researchgate.net/publication/369961648_Paratextual_Negotiations_Fan_Forums_as_Digital_Epitexts_of_Popular_Superhero_Comic_Books_and_Science_Fiction_Pulp_Novel_Series)  
5. Beyond the By-line: Paratextual Readings of Alison Bechdel's Dykes to Watch Out For, accessed October 31, 2025, [https://research-portal.uea.ac.uk/en/publications/beyond-the-by-line-paratextual-readings-of-alison-bechdels-dykes-](https://research-portal.uea.ac.uk/en/publications/beyond-the-by-line-paratextual-readings-of-alison-bechdels-dykes-)  
6. Paratextual Negotiations: Fan Forums as Digital Epitexts of Popular Superhero Comic Books and Science Fiction Pulp Novel Series \- MDPI, accessed October 31, 2025, [https://www.mdpi.com/2076-0752/12/2/77](https://www.mdpi.com/2076-0752/12/2/77)  
7. Comics as Heritage: Theorizing Digital Futures of Vernacular Expression \- MDPI, accessed October 31, 2025, [https://www.mdpi.com/2571-9408/8/8/295](https://www.mdpi.com/2571-9408/8/8/295)  
8. Trade Paperbacks vs Single Issues\! \- YouTube, accessed October 31, 2025, [https://www.youtube.com/watch?v=BtMvEQadgqQ](https://www.youtube.com/watch?v=BtMvEQadgqQ)  
9. Single Issues vs. Trades vs. Graphic Novels \- YouTube, accessed October 31, 2025, [https://www.youtube.com/watch?v=x1pUkJVBM2o](https://www.youtube.com/watch?v=x1pUkJVBM2o)  
10. Changing the Paradigm: Why Digital Comics Need to Move Beyond the Single Issue, accessed October 31, 2025, [https://comicsalliance.com/digital-comics-format/](https://comicsalliance.com/digital-comics-format/)  
11. COMICS PORTAL: Trade Paperback or Individual Issues? \- Major Spoilers, accessed October 31, 2025, [https://majorspoilers.com/2015/12/14/comics-portal-trade-paperback-individual-issues/](https://majorspoilers.com/2015/12/14/comics-portal-trade-paperback-individual-issues/)  
12. Single-Issue vs Trade Paperbacks : r/comicbooks \- Reddit, accessed October 31, 2025, [https://www.reddit.com/r/comicbooks/comments/yw1l1l/singleissue\_vs\_trade\_paperbacks/](https://www.reddit.com/r/comicbooks/comments/yw1l1l/singleissue_vs_trade_paperbacks/)  
13. Transductive Learning for Near-Duplicate Image Detection in Scanned Photo Collections, accessed October 31, 2025, [https://arxiv.org/html/2410.19437v1](https://arxiv.org/html/2410.19437v1)  
14. High-Confidence Near-Duplicate Image Detection \- cs.Princeton, accessed October 31, 2025, [https://www.cs.princeton.edu/cass/papers/icmr12.pdf](https://www.cs.princeton.edu/cass/papers/icmr12.pdf)  
15. \[2308.15517\] Document AI: A Comparative Study of Transformer-Based, Graph-Based Models, and Convolutional Neural Networks For Document Layout Analysis \- arXiv, accessed October 31, 2025, [https://arxiv.org/abs/2308.15517](https://arxiv.org/abs/2308.15517)  
16. (PDF) Document AI: A Comparative Study of Transformer-Based ..., accessed October 31, 2025, [https://www.researchgate.net/publication/373492539\_Document\_AI\_A\_Comparative\_Study\_of\_Transformer-Based\_Graph-Based\_Models\_and\_Convolutional\_Neural\_Networks\_For\_Document\_Layout\_Analysis](https://www.researchgate.net/publication/373492539_Document_AI_A_Comparative_Study_of_Transformer-Based_Graph-Based_Models_and_Convolutional_Neural_Networks_For_Document_Layout_Analysis)  
17. Efficient Comic Content Extraction and Coloring Composite Networks \- MDPI, accessed October 31, 2025, [https://www.mdpi.com/2076-3417/15/5/2641](https://www.mdpi.com/2076-3417/15/5/2641)  
18. A Deep Learning Pipeline for the Synthesis of Graphic Novels \- Association for Computational Creativity, accessed October 31, 2025, [https://computationalcreativity.net/iccc21/wp-content/uploads/2021/09/ICCC\_2021\_paper\_52.pdf](https://computationalcreativity.net/iccc21/wp-content/uploads/2021/09/ICCC_2021_paper_52.pdf)  
19. Insights | Document Layout Analysis With Graphbased Methods, accessed October 31, 2025, [https://prometeia.com/en/about-us/insights/article/document-layout-analysis-with-graphbased-methods-15929781](https://prometeia.com/en/about-us/insights/article/document-layout-analysis-with-graphbased-methods-15929781)  
20. \[2304.11810\] PARAGRAPH2GRAPH: A GNN-based framework for layout paragraph analysis \- arXiv, accessed October 31, 2025, [https://arxiv.org/abs/2304.11810](https://arxiv.org/abs/2304.11810)  
21. Vision Grid Transformer for Document Layout Analysis \- CVF Open Access, accessed October 31, 2025, [https://openaccess.thecvf.com/content/ICCV2023/papers/Da\_Vision\_Grid\_Transformer\_for\_Document\_Layout\_Analysis\_ICCV\_2023\_paper.pdf](https://openaccess.thecvf.com/content/ICCV2023/papers/Da_Vision_Grid_Transformer_for_Document_Layout_Analysis_ICCV_2023_paper.pdf)  
22. arxiv.org, accessed October 31, 2025, [https://arxiv.org/html/2507.10053v1](https://arxiv.org/html/2507.10053v1)  
23. CoSMo: A Multimodal Transformer for Page Stream Segmentation in Comic Books \- arXiv, accessed October 31, 2025, [https://arxiv.org/pdf/2507.10053?](https://arxiv.org/pdf/2507.10053)  
24. CoSMo: A Multimodal Transformer for Page Stream Segmentation in Comic Books, accessed October 31, 2025, [https://www.researchgate.net/publication/393684313\_CoSMo\_A\_Multimodal\_Transformer\_for\_Page\_Stream\_Segmentation\_in\_Comic\_Books](https://www.researchgate.net/publication/393684313_CoSMo_A_Multimodal_Transformer_for_Page_Stream_Segmentation_in_Comic_Books)  
25. COMICORDA: Dialogue Act Recognition in Comic ... \- ACL Anthology, accessed October 31, 2025, [https://aclanthology.org/2024.lrec-main.316.pdf](https://aclanthology.org/2024.lrec-main.316.pdf)  
26. Contextual Text Block Detection towards Scene Text Understanding, accessed October 31, 2025, [https://www.ecva.net/papers/eccv\_2022/papers\_ECCV/papers/136880371.pdf](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880371.pdf)  
27. Deep-Learning-Based Complex Scene Text Detection Algorithm for Architectural Images, accessed October 31, 2025, [https://www.mdpi.com/2227-7390/10/20/3914](https://www.mdpi.com/2227-7390/10/20/3914)  
28. Detect text in images | Cloud Vision API \- Google Cloud Documentation, accessed October 31, 2025, [https://docs.cloud.google.com/vision/docs/ocr](https://docs.cloud.google.com/vision/docs/ocr)  
29. (PDF) AD or Non-AD: A Deep Learning Approach to Detect ..., accessed October 31, 2025, [https://www.researchgate.net/publication/329746938\_AD\_or\_Non-AD\_A\_Deep\_Learning\_Approach\_to\_Detect\_Advertisements\_from\_Magazines](https://www.researchgate.net/publication/329746938_AD_or_Non-AD_A_Deep_Learning_Approach_to_Detect_Advertisements_from_Magazines)  
30. AD or Non-AD: A Deep Learning Approach to Detect Advertisements from Magazines \- PMC, accessed October 31, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7512581/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7512581/)  
31. ADNet: A Deep Network for Detecting Adverts \- CEUR-WS, accessed October 31, 2025, [https://ceur-ws.org/Vol-2259/aics\_6.pdf](https://ceur-ws.org/Vol-2259/aics_6.pdf)  
32. \[2501.10848\] Fake Advertisements Detection Using Automated Multimodal Learning: A Case Study for Vietnamese Real Estate Data \- arXiv, accessed October 31, 2025, [https://arxiv.org/abs/2501.10848](https://arxiv.org/abs/2501.10848)  
33. Comic Story Analysis Based on Genre Classification | Request PDF, accessed October 31, 2025, [https://www.researchgate.net/publication/322776413\_Comic\_Story\_Analysis\_Based\_on\_Genre\_Classification](https://www.researchgate.net/publication/322776413_Comic_Story_Analysis_Based_on_Genre_Classification)  
34. (PDF) Sketch Based Image Retrieval using Deep Learning Based ..., accessed October 31, 2025, [https://www.researchgate.net/publication/352851021\_Sketch\_Based\_Image\_Retrieval\_using\_Deep\_Learning\_Based\_Machine\_Learning](https://www.researchgate.net/publication/352851021_Sketch_Based_Image_Retrieval_using_Deep_Learning_Based_Machine_Learning)  
35. Deep learning for studying drawing behavior: A review \- PMC \- NIH, accessed October 31, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9945213/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9945213/)  
36. Methods and Challenges in Shot Boundary Detection: A Review \- MDPI, accessed October 31, 2025, [https://www.mdpi.com/1099-4300/20/4/214](https://www.mdpi.com/1099-4300/20/4/214)  
37. Paragraph Boundary Recognition in Novels for Story Understanding \- MDPI, accessed October 31, 2025, [https://www.mdpi.com/2076-3417/11/12/5632](https://www.mdpi.com/2076-3417/11/12/5632)  
38. Automatic Segmentation of Narrative Text Into Scenes According to SceneML \- CEUR-WS.org, accessed October 31, 2025, [https://ceur-ws.org/Vol-3964/paper9.pdf](https://ceur-ws.org/Vol-3964/paper9.pdf)  
39. Feature visualization in comic artist classification using deep neural networks \- SciSpace, accessed October 31, 2025, [https://scispace.com/pdf/feature-visualization-in-comic-artist-classification-using-3ibqa9jbug.pdf](https://scispace.com/pdf/feature-visualization-in-comic-artist-classification-using-3ibqa9jbug.pdf)  
40. (PDF) Feature visualization in comic artist classification using deep ..., accessed October 31, 2025, [https://www.researchgate.net/publication/334011441\_Feature\_visualization\_in\_comic\_artist\_classification\_using\_deep\_neural\_networks](https://www.researchgate.net/publication/334011441_Feature_visualization_in_comic_artist_classification_using_deep_neural_networks)  
41. What is Content fingerprinting for tracking? | Fugo Digital Signage Wiki, accessed October 31, 2025, [https://www.fugo.ai/wiki/content-fingerprinting-for-tracking/](https://www.fugo.ai/wiki/content-fingerprinting-for-tracking/)  
42. BASIL: Effective Near-Duplicate Image Detection using Gene Sequence Alignment \- The PIKE Group, accessed October 31, 2025, [https://pike.psu.edu/publications/ecir10.pdf](https://pike.psu.edu/publications/ecir10.pdf)  
43. (PDF) PubLayNet: Largest Dataset Ever for Document Layout Analysis \- ResearchGate, accessed October 31, 2025, [https://www.researchgate.net/publication/336288174\_PubLayNet\_Largest\_Dataset\_Ever\_for\_Document\_Layout\_Analysis](https://www.researchgate.net/publication/336288174_PubLayNet_Largest_Dataset_Ever_for_Document_Layout_Analysis)  
44. ibm-aur-nlp/PubLayNet \- GitHub, accessed October 31, 2025, [https://github.com/ibm-aur-nlp/PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet)  
45. Document layout recognition dataset: PubLayNet \- Kaggle, accessed October 31, 2025, [https://www.kaggle.com/datasets/devashishprasad/documnet-layout-recognition-dataset-publaynet-t0](https://www.kaggle.com/datasets/devashishprasad/documnet-layout-recognition-dataset-publaynet-t0)  
46. \[PDF\] PubLayNet: Largest Dataset Ever for Document Layout Analysis | Semantic Scholar, accessed October 31, 2025, [https://www.semanticscholar.org/paper/PubLayNet%3A-Largest-Dataset-Ever-for-Document-Layout-Zhong-Tang/b5799d10df17de3232540e990da69553800d6376](https://www.semanticscholar.org/paper/PubLayNet%3A-Largest-Dataset-Ever-for-Document-Layout-Zhong-Tang/b5799d10df17de3232540e990da69553800d6376)  
47. DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis \- GitHub, accessed October 31, 2025, [https://github.com/DS4SD/DocLayNet](https://github.com/DS4SD/DocLayNet)  
48. \[2206.01062\] DocLayNet: A Large Human-Annotated Dataset for ..., accessed October 31, 2025, [https://ar5iv.labs.arxiv.org/html/2206.01062](https://ar5iv.labs.arxiv.org/html/2206.01062)  
49. DocLayNet \- Dataset details \- ModelScope, accessed October 31, 2025, [https://modelscope.cn/datasets/swift/DocLayNet](https://modelscope.cn/datasets/swift/DocLayNet)  
50. Advancing Manga Analysis: Comprehensive Segmentation ..., accessed October 31, 2025, [https://openaccess.thecvf.com/content/CVPR2025/html/Xie\_Advancing\_Manga\_Analysis\_Comprehensive\_Segmentation\_Annotations\_for\_the\_Manga109\_Dataset\_CVPR\_2025\_paper.html](https://openaccess.thecvf.com/content/CVPR2025/html/Xie_Advancing_Manga_Analysis_Comprehensive_Segmentation_Annotations_for_the_Manga109_Dataset_CVPR_2025_paper.html)  
51. arxiv.org, accessed October 31, 2025, [https://arxiv.org/html/2407.03550v2](https://arxiv.org/html/2407.03550v2)  
52. ComicsPAP: understanding comic strips by picking the correct panel \- arXiv, accessed October 31, 2025, [https://arxiv.org/html/2503.08561v3](https://arxiv.org/html/2503.08561v3)