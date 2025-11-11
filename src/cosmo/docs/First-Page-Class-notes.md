“multiple discrete stories per issue” is far less common in many modern formats, while classic Golden‑Age/early Silver‑Age books (’40s/’50s) frequently were true anthologies. That difference matters for whether a First-Page metaclass pays off.

Where First-Page is very useful
- Golden‑Age/early anthology books: Issues often contained several short, self-contained stories interleaved with ads and text stories. An explicit First-Page label gives clean, automatable cut points, boosting document-level metrics (DocF1, PQ) and lowering human correction (MnDD). In this regime, First-Page detection is almost the whole game: it converts page classification into correct story blocks with minimal postprocessing.

- Modern anthologies with short arcs: Even if an issue has just two or three segments, First-Page remains a high‑leverage signal because boundary errors are disproportionately costly in downstream indexing and user-facing navigation. A single page shift can invalidate the whole segment span; marking First-Page reduces those off‑by‑one drifts.

Where it needs adaptation
- Ongoing serials with “Part N of M” within the same strip each week (e.g., 2000 AD’s progs): An “episode start” is indeed structurally meaningful, but it isn’t the start of a new narrative universe; it’s the next installment of the same serial. Two refinements help:
  1) Split the metaclass: distinguish Episode-Start from Story-Start. The former marks “new installment of same serial,” the latter “start of a new narrative unit.” This preserves the utility for segmentation while respecting serial continuity.
  2) Add an Episode-ID or Series-ID track: beyond page labels, maintain sequence-level grouping that links episodes across issues. That lets you assemble cross-issue arcs without over-segmenting within an issue.

- Single‑story issues and trades: Many modern issues are one continuous story; in collected editions, even more so. Here, First-Page provides little marginal value after the cover. Practical options:
  - Switch it off via a domain prior: if metadata or a quick probe suggests a single narrative block, you can suppress First-Page predictions beyond the first occurrence.
  - Soft boundary confidence: treat First-Page as a boundary score used only when it exceeds a threshold and when supporting context agrees (layout shift, title card, credits). This avoids spurious splits in single‑story flows.

Cross‑era perspective
- 1940s/50s: First-Page is a strong, high-yield structural label. Title cards, splash pages, and editorial cues are common; segment granularity aligns with how readers and archivists think about those books.
- Late‑20th century to now: Utility becomes conditional on format. For serials, rename/specialize to Episode-Start and combine with Series/Arc identifiers. For one‑shots and graphic novels, demote First-Page to an optional cue and rely more on coarse structural classes (Cover, Story, Back‑matter) plus chapter detection if present.

Implementation suggestions
- Conditional decoding: Use a learned prior over “format type” (anthology vs serial vs single‑story) inferred from a short context window or lightweight metadata; route to format‑specific postprocessors.
- Hierarchical labels: Keep page labels (Cover/Ad/Text/Story) and add a hierarchical layer with boundary types: Story-Start, Episode-Start, Chapter-Start. This supports both classic anthologies and modern serials.
- Confidence‑aware merging: In serials, require boundary agreement across multiple signals (layout shift + title OCR + narrative cue) before committing to a split; otherwise, carry the boundary as a soft marker for downstream tools.

Bottom line: First-Page as a metaclass is a big win for Golden‑Age anthologies and still useful in modern mixed-format issues, but it should be specialized (Episode-Start vs Story-Start) and made conditional for serials and single‑story formats. That preserves its segmentation benefits where they matter, while avoiding over‑segmentation in contemporary structures.