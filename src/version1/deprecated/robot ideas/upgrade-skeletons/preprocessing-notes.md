Great call. Let’s lock down a clear “data contract” and preprocessing checklist so the model is plug-and-play. Short answer: yes, I’d reuse CoMix’s panel segmentation/boundary geometry (and any reading-order module if available). Below is a minimal-to-ideal spec, plus how to convert CoMix outputs into the JSON this pipeline expects.

Overview
- Minimal to train CLOSURE-Lite: page images, per-panel crops or boxes, per-panel text (OCR), basic layout geometry (panel bbox).
- Strongly recommended: panel polygons or masks (for accurate cropping), per-panel OCR tokens with boxes, page size, reading order (if your tool provides it).
- Optional but valuable: character bounding boxes, balloon polygons, per-panel mask path, adjacency.

Preprocessing bill of materials
1) Page images
- Format: PNG or JPG
- Path must be stable and referenced from JSON (page_image_path)
- Page size (W, H) is required (store in JSON or read at runtime from the file)

2) Panel segmentation (use CoMix)
- Use CoMix panel segmentation to extract:
  - Polygon per panel (list of (x, y) points in page pixel coords)
  - Rectangular bbox [x, y, w, h] in page pixel coords (tight bbox of polygon)
- Optional but recommended:
  - Panel mask files (binary PNG per panel) for precise cropping and composition cues
  - Any “reading order”/index CoMix provides

3) OCR and balloon/text association
- OCR the page (or cropped panels) to produce tokens:
  - text string
  - bbox per token [x, y, w, h] in page pixel coords
  - confidence and (optional) language
- Assign OCR tokens to panels by polygon/mask containment (preferred) or by highest IoU with panel bbox.
- Aggregate per-panel text into three buckets if you have balloon type heuristics:
  - dialogue: list of strings
  - narration: list of strings
  - sfx: list of strings
- If you don’t have balloon classes, put everything into dialogue and leave narration/sfx empty.

4) Reading order and adjacency
- If CoMix provides reading order: store order_index per panel (0..N-1).
- If not, we’ll infer it at training time from geometry (you can omit it).
- If your tooling can produce adjacency (neighbors on page layout), store a neighbors dict; else we’ll infer.

5) Character (optional but useful)
- If you have a character/head detector, store character_coords as a list of [x, y, w, h] per panel. It boosts composition features (shot scale, subject position).
- If not available, the model still trains; composition falls back to size/AR features.

6) Panel crops (optional cache)
- You can pre-save panel crops (224x224 or higher) to disk to speed up training.
- If you have masks/polygons, apply masking (outside-polygon = black) before resizing to avoid leakage from neighboring panels.

Directory layout (example)
- data/
  - pages/
    - book123/page_0001.jpg
    - book123/page_0002.jpg
  - masks/              # optional
    - book123/page_0001/panel_0001.png
  - json/
    - book123/page_0001.json
    - book123/page_0002.json

Required JSON schema (DataSpec v0.3)
At minimum, one JSON per page with:

{
  "page_id": "book123_page_0001",
  "page_image_path": "data/pages/book123/page_0001.jpg",
  "page_size": {"width": 1650, "height": 2550},
  "panels": [
    {
      "panel_id": "p0",
      "panel_coords": [x, y, w, h],                // int pixels, page coord
      "polygon": [[x1,y1], [x2,y2], ...],          // optional but recommended
      "mask_path": "data/masks/book123/page_0001/p0.png", // optional
      "order_index": 0,                             // optional if you have it
      "text": {
        "dialogue": ["..."],                        // strings (aggregated OCR)
        "narration": [],
        "sfx": []
      },
      "ocr_tokens": [                               // optional (token-level OCR)
        {"text": "Hello", "bbox": [x,y,w,h], "conf": 0.92, "lang": "en"}
      ],
      "character_coords": [[x,y,w,h], ...]          // optional
    }
  ]
}

Notes
- Coordinates: absolute pixels; origin top-left; x,y = top-left; w,h = width,height.
- If you provide polygons, we’ll still compute and use bboxes; polygons improve masking/assignment.
- If you don’t have text buckets, set narration/sfx to [] and put strings in dialogue.

How we use CoMix outputs
Yes, use CoMix for:
- Panel segmentation: polygons, bboxes
- Balloon/text region detection (if available)
- Reading order estimation (if available)
- Boundary geometry features (panel contours, sizes)

Mapping CoMix → DataSpec v0.3
- CoMix panel polygons → polygon
- Tight bbox of polygon → panel_coords
- If CoMix gives reading order → order_index
- If CoMix gives balloon polygons, associate OCR tokens that fall inside balloon polygons to dialogue; text outside balloons but inside rectangular caption boxes → narration; large stylized tokens with outline/angle → sfx (heuristic; optional)
- Save per-panel masks (render polygon to binary PNG), store mask_path

If CoMix does not provide reading order/adjacency
- Omit order_index and neighbors. Our loader can infer Z-path order and adjacency on the fly using the geometry code I shared earlier.
- If you prefer to precompute once, we can provide a script to write inferred order_index into the JSON.

Minimum viable preprocessing (fits A5000, trains today)
- You provide:
  - Page images
  - One JSON/page with: page_size, panels[].panel_coords, panels[].text buckets (or a single merged string in dialogue)
- We derive:
  - Reading order and adjacency from geometry
  - Composition features from bbox + character_coords if present (else default 0)

Recommended preprocessing (better results, still light)
- Add panel polygons (polygon)
- Add per-panel masks (mask_path)
- Add OCR tokens with bboxes (ocr_tokens), even if you don’t bucket them into dialogue/narration/sfx
- Add order_index if CoMix outputs it

Nice-to-have (advanced tasks later)
- character_coords (from a comic-specific detector)
- balloon polygons and caption boxes
- neighbors (explicit adjacency)
- page-level layout metadata (grid detection, bleed/splash flags)

Conversion helper (pseudo) from CoMix page JSON
Assuming CoMix emits per-page with panel polygons and OCR:

def convert_comix_to_dataspec(comix_page, page_img_path):
    W, H = comix_page['page_width'], comix_page['page_height']
    out = {
        "page_id": comix_page["id"],
        "page_image_path": page_img_path,
        "page_size": {"width": W, "height": H},
        "panels": []
    }
    for k, pnl in enumerate(comix_page["panels"]):
        poly = pnl["polygon"]  # list of [x,y]
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        x, y, w, h = min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)
        text_buckets = bucket_tokens_to_text(pnl.get("ocr_tokens", []), pnl.get("balloons", []))
        out["panels"].append({
            "panel_id": pnl.get("id", f"p{k}"),
            "panel_coords": [int(x), int(y), int(w), int(h)],
            "polygon": poly,
            "mask_path": pnl.get("mask_path", ""),      # if you render and save it
            "order_index": pnl.get("order_index", None),
            "text": text_buckets,
            "ocr_tokens": pnl.get("ocr_tokens", []),
            "character_coords": pnl.get("characters", [])
        })
    return out

bucket_tokens_to_text can be a simple heuristic:
- If a token lies inside a balloon polygon → dialogue
- If inside a caption box polygon (if CoMix provides) → narration
- If token font-size is large, angled, or outside balloons but inside panel → sfx
- Else → dialogue

Validation checklist (quick QA)
- For each page:
  - page_image_path exists and is loadable; page_size matches file
  - panels length >= 1; all panel_coords within page bounds; w,h > 0
  - If polygon present, all points within page bounds
  - OCR assignment coverage: sum of tokens assigned to panels ≥ 80% of tokens (optional warning if low)
  - If order_index present, it’s a permutation of [0..N-1] without gaps

Why masks/polygons help
- Cleaner crops: non-rectangular panels won’t leak neighboring art into the patch
- Better composition: accurate panel area and subject positioning
- More reliable text assignment: tokens near boundaries won’t be misassigned

If you want a zero-friction start
- Give me 2–3 CoMix page outputs + their images as-is.
- I’ll write the converter to DataSpec v0.3 and a quick validator script that flags boundary issues and produces a tiny “train-ready” shard.

Would I use the CoMix repo? Yes
- It accelerates the most brittle steps (panel polygons, boundary geometry, possibly reading order and balloon regions). We’ll adapt to its outputs and only fall back to our inference when a field is missing. If CoMix’s reading-order is reliable for your corpus, we’ll prefer it over heuristic inference and pass it straight to the RPP and sequential training.

Next steps
- Decide minimal vs. recommended preprocessing for your first run.
- Share 2–3 sample CoMix JSONs + images; I’ll return:
  - converter to DataSpec v0.3
  - dataset validator
  - tiny train script wired to your format
- Once that’s green, we can expand to masks/balloons/char boxes incrementally.