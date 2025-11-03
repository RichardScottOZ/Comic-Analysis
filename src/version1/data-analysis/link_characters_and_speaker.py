import argparse
import pandas as pd
import json
import os
from pathlib import Path
from tqdm import tqdm

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    x1_tl, y1_tl, w1, h1 = box1
    x2_tl, y2_tl, w2, h2 = box2

    x1_br, y1_br = x1_tl + w1, y1_tl + h1
    x2_br, y2_br = x2_tl + w2, y2_tl + h2

    # Determine the coordinates of the intersection rectangle
    x_left = max(x1_tl, x2_tl)
    y_top = max(y1_tl, y2_tl)
    x_right = min(x1_br, x2_br)
    y_bottom = min(y1_br, y2_br)

    # Compute the area of intersection rectangle
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Compute the area of both bounding boxes
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Compute the area of union
    union_area = float(box1_area + box2_area - intersection_area)

    # Compute the IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou

def find_closest_bbox(target_bbox, candidates):
    """Finds the candidate bbox with the highest IoU with the target_bbox."""
    if not candidates:
        return None, 0.0
    
    best_iou = 0.0
    best_candidate = None

    for candidate_bbox in candidates:
        iou = calculate_iou(target_bbox, candidate_bbox)
        if iou > best_iou:
            best_iou = iou
            best_candidate = candidate_bbox
    return best_candidate, best_iou

def link_data_for_image(image_path, coco_data, vlm_json_path, category_names, debug=False):
    """Links detections from COCO JSON with textual info from VLM JSON for a single image."""
    image_results = {
        "image_path": image_path,
        "vlm_json_path": vlm_json_path,
        "linked_characters": []
    }

    # Load VLM JSON
    vlm_data = {}
    if vlm_json_path and os.path.exists(vlm_json_path):
        try:
            with open(vlm_json_path, 'r', encoding='utf-8') as f:
                vlm_data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load VLM JSON {vlm_json_path}: {e}")

    # Get COCO detections for this image
    img_detections = coco_data.get(image_path, [])

    # Filter detections by category
    characters = [d for d in img_detections if d['category_id'] == category_names['character']]
    faces = [d for d in img_detections if d['category_id'] == category_names['face']]
    balloons = [d for d in img_detections if d['category_id'] == category_names['balloon']]
    link_sbscs = [d for d in img_detections if d['category_id'] == category_names['link_sbsc']]

    # Process VLM speakers and try to link to visual detections
    vlm_speakers_info = []
    if 'panels' in vlm_data and isinstance(vlm_data['panels'], list):
        for panel_data in vlm_data['panels']:
            if 'speakers' in panel_data and isinstance(panel_data['speakers'], list):
                for speaker_entry in panel_data['speakers']:
                    if 'character' in speaker_entry and 'dialogue' in speaker_entry:
                        vlm_speakers_info.append({
                            'vlm_character_name': speaker_entry['character'],
                            'vlm_dialogue': speaker_entry['dialogue'],
                            'panel_number': panel_data.get('panel_number')
                        })
    # Build a list of VLM-extracted balloons (if present) to match text to detected balloons
    vlm_balloons = []
    if 'panels' in vlm_data and isinstance(vlm_data['panels'], list):
        for panel_data in vlm_data['panels']:
            if 'balloons' in panel_data and isinstance(panel_data['balloons'], list):
                for bal in panel_data['balloons']:
                    # balloon bbox field names may vary; try common ones
                    b_bbox = bal.get('bbox') or bal.get('bounding_box') or bal.get('box')
                    b_text = bal.get('text') or bal.get('text_content') or bal.get('raw_text') or ''
                    if b_bbox is not None:
                        # Normalize VLM balloon bbox to COCO format [x,y,w,h] if it's in [x1,y1,x2,y2]
                        try:
                            if isinstance(b_bbox, (list, tuple)) and len(b_bbox) == 4:
                                x0, y0, x1, y1 = b_bbox
                                # Heuristic: if x1 > x0 and y1 > y0 but x1 looks like an x2 coordinate (> x0)
                                # then convert to [x,y,w,h]
                                if x1 > x0 and y1 > y0 and (x1 - x0) > 1 and (y1 - y0) > 1:
                                    # If the bbox already looks like xywh (width smaller than x1), keep as-is.
                                    # We assume input is xyxy when x1 > x0 and the values look like endpoints.
                                    bw = x1 - x0
                                    bh = y1 - y0
                                    b_bbox_norm = [x0, y0, bw, bh]
                                else:
                                    b_bbox_norm = list(b_bbox)
                            else:
                                b_bbox_norm = b_bbox
                        except Exception:
                            b_bbox_norm = b_bbox
                        vlm_balloons.append({'bbox': b_bbox_norm, 'text': b_text, 'panel_number': panel_data.get('panel_number')})
    
    # Attempt to link balloons to characters/faces and then to VLM speakers
    for balloon_det in balloons:
        linked_char_bbox = None
        linked_char_score = 0.0
        linked_char_id = None
        linked_face_bbox = None
        linked_face_score = 0.0
        linked_face_id = None
        
        # Try to use link_sbsc if available (strongest hint)
        best_link_iou = 0.0
        best_linked_char_from_link = None
        for link_det in link_sbscs:
            # Assuming link_sbsc bbox might cover both balloon and character, or is a point. 
            # For now, let's assume it's a bbox that should overlap a character.
            # This part might need refinement based on actual link_sbsc bbox format.
            iou_with_balloon = calculate_iou(balloon_det['bbox'], link_det['bbox'])
            if iou_with_balloon > 0.1: # If link overlaps balloon
                # Find character closest to this link
                for char_det in characters:
                    iou_with_char = calculate_iou(char_det['bbox'], link_det['bbox'])
                    if iou_with_char > best_link_iou:
                        best_link_iou = iou_with_char
                        best_linked_char_from_link = char_det
        
        if best_linked_char_from_link:
            linked_char_bbox = best_linked_char_from_link['bbox']
            linked_char_score = best_linked_char_from_link.get('score')
            linked_char_id = best_linked_char_from_link.get('id')
        else:
            # If no link_sbsc, find closest character/face by proximity to balloon
            closest_char_bbox, char_iou = find_closest_bbox(balloon_det['bbox'], [c['bbox'] for c in characters])
            closest_face_bbox, face_iou = find_closest_bbox(balloon_det['bbox'], [f['bbox'] for f in faces])

            if closest_char_bbox and char_iou > 0.1: # Threshold for IoU
                linked_char_bbox = closest_char_bbox
                linked_char_score = char_iou # Using IoU as score for now
                linked_char_id = next(d['id'] for d in characters if d['bbox'] == closest_char_bbox) # Get original ID
            
            if closest_face_bbox and face_iou > 0.1: # Threshold for IoU
                linked_face_bbox = closest_face_bbox
                linked_face_score = face_iou
                linked_face_id = next((d.get('id') for d in faces if d['bbox'] == closest_face_bbox), None) # Get original ID

    # Try to match VLM speaker to the linked visual character using VLM balloon texts (if available)
        vlm_speaker_match = None
        balloon_text = ''
        if vlm_balloons:
            # find VLM balloon that best overlaps this detected balloon
            vlm_balloon_bboxes = [b['bbox'] for b in vlm_balloons]
            best_vlm_balloon, best_vlm_iou = find_closest_bbox(balloon_det['bbox'], vlm_balloon_bboxes)
            if best_vlm_balloon and best_vlm_iou > 0.0:
                # find the balloon dict matching this bbox
                for vb in vlm_balloons:
                    if vb['bbox'] == best_vlm_balloon:
                        balloon_text = vb.get('text', '') or ''
                        break

        if vlm_speakers_info:
            for speaker_info in vlm_speakers_info:
                # match either by dialogue substring or by character name appearing in the balloon text
                if speaker_info.get('vlm_dialogue') and speaker_info['vlm_dialogue'] and speaker_info['vlm_dialogue'] in balloon_text:
                    vlm_speaker_match = speaker_info['vlm_character_name']
                    break
                if speaker_info.get('vlm_character_name') and speaker_info['vlm_character_name'] and speaker_info['vlm_character_name'] in balloon_text:
                    vlm_speaker_match = speaker_info['vlm_character_name']
                    break

        # Fallback: if we couldn't find balloon-level text matches, try panel-level speaker/dialogue matching
        if vlm_speaker_match is None and vlm_speakers_info and not balloon_text:
            # Try to match speaker dialogue to any panel-level text (some VLMs place dialogue under speakers but no balloon text)
            panel_texts = []
            if 'panels' in vlm_data and isinstance(vlm_data['panels'], list):
                for p in vlm_data['panels']:
                    # Collect any textual fields in the panel
                    panel_text = []
                    if 'text' in p and p.get('text'):
                        panel_text.append(str(p.get('text')))
                    if 'ocr_text' in p and p.get('ocr_text'):
                        panel_text.append(str(p.get('ocr_text')))
                    # also collect balloon texts if present
                    for b in p.get('balloons', [])[:10]:
                        t = b.get('text') or b.get('text_content') or b.get('raw_text')
                        if t:
                            panel_text.append(str(t))
                    if panel_text:
                        panel_texts.append(' '.join(panel_text))
            joined_panel_text = ' '.join(panel_texts)
            if joined_panel_text:
                for speaker_info in vlm_speakers_info:
                    if speaker_info.get('vlm_dialogue') and speaker_info['vlm_dialogue'] and speaker_info['vlm_dialogue'] in joined_panel_text:
                        vlm_speaker_match = speaker_info['vlm_character_name']
                        break
                    if speaker_info.get('vlm_character_name') and speaker_info['vlm_character_name'] and speaker_info['vlm_character_name'] in joined_panel_text:
                        vlm_speaker_match = speaker_info['vlm_character_name']
                        break

        image_results['linked_characters'].append({
            'balloon_bbox': balloon_det.get('bbox'),
            'balloon_score': balloon_det.get('score'),
            'linked_char_bbox': linked_char_bbox,
            'linked_char_score': linked_char_score,
            'linked_char_id': linked_char_id,
            'linked_face_bbox': linked_face_bbox,
            'linked_face_score': linked_face_score,
            'linked_face_id': linked_face_id,
            'vlm_speaker_name': vlm_speaker_match
        })

    # Fallback when there are no detected balloons: try to attach VLM speakers to detected characters/faces
    if not balloons:
        if vlm_speakers_info:
            # For each VLM speaker, assign the most likely character or face (by score or area)
            for speaker_info in vlm_speakers_info:
                best_char = None
                best_face = None
                if characters:
                    # prefer a detection with a 'score' field, otherwise use bbox area
                    def char_score_fn(d):
                        return d.get('score') if d.get('score') is not None else (d.get('area') or (d['bbox'][2] * d['bbox'][3]))
                    best_char = max(characters, key=char_score_fn)
                if faces:
                    def face_score_fn(d):
                        return d.get('score') if d.get('score') is not None else (d.get('area') or (d['bbox'][2] * d['bbox'][3]))
                    best_face = max(faces, key=face_score_fn)

                image_results['linked_characters'].append({
                    'balloon_bbox': None,
                    'balloon_score': None,
                    'linked_char_bbox': best_char.get('bbox') if best_char else None,
                    'linked_char_score': best_char.get('score') if best_char else None,
                    'linked_char_id': best_char.get('id') if best_char else None,
                    'linked_face_bbox': best_face.get('bbox') if best_face else None,
                    'linked_face_score': best_face.get('score') if best_face else None,
                    'linked_face_id': best_face.get('id') if best_face else None,
                    'vlm_speaker_name': speaker_info.get('vlm_character_name')
                })
    return image_results

def main():
    parser = argparse.ArgumentParser(description="Links COCO detections with VLM textual info for comic pages.")
    parser.add_argument('--input_canonical_csv', type=str, required=True, help='Path to the canonical mapping CSV (e.g., key_mapping_report_claude_amazon.csv).')
    parser.add_argument('--coco_predictions_json', type=str, required=True, help='Path to the COCO predictions JSON file (from YOLOv8/Faster R-CNN).')
    parser.add_argument('--vlm_json_dir', type=str, required=True, help='Directory containing VLM analysis JSON files.')
    parser.add_argument('--output_csv', type=str, default="linked_characters_speakers.csv", help='Output CSV file path.')
    parser.add_argument('--debug', action='store_true', help='Enable debug prints for troubleshooting key mismatches')
    args = parser.parse_args()

    print(f"Loading canonical mapping from {args.input_canonical_csv}...")
    canonical_df = pd.read_csv(args.input_canonical_csv)

    print(f"Loading COCO predictions from {args.coco_predictions_json}...")
    with open(args.coco_predictions_json, 'r', encoding='utf-8') as f:
        coco_raw_data = json.load(f)
    
    # Map COCO annotations keyed by COCO image file_name (normalized path). Also build a basename -> [file_names] index
    coco_detections_by_image = {}
    basename_index = {}
    coco_categories = {cat['name']: cat['id'] for cat in coco_raw_data.get('categories', [])}

    # Build mapping image_id -> file_name from coco images
    id2file = {img['id']: img.get('file_name') for img in coco_raw_data.get('images', [])}
    for ann in coco_raw_data.get('annotations', []):
        image_id = ann.get('image_id')
        file_name = id2file.get(image_id)
        if not file_name:
            # skip annotations for which we don't have a file name
            continue
        # normalize COCO file_name and use it as the primary key
        key = os.path.normpath(file_name)
        coco_detections_by_image.setdefault(key, []).append(ann)
        # also index by basename for disambiguation
        base = Path(file_name).name
        basename_index.setdefault(base, []).append(key)

    # Debug: print samples to help diagnose mismatches
    if args.debug:
        print("\n[DEBUG] COCO id->file sample:")
        for k, v in list(id2file.items())[:10]:
            print(f"  {k} -> {v}")
        print("\n[DEBUG] Sample keys available in coco_detections_by_image (normalized file_name keys):")
        print(list(coco_detections_by_image.keys())[:10])
        print(f"[DEBUG] Total COCO images indexed: {len(coco_detections_by_image)}\n")
        print("[DEBUG] Sample basename collisions (basename -> #candidates):")
        for b, lst in list(basename_index.items())[:10]:
            if len(lst) > 1:
                print(f"  {b} -> {len(lst)} candidates")
        print("[DEBUG] COCO categories mapping (name->id):")
        print(coco_categories)

    # Ensure all required categories are present
    required_categories = ['character', 'face', 'balloon', 'text', 'link_sbsc']
    for cat_name in required_categories:
        if cat_name not in coco_categories:
            print(f"Warning: Category '{cat_name}' not found in COCO predictions. Some linking might be incomplete.")
            coco_categories[cat_name] = -1 # Assign a dummy ID

    all_linked_results = []

    print("Linking characters and speakers...")
    for index, row in tqdm(canonical_df.iterrows(), total=len(canonical_df), desc="Processing images"):
        image_path = row.get('image_path')
        # Prefer explicit COCO mapping columns from the canonical CSV if available
        image_key = None
        chosen_source = None
        chosen_source_value = None
        # 1) numeric COCO image id column (map via id2file)
        coco_id_colnames = ['coco_image_id', 'image_id', 'coco_id', 'imageId', 'img_id']
        for col in coco_id_colnames:
            if col in row and pd.notna(row[col]):
                try:
                    cid = int(row[col])
                    fname = id2file.get(cid)
                    if fname:
                        image_key = os.path.normpath(fname)
                        chosen_source = f'coco_id:{col}'
                        chosen_source_value = row[col]
                        break
                except Exception:
                    pass

        # 2) explicit coco file name column - prefer the exact canonical column the CSV uses
        if image_key is None:
            # Prefer the authoritative column you reported: 'predictions_json_filename'
            if 'predictions_json_filename' in row and pd.notna(row['predictions_json_filename']):
                image_key = os.path.normpath(str(row['predictions_json_filename']))
                chosen_source = 'coco_fname:predictions_json_filename'
                chosen_source_value = row['predictions_json_filename']
            else:
                # fallback to a small set of common alternatives if the canonical one isn't present
                coco_fname_cols = ['predictions_json', 'predictions_json_fname', 'pred_file', 'pred_file_name', 'coco_file_name', 'coco_filename', 'file_name']
                for col in coco_fname_cols:
                    if col in row and pd.notna(row[col]):
                        image_key = os.path.normpath(str(row[col]))
                        chosen_source = f'coco_fname:{col}'
                        chosen_source_value = row[col]
                        break

        # 3) fallback to CSV image_path basename
        if image_key is None and image_path:
            image_key = Path(image_path).name
            chosen_source = 'csv_image_path_basename'
            chosen_source_value = Path(image_path).name
        # At this point image_key may be a full/relative path (from a CSV column) or a bare basename.
        # For basename-based lookups we should always use the basename portion.
        image_key_basename = Path(image_key).name if image_key is not None else None
        vlm_json_path = os.path.join(args.vlm_json_dir, Path(row.get('vlm_json_filename', '')).name) # Construct full VLM JSON path
        
        if args.debug:
            print(f"[DEBUG] Processing row {index}: csv image_path={image_path}")
            print(f"         csv lookup image_key(raw)={image_key}")
            print(f"         csv lookup image_key(basename)={image_key_basename}")
            print(f"         chosen_source={chosen_source}")
            print(f"         chosen_source_value={chosen_source_value}")
            # find candidate COCO keys for this basename
            candidates = basename_index.get(image_key_basename, [])
            print(f"         candidates for basename ({image_key_basename}): {len(candidates)}")
            if len(candidates) > 0:
                # attempt to disambiguate using parent folder name
                parent_folder = Path(image_path).parent.name
                matched = [k for k in candidates if parent_folder in k]
                print(f"         parent_folder='{parent_folder}' matched candidates: {len(matched)}")
                # show some candidates (truncated)
                print("         sample candidates:", candidates[:5])
            else:
                print("         no candidates found for this basename in basename_index")
            print(f"         vlm_json_path={vlm_json_path} exists={os.path.exists(vlm_json_path)}")

        # Resolve chosen_key from basename_index and candidate disambiguation
        chosen_key = None
        # 1) try exact normalized CSV path match (using image_path)
        norm_csv_path = os.path.normpath(image_path) if image_path else None
        if norm_csv_path and norm_csv_path in coco_detections_by_image:
            chosen_key = norm_csv_path
        else:
            # 1b) try direct match using CSV-provided filename (image_key) normalized
            if image_key is not None:
                norm_image_key = os.path.normpath(image_key)
                if norm_image_key in coco_detections_by_image:
                    chosen_key = norm_image_key
            # 2) try basename candidates
            candidates = basename_index.get(image_key_basename, [])
            if len(candidates) == 1:
                chosen_key = candidates[0]
            elif len(candidates) > 1:
                # try match by parent folder name
                parent_folder = Path(image_path).parent.name
                matched = [k for k in candidates if parent_folder in k]
                if len(matched) == 1:
                    chosen_key = matched[0]
                elif len(matched) > 1:
                    # ambiguous even after parent folder filtering; pick best candidate heuristically
                    if args.debug:
                        print(f"[WARN] Ambiguous candidates for {image_key}: {len(matched)}; using first matched")
                        print("       matched candidates:", matched[:5])
                    chosen_key = matched[0]
                else:
                    # fallback to first candidate but log in debug mode
                    if args.debug:
                        print(f"[WARN] Multiple candidates for basename {image_key} but none matched parent folder; using first candidate")
                        print("       candidates:", candidates[:5])
                    chosen_key = candidates[0]
            else:
                # no candidates, leave chosen_key None
                chosen_key = None

        if args.debug:
            print(f"         chosen_key={chosen_key}")
            if chosen_key is None:
                print("         [DEBUG] No chosen_key resolved for this row.")
            else:
                present = chosen_key in coco_detections_by_image
                print(f"         chosen_key present in COCO index? {present}")
                if present:
                    num_dets = len(coco_detections_by_image[chosen_key])
                    print(f"         # detections for chosen_key: {num_dets}")
                    print("         sample coco detections (up to 10):")
                    for d in coco_detections_by_image[chosen_key][:10]:
                        print(f"           {d}")
                else:
                    # try a few normalization variants to aid debugging
                    alt1 = os.path.normpath(chosen_key).replace('\\', '/')
                    alt2 = os.path.normpath(chosen_key).replace('/', '\\')
                    alt3 = Path(chosen_key).name
                    print(f"         tried variants: norm-> {alt1}, swap-> {alt2}, basename-> {alt3}")
                    print(f"         present? alt1:{alt1 in coco_detections_by_image}, alt2:{alt2 in coco_detections_by_image}, basename:{alt3 in coco_detections_by_image}")
                # also show a short VLM summary (panels, sample speakers/balloons)
                try:
                    if os.path.exists(vlm_json_path):
                        with open(vlm_json_path, 'r', encoding='utf-8') as vf:
                            _vlm = json.load(vf)
                        panels = _vlm.get('panels', []) if isinstance(_vlm.get('panels', []), list) else []
                        print(f"         VLM panels: {len(panels)}")
                        sample_speakers = []
                        sample_balloons = []
                        for p in panels[:5]:
                            for s in p.get('speakers', [])[:3]:
                                sample_speakers.append({k: s.get(k) for k in ('character','dialogue') if k in s})
                            for b in p.get('balloons', [])[:3]:
                                sample_balloons.append({k: b.get(k) for k in ('bbox','text','text_content','raw_text') if k in b})
                        print(f"         sample_vlm_speakers: {sample_speakers}")
                        print(f"         sample_vlm_balloons: {sample_balloons}")
                    else:
                        print(f"         VLM JSON file not found at {vlm_json_path}")
                except Exception as _e:
                    print(f"         Error reading VLM JSON for debug: {_e}")

        # If we still don't have a chosen_key, try a substring/variant match using the chosen_source_value
        if chosen_key is None and chosen_source_value:
            sval = str(chosen_source_value)
            # try simple substring matches against the COCO keys
            substring_matches = [k for k in coco_detections_by_image.keys() if sval in k or Path(k).name == sval or k.endswith(sval)]
            if len(substring_matches) == 1:
                if args.debug:
                    print(f"         [INFO] Resolved chosen_key by substring_match: {substring_matches[0]}")
                chosen_key = substring_matches[0]
            elif len(substring_matches) > 1:
                # prefer matches that contain the parent folder name
                parent_folder = Path(image_path).parent.name if image_path else None
                matched2 = [k for k in substring_matches if parent_folder and parent_folder in k]
                if len(matched2) >= 1:
                    if args.debug:
                        print(f"         [INFO] Resolved chosen_key by substring+parent_folder: {matched2[0]}")
                    chosen_key = matched2[0]
                else:
                    # try a common variant: parent_folder + '_' + sval (helps map JPG4CBZ\0001.jpg -> JPG4CBZ_0001.jpg)
                    if parent_folder:
                        # construct a safe variant replacing path separators with underscores
                        variant = parent_folder + "_" + sval.replace('\\', '_').replace('/', '_')
                        var_matches = [k for k in coco_detections_by_image.keys() if variant in k]
                        if var_matches:
                            if args.debug:
                                print(f"         [INFO] Resolved chosen_key by variant match: {var_matches[0]}")
                            chosen_key = var_matches[0]

        linked_data = link_data_for_image(chosen_key, coco_detections_by_image, vlm_json_path, coco_categories, debug=args.debug)
        # keep original CSV path in output
        linked_data['csv_image_path'] = image_path
        all_linked_results.append(linked_data)

    # Flatten results for CSV output
    flat_results = []
    for img_res in all_linked_results:
        if not img_res['linked_characters']:
            flat_results.append({
                'image_path': img_res.get('csv_image_path', img_res.get('image_path')),
                'coco_image_key': img_res.get('image_path'),
                'vlm_json_path': img_res['vlm_json_path'],
                'balloon_bbox': None,
                'balloon_score': None,
                'linked_char_bbox': None,
                'linked_char_score': None,
                'linked_char_id': None,
                'linked_face_bbox': None,
                'linked_face_score': None,
                'linked_face_id': None,
                'vlm_speaker_name': None
            })
        else:
            for char_link in img_res['linked_characters']:
                flat_results.append({
                    'image_path': img_res.get('csv_image_path', img_res.get('image_path')),
                    'coco_image_key': img_res.get('image_path'),
                    'vlm_json_path': img_res['vlm_json_path'],
                    **char_link
                })

    output_df = pd.DataFrame(flat_results)
    output_df.to_csv(args.output_csv, index=False)
    print(f"Linked results saved to {args.output_csv}")

if __name__ == "__main__":
    main()
