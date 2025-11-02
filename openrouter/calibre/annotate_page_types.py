import argparse
import os
import re
import json
import zlib
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple


def infer_book_folder(image_path: str) -> str:
    p = image_path.replace('\\', '/')
    parts = p.split('/')
    # Prefer parent before common containers like JPG4CBZ/pages
    if len(parts) >= 3 and parts[-2].lower() in { 'jpg4cbz', 'pages', 'page' }:
        return parts[-3]
    if len(parts) >= 2:
        return parts[-2]
    return ''


def extract_page_num(image_path: str) -> Optional[int]:
    name = os.path.splitext(os.path.basename(image_path))[0].lower()
    # Some extracted images retain a double extension (e.g., *.jpg.png)
    name = re.sub(r'\.[a-z0-9]+$', '', name)

    try:
        pattern = re.compile(r'(?:jpg4cbz_)?(?:page[_\-]?)?(\d{1,6})(?![0-9a-z])', re.IGNORECASE)
        candidates = list(pattern.finditer(name))
    except Exception:
        return None

    if candidates:
        # Prefer the match closest to the end of the string.
        best = max(candidates, key=lambda m: (m.end(1), m.start(1)))
        try:
            return int(best.group(1))
        except Exception:
            return None
    return None


KEYWORDS = {
    'cover': [
        'cover', 'variant cover', 'main cover', 'front cover', 'back cover',
        'series title typography', 'logo', 'barcode', 'price', 'issue', 'no.'
    ],
    'splash': ['splash page', 'full-page', 'full page'],
    'title_page': [
        'title page', 'title', 'title:', 'title -', 'title —', 'title –', 'issue title',
        'story title', 'chapter title', 'series title', 'title of the story',
        'credits', 'written by', 'story by', 'art by', 'illustrated by',
        'lettered by', 'colors by', 'colours by', 'colored by', 'coloured by',
        'cover art', 'cover by', 'edited by', 'created by',
        'chapter', 'part one', 'part two', 'part three', 'prologue', 'epilogue'
    ],
    'ads': [
        'advertisement', 'advertising', 'promo', 'promotional', 'subscribe',
        'coming soon', 'on sale', 'available now', 'visit', 'www.', '.com',
        'facebook', 'twitter', 'instagram', 'hashtag', 'merchandise', 'trade paperback'
    ],
    'ad_third_party': [
        'www.', '.com', '.net', '.org', '@', 'facebook', 'twitter', 'instagram', 'youtube',
        'subscribe', 'statue', 'collectible', 'collectibles', 't-shirt', 'apparel', 'poster',
        'blu-ray', 'dvd', 'soundtrack', 'video game', 'tv series', 'movie', 'film', '$', '£', '€'
    ],
    'promo_inhouse': [
        'coming soon', 'on sale', 'available now', 'new series', 'now from', 'miniseries', 'one-shot',
        'issue #', 'in stores', 'pre-order', 'your local comic shop', 'lcs', 'from the creators of'
    ],
    'preview': [
        'preview', 'special preview', 'exclusive preview', 'first look', 'sample pages', 'turn the page'
    ],
    'subscription': [
        'subscribe', 'subscription', 'coupon', 'order form', 'mail-in', 'mail order', 'mail to',
        'send check', 'money order', 'check or money order', 'c.o.d.', 'no cod', 'allow 4-6 weeks', 'allow 6-8 weeks',
        'dept.', 'department', 'p.o. box', 'city', 'state', 'zip', 'offer expires', 'void where prohibited',
        'self-addressed stamped envelope', 's.a.s.e.'
    ],
    'vintage_ad_tokens': [
        'sea-monkeys', 'sea monkeys', 'x-ray specs', 'xray specs', '100 toy soldiers', 'toy soldiers',
        '7-foot', '7 foot', 'giant', 'polaris submarine', 'polaris', 'submarine', 'grit', 'charles atlas',
        'muscles', 'sneezing powder', 'stamp collection', 'comic book club'
    ],
    'letters': ['letters page', 'letter column', 'letters column', 'fan mail', 'mailbag', 'hellmail', 'letters to the editor'],
    'editorial': ['editorial', 'table of contents', 'contents', 'credits', 'staff'],
    'backmatter': [
        'backmatter', 'feature', 'features', 'essay', 'essays', 'interview', 'interviews',
        'article', 'articles', 'column', 'columns', 'creator commentary', 'commentary',
        'profile', 'spotlight', 'art of', 'sketchbook', 'pin-up', 'pinup', 'gallery',
        'process', 'behind the scenes', 'afterword', 'foreword', 'bonus material', 'bonus'
    ],
    'recap': ['previously', 'previously on', 'recap', 'last issue', 'summary']
}


def text_corpus_from_row(row, read_vlm_json: bool = False, use_inline_vlm: bool = False) -> str:
    buf = []
    # Try inline vlm_data if present as JSON-like string
    v = row.get('vlm_data')
    if use_inline_vlm and isinstance(v, str) and v:
        try:
            obj = json.loads(v)
        except Exception:
            # Try tolerant parsing for single-quoted or python-repr strings
            try:
                import ast
                obj = ast.literal_eval(v)
            except Exception:
                obj = None
        if isinstance(obj, dict):
            # Common fields
            for k in ['overall_summary', 'summary']:
                val = obj.get(k)
                if isinstance(val, str):
                    buf.append(val)
                elif isinstance(val, dict):
                    # include plot if available
                    plot = val.get('plot')
                    if isinstance(plot, str):
                        buf.append(plot)
                    chars = val.get('characters')
                    if isinstance(chars, list):
                        buf.extend([str(c) for c in chars])
            # Panels text
            pls = obj.get('panels')
            if isinstance(pls, list):
                for p in pls:
                    if isinstance(p, dict):
                        if isinstance(p.get('caption'), str):
                            buf.append(p['caption'])
                        if isinstance(p.get('dialogue'), str):
                            buf.append(p['dialogue'])
                        if isinstance(p.get('narration'), str):
                            buf.append(p['narration'])
                        if isinstance(p.get('text'), str):
                            buf.append(p['text'])
                        if isinstance(p.get('speakers'), list):
                            for sp in p['speakers']:
                                if isinstance(sp, dict) and isinstance(sp.get('dialogue'), str):
                                    buf.append(sp['dialogue'])
    # Optionally read from JSON path for better keywords
    elif read_vlm_json:
        jp = row.get('vlm_json_path')
        if isinstance(jp, str) and os.path.exists(jp):
            try:
                with open(jp, 'r', encoding='utf-8') as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    val = obj.get('overall_summary')
                    if isinstance(val, str):
                        buf.append(val)
                    summ = obj.get('summary')
                    if isinstance(summ, str):
                        buf.append(summ)
                    elif isinstance(summ, dict):
                        if isinstance(summ.get('plot'), str):
                            buf.append(summ['plot'])
                        chars = summ.get('characters')
                        if isinstance(chars, list):
                            buf.extend([str(c) for c in chars])
            except Exception:
                pass
    # Add path tokens
    buf.append(str(row.get('image_path', '')))
    return ' \n '.join(buf).lower()


def score_page_types(row, pos_pct: float, group_len: int, corpus: str) -> Tuple[str, float, dict]:
    rcnn = int(row.get('rcnn_panels', 0) or 0)
    vlm = int(row.get('vlm_panels', 0) or 0)
    dlg = int(row.get('vlm_text_dialogue_panels', 0) or 0)
    nar = int(row.get('vlm_text_narration_panels', 0) or 0)
    sfx = int(row.get('vlm_text_sfx_panels', 0) or 0)
    txtcov = float(row.get('text_coverage', 0.0) or 0.0)

    scores = {
        'cover': 0.0,
        'back_cover': 0.0,
        'back_cover_ad': 0.0,
        'splash': 0.0,
        'title_page': 0.0,
        'ads': 0.0,
        'ad_third_party': 0.0,
        'promo_inhouse': 0.0,
        'preview_comic_pages': 0.0,
        'subscription_coupon': 0.0,
        'letters': 0.0,
        'editorial': 0.0,
        'backmatter': 0.0,
        'recap': 0.0,
        'interior': 0.0
    }

    title_hits = any(k in corpus for k in KEYWORDS['title_page'])

    # Cover/front heuristic
    if pos_pct <= 0.03:
        scores['cover'] += 2.0
    if rcnn <= 1 or vlm <= 2:
        scores['cover'] += 0.5
    if any(k in corpus for k in KEYWORDS['cover']):
        scores['cover'] += 2.0
        scores['backmatter'] -= 1.0

    # Back cover heuristic
    if pos_pct >= 0.97:
        scores['back_cover'] += 2.0
    if (rcnn <= 1 or vlm <= 2) and pos_pct >= 0.9:
        scores['back_cover'] += 0.5
    if any(k in corpus for k in ['back cover', 'barcode']):
        scores['back_cover'] += 1.5

    # Title page (distinct from splash)
    if pos_pct <= 0.3 and (rcnn <= 3 or vlm <= 3) and title_hits:
        scores['title_page'] += 2.5
    if title_hits:
        scores['title_page'] += 1.0
        if nar >= 1 and dlg <= 2:
            scores['title_page'] += 0.5
    # Splash page (single-panel impact page, not necessarily title)
    if 0.05 <= pos_pct <= 0.95 and (rcnn <= 1 or vlm <= 1) and not any(k in corpus for k in KEYWORDS['title_page']):
        scores['splash'] += 1.8
    if any(k in corpus for k in KEYWORDS['splash']):
        scores['splash'] += 0.7

    # Ads/Promos (generic)
    if pos_pct >= 0.8:
        scores['ads'] += 0.8
    if dlg == 0 and (nar == 0 or txtcov < 0.2):
        scores['ads'] += 0.7
    if any(k in corpus for k in KEYWORDS['ads']):
        scores['ads'] += 2.0

    # Distinguish third-party ads vs in-house promos
    if any(k in corpus for k in KEYWORDS['ad_third_party']):
        scores['ad_third_party'] += 2.0
        if dlg == 0 and txtcov < 0.3:
            scores['ad_third_party'] += 0.7
        if pos_pct >= 0.75:
            scores['ad_third_party'] += 0.3

    if any(k in corpus for k in KEYWORDS['promo_inhouse']):
        scores['promo_inhouse'] += 1.6
        if pos_pct >= 0.6:
            scores['promo_inhouse'] += 0.3
        if nar >= 1 and dlg <= 1:
            scores['promo_inhouse'] += 0.3

    # Preview pages from another comic (actual interior-like pages)
    if any(k in corpus for k in KEYWORDS['preview']) and rcnn >= 3 and dlg >= 1 and txtcov >= 0.3:
        scores['preview_comic_pages'] += 2.5
        if pos_pct >= 0.6:
            scores['preview_comic_pages'] += 0.3

    # Subscriptions/Coupons (older magazines/comics)
    if any(k in corpus for k in KEYWORDS['subscription']) and not title_hits:
        scores['subscription_coupon'] += 2.5
        scores['ad_third_party'] += 0.5  # often third-party-like page layout
        if pos_pct >= 0.6:
            scores['subscription_coupon'] += 0.3
    if any(k in corpus for k in KEYWORDS['vintage_ad_tokens']):
        scores['ad_third_party'] += 1.2
        if pos_pct >= 0.6:
            scores['ad_third_party'] += 0.2

    # Letters page (fan mail columns)
    if any(k in corpus for k in KEYWORDS['letters']):
        scores['letters'] += 3.0
    # Letters pages often have lots of text but limited dialogue bubbles; narration-like blocks
    if nar >= 1 and dlg <= 1 and txtcov >= 0.6:
        scores['letters'] += 0.8

    # Backmatter (essays/interviews/articles/features, etc.)
    if any(k in corpus for k in KEYWORDS['backmatter']):
        scores['backmatter'] += 2.5
    # Backmatter often later in the book, text-heavy with limited dialogue
    if pos_pct >= 0.5:
        scores['backmatter'] += 0.5
    if nar >= 1 and txtcov >= 0.5 and dlg <= 1:
        scores['backmatter'] += 0.7

    # Early pages should never be backmatter; reinforce cover label.
    page_num = row.get('page_num')
    if pos_pct <= 0.08 or (isinstance(page_num, (int, float)) and page_num is not None and page_num <= 1):
        scores['backmatter'] = min(scores['backmatter'], -5.0)
        if scores['cover'] < 5.0:
            scores['cover'] = max(scores['cover'], 5.0)
        scores['subscription_coupon'] = min(scores['subscription_coupon'], 0.0)

    # Editorial/toc/credits
    if pos_pct <= 0.1 and nar >= 1 and dlg == 0:
        scores['editorial'] += 1.0
    if any(k in corpus for k in KEYWORDS['editorial']):
        scores['editorial'] += 2.0

    # Recap
    if pos_pct <= 0.15 and any(k in corpus for k in KEYWORDS['recap']):
        scores['recap'] += 2.0

    # Interior baseline (default content pages)
    if 0.05 <= pos_pct <= 0.95 and (rcnn >= 2 or vlm >= 2) and dlg >= 1:
        scores['interior'] += 1.5
    if txtcov >= 0.4 and dlg >= 1:
        scores['interior'] += 0.7

    # Choose best
    page_type = max(scores.items(), key=lambda x: x[1])[0]
    confidence = scores[page_type]
    return page_type, confidence, scores


def annotate_page_types(
    input_csv: str,
    output_csv: Optional[str] = None,
    read_vlm_json: bool = False,
    use_inline_vlm: bool = False,
    return_vlm_json: bool = False,
    num_splits: int = 1,
    index: int = 0,
    combine_splits: bool = False,
    only_combine: bool = False,
    split_output_prefix: Optional[str] = None,
):
    if only_combine:
        if num_splits <= 1:
            print("Cannot combine splits if num_splits is 1 or less.")
            return
        if split_output_prefix is None:
            print("Error: split_output_prefix must be provided when combining splits.")
            return
        
        print(f"Combining {num_splits} splits with prefix '{split_output_prefix}'...")
        all_split_dfs = []
        for i in range(num_splits):
            split_file_path = Path(f"{split_output_prefix}{i}_with_page_types.csv")
            if not split_file_path.exists():
                print(f"Warning: Split file not found: {split_file_path}. Skipping.")
                continue
            try:
                all_split_dfs.append(pd.read_csv(split_file_path))
            except Exception as e:
                print(f"Error reading split file {split_file_path}: {e}. Skipping.")
        
        if not all_split_dfs:
            print("No split files found or readable to combine. Nothing to do.")
            return
            
        combined_df = pd.concat(all_split_dfs, ignore_index=True)
        combined_output_csv = output_csv if output_csv else input_csv.replace('.csv', '_combined.csv')
        combined_df.to_csv(combined_output_csv, index=False)
        print(f"Successfully combined {len(all_split_dfs)} splits into: {combined_output_csv}")
        return # Exit after combining

    # --- Existing annotation logic starts here (only runs if not only_combine) ---
    df = pd.read_csv(input_csv)
    if output_csv is None:
        output_csv = input_csv.replace('.csv', '_with_page_types.csv')

    # Group by inferred book folder
    df['book_folder'] = df['image_path'].apply(infer_book_folder)
    df['page_num'] = df['image_path'].apply(extract_page_num)

    # Optional split processing by stable hash of book folder (to parallelize runs)
    if num_splits is not None and int(num_splits) > 1:
        num_splits = int(num_splits)
        index = int(index)
        if index < 0 or index >= num_splits:
            raise ValueError(f"index must be in [0, {num_splits-1}], got {index}")
        def _bucket(book: str) -> int:
            if not isinstance(book, str):
                book = str(book)
            return zlib.adler32(book.encode('utf-8')) % num_splits
        mask = df['book_folder'].apply(_bucket) == index
        df = df[mask].copy()
        if df.empty:
            print(f"No rows selected for split {index}/{num_splits}. Nothing to do.")
            return

    page_types = []
    confidences = []
    pos_pcts = []
    vlm_json_contents = [] # New list to store VLM JSONs

    for book, g in df.groupby('book_folder'):
        g_sorted = g.sort_values(by=['page_num'], kind='mergesort')  # stable
        n = len(g_sorted)
        # Build quick lookup from original index to percentile
        idx_to_pct = {}
        for i, (idx, row) in enumerate(g_sorted.iterrows(), start=1):
            pct = (i - 1) / max(1, n - 1) if n > 1 else 0.0
            idx_to_pct[idx] = pct
        # Determine first and last page indices for overrides
        first_idx = g_sorted.index[0] if n >= 1 else None
        last_idx = g_sorted.index[-1] if n >= 1 else None
        for idx, row in g.iterrows():
            pos_pct = idx_to_pct.get(idx, 0.0)
            
            # Load VLM JSON content if requested
            current_vlm_json_obj = None
            if read_vlm_json:
                jp = row.get('vlm_json_path')
                if isinstance(jp, str) and os.path.exists(jp):
                    try:
                        with open(jp, 'r', encoding='utf-8') as f:
                            current_vlm_json_obj = json.load(f)
                    except Exception as e:
                        print(f"Warning: Could not load VLM JSON from {jp}: {e}")
            
            corpus = text_corpus_from_row(row, read_vlm_json=read_vlm_json, use_inline_vlm=use_inline_vlm)
            pg_type, conf, _scores = score_page_types(row, pos_pct, n, corpus)
            if idx != last_idx and pg_type in {'back_cover', 'back_cover_ad'}:
                fallback = sorted(
                    (
                        (k, v) for k, v in _scores.items()
                        if k not in {'back_cover', 'back_cover_ad'}
                    ),
                    key=lambda kv: kv[1]
                )
                if fallback:
                    pg_type, conf = fallback[-1]
            # Strong override: first page = cover, last page = back_cover (unless an ad is very strong)
            if idx == first_idx:
                pg_type, conf = 'cover', 10.0
            elif idx == last_idx:
                # If the ad-related scores are high, label as back_cover_ad
                ad_strength = _scores.get('ad_third_party', 0) + _scores.get('promo_inhouse', 0) + _scores.get('subscription_coupon', 0)
                if ad_strength >= 2.5:
                    pg_type, conf = 'back_cover_ad', 10.0
                else:
                    pg_type, conf = 'back_cover', 10.0
            page_types.append((idx, pg_type))
            confidences.append((idx, conf))
            pos_pcts.append((idx, pos_pct))
            vlm_json_contents.append((idx, current_vlm_json_obj)) # Store VLM JSON

    # Assign back to df preserving original order
    type_map = dict(page_types)
    conf_map = dict(confidences)
    pct_map = dict(pos_pcts)
    vlm_json_map = dict(vlm_json_contents) # New map for VLM JSONs

    df['page_type'] = df.index.map(type_map)
    df['page_type_conf'] = df.index.map(conf_map)
    df['page_pos_pct'] = df.index.map(pct_map)
    df['is_front_cover'] = (df['page_type'] == 'cover')
    df['is_back_cover'] = (df['page_type'].isin(['back_cover', 'back_cover_ad']))
    
    if return_vlm_json: # Conditionally add VLM JSON column
        df['vlm_json_content'] = df.index.map(vlm_json_map)
        # Convert dicts to JSON strings for CSV compatibility
        df['vlm_json_content'] = df['vlm_json_content'].apply(lambda x: json.dumps(x) if x else None)

    df.to_csv(output_csv, index=False)
    # Print summary
    counts = df['page_type'].value_counts(dropna=False).to_dict()
    print('Page type distribution:')
    for k, v in counts.items():
        print(f'  {k}: {v}')
    print(f"Saved: {output_csv}")

    # If combine_splits is true, combine them after processing
    if combine_splits:
        if num_splits <= 1:
            print("Cannot combine splits if num_splits is 1 or less.")
            return
        if split_output_prefix is None:
            print("Error: split_output_prefix must be provided when combining splits.")
            return
        
        print(f"Combining {num_splits} splits with prefix '{split_output_prefix}'...")
        all_split_dfs = []
        for i in range(num_splits):
            split_file_path = Path(f"{split_output_prefix}{i}_with_page_types.csv")
            if not split_file_path.exists():
                print(f"Warning: Split file not found: {split_file_path}. Skipping.")
                continue
            try:
                all_split_dfs.append(pd.read_csv(split_file_path))
            except Exception as e:
                print(f"Error reading split file {split_file_path}: {e}. Skipping.")
        
        if not all_split_dfs:
            print("No split files found or readable to combine. Nothing to do.")
            return
            
        combined_df = pd.concat(all_split_dfs, ignore_index=True)
        combined_output_csv = output_csv if output_csv else input_csv.replace('.csv', '_combined.csv')
        combined_df.to_csv(combined_output_csv, index=False)
        print(f"Successfully combined {len(all_split_dfs)} splits into: {combined_output_csv}")
        return # Exit after combining


def main():
    ap = argparse.ArgumentParser(description='Annotate page types (cover/interior/ads/etc.) from analysis CSV')
    ap.add_argument('--input_csv', required=True, help='Path to analysis CSV (e.g., calibre_rcnn_vlm_analysis_v2.csv)')
    ap.add_argument('--output_csv', help='Output CSV path; defaults to *_with_page_types.csv')
    ap.add_argument('--read_vlm_json', action='store_true', help='Also read VLM JSON files to improve keyword detection (slower)')
    ap.add_argument('--use_inline_vlm', action='store_true', help='Parse inline VLM JSON in the CSV (much slower, better keywords). Default: off')
    ap.add_argument('--return_vlm_json', action='store_true', help='If --read_vlm_json is true, also include the full VLM JSON content as a column in the output CSV.') # New argument
    ap.add_argument('--num_splits', type=int, default=1, help='Split the work by book folder into N parts (for parallel runs). Default: 1')
    ap.add_argument('--index', type=int, default=0, help='Index of the split to process (0-based). Default: 0')
    ap.add_argument('--combine_splits', action='store_true', help='After processing, combine all split output CSVs into a single file.') # New argument
    ap.add_argument('--only_combine', action='store_true', help='Only combine previously generated split CSVs, skipping the annotation process.') # New argument
    ap.add_argument('--split_output_prefix', help='Prefix for split output CSVs when combining (e.g., path/to/my_output_split). Required if --combine_splits or --only_combine is used.') # New argument
    args = ap.parse_args()
    
    if args.only_combine and not args.split_output_prefix:
        ap.error("--split_output_prefix is required when --only_combine is used.")

    annotate_page_types(
        args.input_csv,
        args.output_csv,
        read_vlm_json=args.read_vlm_json,
        use_inline_vlm=args.use_inline_vlm,
        return_vlm_json=args.return_vlm_json,
        num_splits=args.num_splits,
        index=args.index,
        combine_splits=args.combine_splits,
        only_combine=args.only_combine,
        split_output_prefix=args.split_output_prefix,
    )


if __name__ == '__main__':
    main()
