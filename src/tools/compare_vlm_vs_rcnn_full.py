import pandas as pd
import json
import collections

PARQUET = "documentation/plots/umap_stage3_pages.parquet"
RCNN_METADATA = "E:/Comic_Analysis_Results_v2/stage3_metadata.json"
OUT_FILE = "documentation/plots/vlm_vs_rcnn_full_comparison.txt"

def normalize_key(cid):
    # Strip common prefixes
    for prefix in ["CalibreComics_extracted/", "CalibreComics_extracted_20251107/", "CalibreComics_extracted\\", "amazon/"]:
        if cid.startswith(prefix):
            cid = cid[len(prefix):]
    cid = cid.lower()
    for ext in ('.jpg.png', '.png', '.jpg', '.jpeg'):
        if cid.endswith(ext):
            cid = cid[:-len(ext)]
            break
    return cid.replace('/', '_').replace('\\', '_').strip()

def main():
    print("Loading VLM parquet data...")
    df_vlm = pd.read_parquet(PARQUET)
    
    print("Loading RCNN metadata JSON...")
    with open(RCNN_METADATA, 'r', encoding='utf-8') as f:
        data_rcnn = json.load(f)
        
    print("Indexing RCNN metadata...")
    rcnn_map = {}
    for item in data_rcnn:
        norm_k = normalize_key(item['canonical_id'])
        rcnn_map[norm_k] = item['num_panels']
        
    print("Matching datasets...")
    vlm_counts = []
    rcnn_counts = []
    matched_ids = []
    missing_rcnn_count = 0
    
    for _, row in df_vlm.iterrows():
        cid = row['canonical_id']
        vlm_n = row['panel_count']
        norm_k = normalize_key(cid)
        
        if norm_k in rcnn_map:
            vlm_counts.append(vlm_n)
            rcnn_counts.append(rcnn_map[norm_k])
            matched_ids.append(cid)
        else:
            missing_rcnn_count += 1
            
    total_vlm = len(df_vlm)
    matched = len(matched_ids)
    print(f"Matched {matched:,} / {total_vlm:,} pages ({100*matched/total_vlm:.2f}%)")
    print(f"Missing RCNN counterpart for {missing_rcnn_count:,} pages")
    
    if matched == 0:
        print("No matched keys. Exiting.")
        return
        
    # Analyze differences
    vlm_s = pd.Series(vlm_counts)
    rcnn_s = pd.Series(rcnn_counts)
    diffs = vlm_s - rcnn_s
    
    avg_vlm = vlm_s.mean()
    avg_rcnn = rcnn_s.mean()
    
    vlm_more = (diffs > 0).sum()
    rcnn_more = (diffs < 0).sum()
    tied = (diffs == 0).sum()
    
    exact_match = (diffs == 0).mean() * 100
    off_by_one = (diffs.abs() <= 1).mean() * 100
    diff_greater_one = (diffs.abs() > 1).mean() * 100
    
    # Stored panel_count=1 (VLM fallback/grounding fail) where RCNN was > 1
    vlm_fallback_rcnn_more = ((vlm_s == 1) & (rcnn_s > 1)).sum()
    vlm_fallback_total = (vlm_s == 1).sum()
    
    diff_counts = collections.Counter(diffs)
    
    lines = []
    lines.append("=============================================================")
    lines.append("VLM VS RCNN PANEL COUNT COMPARISON REPORT (FULL DATASET)")
    lines.append("=============================================================")
    lines.append(f"Total matched pages: {matched:,} / {total_vlm:,} ({100*matched/total_vlm:.2f}%)")
    lines.append(f"Missing RCNN match:  {missing_rcnn_count:,} pages")
    lines.append("")
    lines.append("PANEL COUNT STATISTICS")
    lines.append("-------------------------------------------------------------")
    lines.append(f"Avg VLM panels:   {avg_vlm:.2f}")
    lines.append(f"Avg RCNN panels:  {avg_rcnn:.2f}")
    lines.append(f"VLM > RCNN (VLM grounded more):  {vlm_more:,} ({100*vlm_more/matched:.1f}%)")
    lines.append(f"RCNN > VLM (RCNN detected more): {rcnn_more:,} ({100*rcnn_more/matched:.1f}%)")
    lines.append(f"Tied (equal count):              {tied:,} ({100*tied/matched:.1f}%)")
    lines.append("")
    lines.append("ACCURACY BREAKDOWN")
    lines.append("-------------------------------------------------------------")
    lines.append(f"Exact match:       {exact_match:.2f}%")
    lines.append(f"Off by 1:          {off_by_one:.2f}%")
    lines.append(f"Diff > 1:          {diff_greater_one:.2f}%")
    lines.append("")
    lines.append("VLM GROUNDING FALLBACK ANALYSIS")
    lines.append("-------------------------------------------------------------")
    lines.append(f"VLM panel_count=1: {vlm_fallback_total:,} pages")
    pct_fallback = 100 * vlm_fallback_rcnn_more / vlm_fallback_total if vlm_fallback_total > 0 else 0
    lines.append(f"VLM panel_count=1 where RCNN detected > 1: {vlm_fallback_rcnn_more:,} pages ({pct_fallback:.2f}%)")
    lines.append("")
    lines.append("DIFFERENCE DISTRIBUTION (VLM_n - RCNN_n)")
    lines.append("-------------------------------------------------------------")
    for diff, cnt in sorted(diff_counts.items(), key=lambda x: -x[1])[:15]:
        pct = 100 * cnt / matched
        bar = '#' * min(int(pct * 2), 40)
        lines.append(f"  {diff:+3d}: {cnt:7,}  ({pct:5.2f}%)  {bar}")
        
    text = "\n".join(lines)
    print(text)
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"\nReport saved to: {OUT_FILE}")

if __name__ == '__main__':
    main()
