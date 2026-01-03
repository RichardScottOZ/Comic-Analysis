#!/usr/bin/env python3
"""
Calculate VLM Costs for 1.22M Pages
Based on actual activity logs.
"""

import csv
import argparse
from collections import defaultdict

def parse_activity_log(csv_path, target_pages):
    stats = defaultdict(lambda: {'requests': 0, 'cost': 0.0, 'prompt_tokens': 0, 'completion_tokens': 0})
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            slug = row['Slug']
            
            requests = int(row['Requests'])
            if requests == 0: continue
            
            usage_cost = float(row['Usage'])
            p_tokens = int(row['Prompt Tokens'])
            c_tokens = int(row['Completion Tokens'])
            
            stats[slug]['requests'] += requests
            stats[slug]['cost'] += usage_cost
            stats[slug]['prompt_tokens'] += p_tokens
            stats[slug]['completion_tokens'] += c_tokens

    # Generate Report
    print(f"Generating Cost Report for {target_pages:,} pages based on {csv_path}...")
    
    report_lines = []
    report_lines.append(f"# VLM Cost Projection for {target_pages:,} Pages")
    report_lines.append(f"**Source Data:** `{csv_path}`\n")
    
    headers = ["Model", "Sample Size", "Avg Input Tok", "Avg Output Tok", "Cost/1k Pages", f"Total Cost ({target_pages/1e3:.0f}K)"]
    report_lines.append("| " + " | ".join(headers) + " |")
    report_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    
    results = []
    
    for slug, data in stats.items():
        reqs = data['requests']
        if reqs < 1: continue 
        
        avg_cost = data['cost'] / reqs
        avg_in = data['prompt_tokens'] / reqs
        avg_out = data['completion_tokens'] / reqs
        
        cost_per_1k = avg_cost * 1000
        total_projected = avg_cost * target_pages
        
        results.append({
            'slug': slug,
            'reqs': reqs,
            'avg_in': avg_in,
            'avg_out': avg_out,
            'cost_1k': cost_per_1k,
            'total': total_projected
        })
        
    # Sort by Total Cost
    results.sort(key=lambda x: x['total'])
    
    for r in results:
        line = f"| **{r['slug']}** | {r['reqs']} | {r['avg_in']:.0f} | {r['avg_out']:.0f} | ${r['cost_1k']:.4f} | **${r['total']:.2f}** |"
        report_lines.append(line)
        
    # Write to file
    with open("VLM_COST_COMPARISON_FULL.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print("Report saved to VLM_COST_COMPARISON_FULL.md")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='openrouter-activity-20260102-001729.csv')
    parser.add_argument('--target-pages', type=int, default=1220000, help="Number of pages to project costs for")
    args = parser.parse_args()
    
    parse_activity_log(args.csv, args.target_pages)
