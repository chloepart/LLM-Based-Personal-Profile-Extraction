#!/usr/bin/env python3
"""
Generate baseline predictions only (without ground truth).

This script extracts profiles from Senate HTML files using all four baseline
methods and outputs only the extracted values (predictions) to a CSV file.

Output CSV Schema:
  - senator_id: ID of senator (e.g., "Bernie_Sanders_VT")
  - baseline: Baseline method used ("regex", "keyword", "spacy", "bert")
  - name: Extracted name
  - gender: Extracted gender
  - birthdate: Extracted birthdate (YYYY-MM-DD)
  - education: Extracted education (pipe-delimited: "BA|institution|year")
  - religion: Extracted religion
  - committee_roles: Extracted committee roles (semicolon-delimited)

Output: baseline_predictions_only.csv (400 rows × 8 columns)
"""

import os
import csv
from pathlib import Path
from typing import Optional, List, Dict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.evaluation.baselines_v2 import (
    extract_via_regex,
    extract_via_keyword,
    extract_via_spacy,
    extract_via_bert,
)


def serialize_education(education: Optional[List[Dict]]) -> Optional[str]:
    """Serialize education list to pipe-delimited string."""
    if not education:
        return None
    
    entries = []
    for edu in education:
        degree = edu.get("degree", "")
        institution = edu.get("institution", "")
        year = edu.get("year", "")
        entries.append(f"{degree}|{institution}|{year}")
    
    return "; ".join(entries) if entries else None


def serialize_committees(committees: Optional[List[str]]) -> Optional[str]:
    """Serialize committee roles list to semicolon-delimited string."""
    if not committees:
        return None
    return "; ".join(committees)


def run_predictions_only():
    """Generate baseline predictions for all senators."""
    
    # Paths
    workspace_root = Path(__file__).parent.parent
    html_dir = workspace_root / "external_data" / "senate_html"
    output_file = workspace_root / "outputs" / "baseline_predictions_only.csv"
    
    # ========================================================================
    # [1/4] Scan HTML directory
    # ========================================================================
    print(f"\n[1/4] Scanning HTML directory: {html_dir}...")
    html_files = sorted([f for f in html_dir.glob("*.html")])
    print(f"✓ Found {len(html_files)} HTML files\n")
    
    # ========================================================================
    # [2/4] Run baselines on all senators
    # ========================================================================
    print(f"[2/4] Running baselines on all senators...")
    print(f"      Baselines: regex, keyword, spacy, bert")
    
    results = []
    baselines_func = {
        "regex": extract_via_regex,
        "keyword": extract_via_keyword,
        "spacy": extract_via_spacy,
        "bert": extract_via_bert,
    }
    
    for idx, html_file in enumerate(html_files, 1):
        # Progress checkpoint every 20 files
        if idx % 20 == 0:
            print(f"  Processing {idx}/{len(html_files)} - {html_file.stem}")
        
        senator_id = html_file.stem
        
        # Read HTML
        try:
            with open(html_file, "r", encoding="utf-8") as f:
                html_content = f.read()
        except Exception as e:
            print(f"  ⚠ Error reading {html_file}: {e}")
            continue
        
        # Run all 4 baselines
        for baseline_name, baseline_func in baselines_func.items():
            try:
                extraction = baseline_func(html_content, senator_id=senator_id)
                
                results.append({
                    "senator_id": senator_id,
                    "baseline": baseline_name,
                    "name": extraction.name,
                    "gender": extraction.gender,
                    "birthdate": extraction.birthdate,
                    "education": serialize_education(extraction.education),
                    "religion": extraction.religion,
                    "committee_roles": serialize_committees(extraction.committee_roles),
                })
            except Exception as e:
                print(f"  ⚠ Error running {baseline_name} on {senator_id}: {e}")
                results.append({
                    "senator_id": senator_id,
                    "baseline": baseline_name,
                    "name": None,
                    "gender": None,
                    "birthdate": None,
                    "education": None,
                    "religion": None,
                    "committee_roles": None,
                })
    
    print(f"  ✓ Processed all {len(html_files)} senators with 4 baselines each\n")
    
    # ========================================================================
    # [3/4] Save to CSV
    # ========================================================================
    print(f"[3/4] Saving results to {output_file}...")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "senator_id",
        "baseline",
        "name",
        "gender",
        "birthdate",
        "education",
        "religion",
        "committee_roles",
    ]
    
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✓ Saved {len(results)} prediction rows to {output_file}\n")
    
    # ========================================================================
    # [4/4] Summary statistics
    # ========================================================================
    print("=" * 80)
    print("PREDICTION STATISTICS")
    print("=" * 80)
    
    for baseline_name in ["regex", "keyword", "spacy", "bert"]:
        baseline_results = [r for r in results if r["baseline"] == baseline_name]
        print(f"\n{baseline_name.upper()} Baseline:")
        
        for field in ["name", "gender", "birthdate", "education", "religion", "committee_roles"]:
            non_null_count = sum(1 for r in baseline_results if r[field] is not None)
            total_count = len(baseline_results)
            pct = (non_null_count / total_count * 100) if total_count > 0 else 0
            print(f"  - {field.capitalize():20} {non_null_count:3} / {total_count} ({pct:5.1f}%)")
    
    print("\n" + "=" * 80)
    print("✓ Predictions-only CSV generated successfully")
    print(f"  Output: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_predictions_only()
