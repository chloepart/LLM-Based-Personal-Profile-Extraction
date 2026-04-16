"""
Runner script for comprehensive baseline comparison on Senate HTML files.

Processes all senator HTML files from external_data/senate_html/ and runs
all four baselines (Regex, Keyword, spaCy, BERT) on each, comparing
predictions against ground truth from senate_ground_truth_updated_manual.csv.

Output: CSV file with predictions from all baselines, alongside ground truth
for easy comparison and scoring.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json
import warnings

# Suppress SSL warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add workspace to path
WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from modules.evaluation.baselines_v2 import (
    UnifiedExtraction,
    extract_via_regex,
    extract_via_keyword,
    extract_via_spacy,
    extract_via_bert,
)


# ============================================================================
# RUNNER CONFIGURATION
# ============================================================================

HTML_DIR = WORKSPACE_ROOT / "external_data" / "senate_html"
GROUND_TRUTH_CSV = WORKSPACE_ROOT / "external_data" / "ground_truth" / "senate_ground_truth_updated_manual.csv"
OUTPUT_DIR = WORKSPACE_ROOT / "outputs"
OUTPUT_CSV = OUTPUT_DIR / "baseline_predictions_comparison.csv"

# List of baselines to run (in order of increasing complexity)
BASELINES = [
    ("regex", extract_via_regex),
    ("keyword", extract_via_keyword),
    ("spacy", extract_via_spacy),
    ("bert", extract_via_bert),
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def senator_id_from_filename(filename: str) -> str:
    """
    Extract senator_id from HTML filename.
    
    Example: "Bernie_Sanders_VT.html" → "Bernie_Sanders_VT"
    """
    return Path(filename).stem


def extract_education_for_csv(education: Optional[List[Dict]]) -> Optional[str]:
    """Serialize education list to CSV-friendly string."""
    if not education:
        return None
    
    entries = []
    for edu in education:
        degree = edu.get("degree", "")
        institution = edu.get("institution", "")
        year = edu.get("year", "")
        entries.append(f"{degree}|{institution}|{year}")
    
    return "; ".join(entries) if entries else None


def extract_committees_for_csv(committees: Optional[List[str]]) -> Optional[str]:
    """Serialize committee roles list to CSV-friendly string."""
    if not committees:
        return None
    return "; ".join(committees)


def run_baseline(method_name: str, extract_func, html: str, senator_id: str) -> UnifiedExtraction:
    """
    Run a single baseline extraction method with error handling.
    
    Args:
        method_name: Name of baseline (for logging)
        extract_func: Extraction function
        html: HTML content
        senator_id: Senator identifier
    
    Returns:
        UnifiedExtraction result or empty result on error
    """
    try:
        result = extract_func(html, senator_id=senator_id)
        return result
    except Exception as e:
        print(f"  ⚠ {method_name} failed for {senator_id}: {str(e)[:60]}")
        # Return empty result on error
        return UnifiedExtraction(baseline=method_name, senator_id=senator_id)


def load_ground_truth(csv_path: Path) -> pd.DataFrame:
    """
    Load ground truth data from CSV.
    
    Args:
        csv_path: Path to ground truth CSV
    
    Returns:
        DataFrame with columns: senator_id, name, gender, birthdate, education, religion, committee_roles
    """
    if not csv_path.exists():
        print(f"⚠ Ground truth file not found: {csv_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    
    # Ensure senator_id column (may be first column without header)
    if "senator_id" not in df.columns:
        df.rename(columns={df.columns[0]: "senator_id"}, inplace=True)
    
    return df


def normalize_senator_id(filename: str, gt_df: pd.DataFrame) -> Optional[str]:
    """
    Find matching senator_id in ground truth by filename or name.
    
    Example: "Bernie_Sanders_VT.html" → find matching row in GT
    """
    # Extract from filename
    file_senator_id = senator_id_from_filename(filename)
    
    # Try exact match in GT
    if file_senator_id in gt_df["senator_id"].values:
        return file_senator_id
    
    # Try fallback: match by last_name_state in filename
    return file_senator_id


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main runner: process all senators, run all baselines, output results."""
    
    print("=" * 80)
    print("BASELINE COMPARISON RUNNER FOR SENATE HTML FILES")
    print("=" * 80)
    
    # Load ground truth
    print(f"\n[1/5] Loading ground truth from {GROUND_TRUTH_CSV}...")
    gt_df = load_ground_truth(GROUND_TRUTH_CSV)
    
    if gt_df.empty:
        print("✗ Ground truth data is empty. Exiting.")
        return
    
    print(f"✓ Loaded ground truth for {len(gt_df)} senators")
    
    # Find all HTML files
    print(f"\n[2/5] Scanning HTML directory: {HTML_DIR}...")
    html_files = sorted(glob.glob(str(HTML_DIR / "*.html")))
    
    if not html_files:
        print(f"✗ No HTML files found in {HTML_DIR}")
        return
    
    print(f"✓ Found {len(html_files)} HTML files")
    
    # Prepare output structure
    print(f"\n[3/5] Running baselines on all senators...")
    print(f"      Baselines: {', '.join([b[0] for b in BASELINES])}")
    
    results_rows = []
    
    for i, html_file in enumerate(html_files, 1):
        senator_id = normalize_senator_id(html_file, gt_df)
        
        if i % 20 == 0 or i == len(html_files):
            print(f"  Processing {i}/{len(html_files)} - {senator_id}")
        
        # Load HTML
        try:
            with open(html_file, "r", encoding="utf-8") as f:
                html = f.read()
        except Exception as e:
            print(f"  ✗ Failed to read {html_file}: {e}")
            continue
        
        # Run each baseline
        for baseline_name, extract_func in BASELINES:
            result = run_baseline(baseline_name, extract_func, html, senator_id)
            
            # Get ground truth for this senator (if available)
            gt_row = gt_df[gt_df["senator_id"] == senator_id]
            
            # Build output row
            row = {
                "senator_id": senator_id,
                "baseline": baseline_name,
                # Predictions
                "pred_name": result.name,
                "pred_gender": result.gender,
                "pred_birthdate": result.birthdate,
                "pred_education": extract_education_for_csv(result.education),
                "pred_religion": result.religion,
                "pred_committee_roles": extract_committees_for_csv(result.committee_roles),
            }
            
            # Ground truth (if available)
            if not gt_row.empty:
                gt_vals = gt_row.iloc[0]
                row["gt_name"] = gt_vals.get("full_name") if "full_name" in gt_vals.index else None
                row["gt_gender"] = gt_vals.get("gender") if "gender" in gt_vals.index else None
                row["gt_birthdate"] = gt_vals.get("birthdate") if "birthdate" in gt_vals.index else None
                row["gt_education"] = gt_vals.get("education") if "education" in gt_vals.index else None
                row["gt_religion"] = gt_vals.get("religion") if "religion" in gt_vals.index else None
                row["gt_committee_roles"] = gt_vals.get("committee_roles") if "committee_roles" in gt_vals.index else None
            else:
                row["gt_name"] = None
                row["gt_gender"] = None
                row["gt_birthdate"] = None
                row["gt_education"] = None
                row["gt_religion"] = None
                row["gt_committee_roles"] = None
            
            results_rows.append(row)
    
    # Convert to DataFrame
    print(f"\n[4/5] Compiling results ({len(results_rows)} rows)...")
    results_df = pd.DataFrame(results_rows)
    
    # Create output directory if needed
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    print(f"\n[5/5] Saving results to {OUTPUT_CSV}...")
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✓ Saved {len(results_df)} prediction rows to {OUTPUT_CSV}")
    
    # Print summary statistics
    print(f"\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    for baseline in [b[0] for b in BASELINES]:
        baseline_rows = results_df[results_df["baseline"] == baseline]
        
        pred_name_filled = baseline_rows["pred_name"].notna().sum()
        pred_gender_filled = baseline_rows["pred_gender"].notna().sum()
        pred_birthdate_filled = baseline_rows["pred_birthdate"].notna().sum()
        pred_education_filled = baseline_rows["pred_education"].notna().sum()
        pred_religion_filled = baseline_rows["pred_religion"].notna().sum()
        pred_committees_filled = baseline_rows["pred_committee_roles"].notna().sum()
        
        total = len(baseline_rows)
        
        print(f"\n{baseline.upper()} Baseline:")
        print(f"  - Name:        {pred_name_filled:3d} / {total} ({100*pred_name_filled/total:.1f}%)")
        print(f"  - Gender:      {pred_gender_filled:3d} / {total} ({100*pred_gender_filled/total:.1f}%)")
        print(f"  - Birthdate:   {pred_birthdate_filled:3d} / {total} ({100*pred_birthdate_filled/total:.1f}%)")
        print(f"  - Education:   {pred_education_filled:3d} / {total} ({100*pred_education_filled/total:.1f}%)")
        print(f"  - Religion:    {pred_religion_filled:3d} / {total} ({100*pred_religion_filled/total:.1f}%)")
        print(f"  - Committees:  {pred_committees_filled:3d} / {total} ({100*pred_committees_filled/total:.1f}%)")
    
    print(f"\n" + "=" * 80)
    print(f"✓ Runner completed successfully")
    print(f"  Output: {OUTPUT_CSV}")
    print("=" * 80)


if __name__ == "__main__":
    main()
