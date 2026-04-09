"""
Evaluation suite with result caching for quick notebook reruns.

Computes accuracy metrics for LLM predictions vs ground truth, with:
- JSON caching of computed metrics
- Overwrite flags for recomputation
- Structured output for visualization
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from rouge_score import rouge_scorer
import bert_score

from .evaluator import (
    match_by_fuzzy_name,
    gender_match_score,
    religion_match_score,
)
from .parsing import (
    birthdate_scores,
    compare_education_components,
    parse_education,
    parse_committee_roles,
)


def load_and_merge_results(
    pred_path: Path,
    gt_path: Path,
    merge_method: str = "exact"
) -> Dict[str, pd.DataFrame]:
    """
    Load and merge predictions with ground truth.
    
    Args:
        pred_path: Path to task1_pii.csv (predictions)
        gt_path: Path to senate_ground_truth.csv (ground truth)
        merge_method: "exact" (senator_id match) or "fuzzy" (name match)
        
    Returns:
        Dictionary with keys:
            - df_pred: Predictions dataframe
            - df_gt: Ground truth dataframe
            - merged_by_style: Dict[style_name -> merged dataframe]
    """
    
    df_pred = pd.read_csv(pred_path)
    df_gt = pd.read_csv(gt_path)
    
    print(f"✓ Loaded {len(df_pred)} predictions")
    print(f"✓ Loaded {len(df_gt)} ground truth records")
    print(f"✓ Prompt styles: {df_pred['prompt_style'].unique()}\n")
    
    # Merge by style
    merged_by_style = {}
    
    for style in df_pred["prompt_style"].unique():
        df_style = df_pred[df_pred["prompt_style"] == style]
        
        # Try exact merge first
        merged = df_gt.merge(df_style, on="senator_id", how="inner")
        
        # Fall back to fuzzy if needed
        if merged.empty and merge_method == "fuzzy":
            merged = match_by_fuzzy_name(df_gt, df_style)
        
        merged_by_style[style] = merged
    
    return {
        "df_pred": df_pred,
        "df_gt": df_gt,
        "merged_by_style": merged_by_style,
    }


def evaluate_all_styles(
    merged_by_style: Dict[str, pd.DataFrame],
    output_dir: Path = None,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Compute accuracy metrics for all prompt styles.
    
    Caches results to JSON for fast notebook reruns.
    
    Args:
        merged_by_style: Dict from load_and_merge_results()
        output_dir: Directory to cache results JSON
        overwrite: Whether to recompute if cache exists
        
    Returns:
        Dictionary with results for each style and shared metrics
    """
    
    # ─────────────────────────────────────────────────────────────────────
    # Check cache
    # ─────────────────────────────────────────────────────────────────────
    
    cache_path = None
    if output_dir:
        output_dir = Path(output_dir)
        cache_path = output_dir / "evaluation_results_cache.json"
        
        if cache_path.exists() and not overwrite:
            with open(cache_path) as f:
                cached = json.load(f)
            print(f"✓ Loaded cached evaluation results from {cache_path}\n")
            return cached
    
    # ─────────────────────────────────────────────────────────────────────
    # Compute metrics
    # ─────────────────────────────────────────────────────────────────────
    
    # Setup scoring tools
    scorer_rouge = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    
    results = {
        "timestamp": str(pd.Timestamp.now()),
        "by_style": {},
        "summary": {}
    }
    
    for style, merged in merged_by_style.items():
        if merged.empty:
            print(f"⚠ No matches for style '{style}'")
            continue
        
        print(f"Evaluating [{style.upper()}] ({len(merged)} records)")
        
        style_results = _evaluate_style(
            merged, style, scorer_rouge
        )
        
        results["by_style"][style] = style_results
    
    # ─────────────────────────────────────────────────────────────────────
    # Cache results
    # ─────────────────────────────────────────────────────────────────────
    
    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Cached evaluation results to {cache_path}")
    
    return results


def _evaluate_style(
    merged: pd.DataFrame,
    style: str,
    scorer_rouge
) -> Dict[str, Any]:
    """
    Compute all metrics for a single prompt style.
    
    Internal helper for evaluate_all_styles().
    """
    
    style_results = {
        "style": style,
        "record_count": len(merged),
        "metrics": {}
    }
    
    # ─────────────────────────────────────────────────────────────────────
    # Birthdate evaluation
    # ─────────────────────────────────────────────────────────────────────
    
    birthdate_scores_list = []
    for gt_date, pred_date in zip(merged.get("birthdate", []), merged.get("birthdate", [])):
        scores = birthdate_scores(gt_date, pred_date)
        if not pd.isna(scores["exact"]):
            birthdate_scores_list.append(scores["exact"])
    
    if birthdate_scores_list:
        birthdate_acc = sum(birthdate_scores_list) / len(birthdate_scores_list)
        style_results["metrics"]["birthdate_exact"] = float(birthdate_acc)
    
    # ─────────────────────────────────────────────────────────────────────
    # Gender evaluation
    # ─────────────────────────────────────────────────────────────────────
    
    # Find gender columns (may be suffixed after merge)
    gt_gender_col = next((c for c in ["gender_x", "gender"] if c in merged.columns), None)
    pred_gender_col = next((c for c in ["gender_y", "gender"] if c in merged.columns), None)
    
    if gt_gender_col and pred_gender_col:
        gender_scores_list = [
            gender_match_score(gt, pred)
            for gt, pred in zip(merged[gt_gender_col], merged[pred_gender_col])
        ]
        valid = [s for s in gender_scores_list if not pd.isna(s)]
        if valid:
            gender_acc = sum(valid) / len(valid)
            style_results["metrics"]["gender_exact"] = float(gender_acc)
    
    # ─────────────────────────────────────────────────────────────────────
    # Religion evaluation
    # ─────────────────────────────────────────────────────────────────────
    
    gt_relig_col = next((c for c in ["religious_affiliation_x", "religious_affiliation"] 
                         if c in merged.columns), None)
    pred_relig_col = next((c for c in ["religious_affiliation_y", "religious_affiliation"] 
                           if c in merged.columns), None)
    
    if gt_relig_col and pred_relig_col:
        relig_scores_list = [
            religion_match_score(gt, pred)
            for gt, pred in zip(merged[gt_relig_col], merged[pred_relig_col])
        ]
        valid = [s for s in relig_scores_list if not pd.isna(s)]
        if valid:
            relig_acc = sum(valid) / len(valid)
            style_results["metrics"]["religion_hierarchical"] = float(relig_acc)
    
    # ─────────────────────────────────────────────────────────────────────
    # Education evaluation (ROUGE + component matching)
    # ─────────────────────────────────────────────────────────────────────
    
    if "education" in merged.columns and "education_text" in merged.columns:
        rouge_scores = []
        comp_results = []
        
        for gt_edu, pred_edu in zip(merged["education_text"], merged["education"]):
            if pd.notna(gt_edu) and pd.notna(pred_edu):
                # ROUGE
                rouge_score = scorer_rouge.score(str(gt_edu), str(pred_edu))["rouge1"].fmeasure
                rouge_scores.append(rouge_score)
                
                # Component matching
                gt_parsed = parse_education(gt_edu)
                pred_parsed = parse_education(pred_edu)
                if gt_parsed and pred_parsed:
                    comp = compare_education_components(gt_parsed, pred_parsed)
                    if not pd.isna(comp["combined_score"]):
                        comp_results.append(comp["combined_score"])
        
        if rouge_scores:
            style_results["metrics"]["education_rouge1"] = float(sum(rouge_scores) / len(rouge_scores))
        if comp_results:
            style_results["metrics"]["education_components"] = float(sum(comp_results) / len(comp_results))
    
    return style_results


def print_evaluation_summary(results: Dict[str, Any]) -> None:
    """
    Print formatted summary of evaluation results.
    
    Args:
        results: Dictionary from evaluate_all_styles()
    """
    
    print("\n" + "=" * 70)
    print(" EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    
    for style, style_results in results.get("by_style", {}).items():
        print(f"\n[{style.upper()}] ({style_results['record_count']} records)")
        
        for metric, value in style_results.get("metrics", {}).items():
            if pd.notna(value):
                print(f"  {metric:<30}: {value:.3f}")
    
    print("\n" + "=" * 70 + "\n")
