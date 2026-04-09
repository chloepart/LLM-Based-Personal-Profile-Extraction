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
    name_match_score,  # ← ADD THIS
    evaluate_text_fields,
    evaluate_education_components,
)
from .parsing import (
    birthdate_scores,
    parse_education,
    parse_committee_roles,
)
from .config_unified import RELIGION_HIERARCHY

import warnings
warnings.filterwarnings("ignore", message="Some weights of RobertaModel were not initialized")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

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
    overwrite: bool = False,
    religion_hierarchy: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Compute accuracy metrics for all prompt styles.
    
    Caches results to JSON for fast notebook reruns.
    
    Args:
        merged_by_style: Dict from load_and_merge_results()
        output_dir: Directory to cache results JSON
        overwrite: Whether to recompute if cache exists
        religion_hierarchy: Dict mapping religions to categories (defaults to RELIGION_HIERARCHY)
        
    Returns:
        Dictionary with results for each style and shared metrics
    """
    
    # Use provided hierarchy or default to config
    if religion_hierarchy is None:
        religion_hierarchy = RELIGION_HIERARCHY
    
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
            merged, style, scorer_rouge, religion_hierarchy
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
    scorer_rouge,
    religion_hierarchy: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Compute all metrics for a single prompt style.
    
    Internal helper for evaluate_all_styles().
    """
    
    if religion_hierarchy is None:
        religion_hierarchy = RELIGION_HIERARCHY
    
    # ─────────────────────────────────────────────────────────────────────
    # Initialize results dict
    # ─────────────────────────────────────────────────────────────────────
    
    style_results = {
        "style": style,
        "record_count": len(merged),
        "metrics": {}
    }
    
   # ─────────────────────────────────────────────────────────────────────
    # Track NaN counts for all fields
    # ─────────────────────────────────────────────────────────────────────

    nan_counts = {
        "name": int(merged[["full_name_x", "full_name_y"]].isna().any(axis=1).sum()) if "full_name_x" in merged.columns else int(merged["full_name"].isna().sum()),
        "gender": int(merged["gender_x"].isna().sum()) if "gender_x" in merged.columns else int(merged["gender"].isna().sum()),
        "birthdate": int(merged["birthdate_x"].isna().sum()) if "birthdate_x" in merged.columns else int(merged["birthdate"].isna().sum()),
        "religion": int(merged["religious_affiliation_x"].isna().sum()) if "religious_affiliation_x" in merged.columns else (int(merged["religion"].isna().sum()) if "religion" in merged.columns else 0),
        "education": int(merged["education_x"].isna().sum()) if "education_x" in merged.columns else int(merged["education"].isna().sum()),
        "committee_roles": int(merged["committee_roles_x"].isna().sum()) if "committee_roles_x" in merged.columns else int(merged["committee_roles"].isna().sum()),
    }
    style_results["nan_counts"] = nan_counts
    style_results["total_records"] = len(merged)
    
    # ─────────────────────────────────────────────────────────────────────
    # Birthdate evaluation (exact, year, month granularity)
    # ─────────────────────────────────────────────────────────────────────

    birthdate_exact_list = []
    birthdate_year_list = []
    birthdate_month_list = []

    for gt_date, pred_date in zip(merged.get("birthdate_x", []), merged.get("birthdate_y", [])):
        scores = birthdate_scores(gt_date, pred_date)
        if not pd.isna(scores["exact"]):
            birthdate_exact_list.append(scores["exact"])
            birthdate_year_list.append(scores["year"])
            birthdate_month_list.append(scores["month"])

    if birthdate_exact_list:
        style_results["metrics"]["birthdate_exact"] = float(sum(birthdate_exact_list) / len(birthdate_exact_list))
        style_results["metrics"]["birthdate_year"] = float(sum(birthdate_year_list) / len(birthdate_year_list))
        style_results["metrics"]["birthdate_month"] = float(sum(birthdate_month_list) / len(birthdate_month_list)) 
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
    # Name evaluation (EXACT MATCH)
    # ─────────────────────────────────────────────────────────────────────

    gt_name_col = next((c for c in ["full_name_x", "full_name"] if c in merged.columns), None)
    pred_name_col = next((c for c in ["full_name_y", "full_name"] if c in merged.columns), None)

    if gt_name_col and pred_name_col:
        name_scores_list = [
            name_match_score(gt, pred)
            for gt, pred in zip(merged[gt_name_col], merged[pred_name_col])
        ]
        valid = [s for s in name_scores_list if not pd.isna(s)]
        if valid:
            name_acc = sum(valid) / len(valid)
            style_results["metrics"]["name_exact"] = float(name_acc)
    

    # ─────────────────────────────────────────────────────────────────────
    # Religion evaluation
    # ─────────────────────────────────────────────────────────────────────
    
    gt_relig_col = next((c for c in ["religious_affiliation_x", "religious_affiliation"] 
                         if c in merged.columns), None)
    pred_relig_col = next((c for c in ["religious_affiliation_y", "religious_affiliation"] 
                           if c in merged.columns), None)
    
    if gt_relig_col and pred_relig_col:
        relig_scores_list = [
            religion_match_score(gt, pred, religion_hierarchy=religion_hierarchy)
            for gt, pred in zip(merged[gt_relig_col], merged[pred_relig_col])
        ]
        valid = [s for s in relig_scores_list if not pd.isna(s)]
        if valid:
            relig_acc = sum(valid) / len(valid)
            style_results["metrics"]["religion_hierarchical"] = float(relig_acc)
    
    # ─────────────────────────────────────────────────────────────────────
    # Education evaluation (ROUGE + BERT + component breakdown)
    # ─────────────────────────────────────────────────────────────────────
    
    if "education" in merged.columns and "education_text" in merged.columns:
        # Define field specifications for evaluate_text_fields
        education_fields = [
            ("education", "education_text", "education", parse_education)
        ]
        
        # Call evaluate_text_fields for batched ROUGE + BERT scoring
        text_results = evaluate_text_fields(
            merged, education_fields, scorer_rouge, bert_scorer=bert_score
        )
        
        # Map text field results to metrics
        if "education" in text_results:
            ed = text_results["education"]
            style_results["metrics"]["education_rouge1"] = float(ed.get("rouge1"))
            if ed.get("bertscore_f1") is not None:
                style_results["metrics"]["education_bert"] = float(ed["bertscore_f1"])
        
        # Call evaluate_education_components for per-component breakdown
        comp_results = evaluate_education_components(merged)
        
        # Surface all component metrics
        for component_key in ["degree_exact", "institution_fuzzy", "year_exact", "combined_score"]:
            value = comp_results.get(component_key)
            if not pd.isna(value):
                style_results["metrics"][f"education_component_{component_key}"] = float(value)
        

    # ─────────────────────────────────────────────────────────────────────
    # Committee roles evaluation (ROUGE + BERT)
    # ─────────────────────────────────────────────────────────────────────

    if "committee_roles" in merged.columns:
        committee_fields = [
            ("committee_roles", "committee_roles", "committee_roles", parse_committee_roles)
        ]
        
        text_results = evaluate_text_fields(
            merged, committee_fields, scorer_rouge, bert_scorer=bert_score
        )
        
        if "committee_roles" in text_results:
            cr = text_results["committee_roles"]
            style_results["metrics"]["committee_roles_rouge1"] = float(cr.get("rouge1"))
            if cr.get("bertscore_f1") is not None:
                style_results["metrics"]["committee_roles_bert"] = float(cr["bertscore_f1"])

    # ─────────────────────────────────────────────────────────────────────
    # Religion stratification by signal type (explicit vs not_explicit)
    # ─────────────────────────────────────────────────────────────────────

    if "religious_signal_x" in merged.columns:  # Assumes merged has signal column
        religion_by_signal = {}
        for signal_type in ["explicit", "not_explicit"]:
            subset = merged[merged["religious_signal_x"] == signal_type]
            if len(subset) > 0:
                scores = [
                    religion_match_score(gt, pred, religion_hierarchy)
                    for gt, pred in zip(subset["religious_affiliation_x"], subset["religious_affiliation_y"])
                ]
                valid = [s for s in scores if not pd.isna(s)]
                if valid:
                    religion_by_signal[signal_type] = {
                        "accuracy": sum(valid) / len(valid),
                        "n": len(valid)
                    }
    
        if religion_by_signal:
            style_results["religion_by_signal"] = religion_by_signal

    return style_results


def print_evaluation_summary(results: Dict[str, Any]) -> None:
    """Print formatted structured evaluation summary with sample counts and stratification."""
    
    print("\n" + "=" * 90)
    print(" EVALUATION RESULTS BY PROMPT STYLE")
    print("=" * 90)
    
    for style, style_results in results.get("by_style", {}).items():
        metrics = style_results.get("metrics", {})
        nans = style_results.get("nan_counts", {})
        total = style_results.get("total_records", 0)
        
        print(f"\nPROMPT STYLE: {style.upper():20s} (n={total})\n")
        
        # ────── Accuracy Metrics ──────
        for field in ["name_exact", "gender_exact", "birthdate_exact", "birthdate_year", "birthdate_month", "religion_hierarchical"]:
            if field in metrics:
                val = metrics[field]
                if pd.notna(val):
                    pct = val * 100
                    # Parse field name for display
                    if field == "name_exact":
                        label = "full_name (exact)"
                        skipped = nans.get("name", 0)
                        n_scored = total - skipped
                    elif field == "gender_exact":
                        label, skipped = "gender (exact)", nans.get("gender", 0)
                        n_scored = total - skipped
                    elif "birthdate" in field:
                        suffix = field.split("_")[1]
                        label = f"birthdate_{suffix}"
                        skipped = nans.get("birthdate", 0)
                        n_scored = total - skipped
                    elif "religion" in field:
                        label = "religion"
                        skipped = nans.get("religion", 0)
                        n_scored = total - skipped
                    else:
                        label = field
                        n_scored = total
                        skipped = 0
                    
                    print(f"Accuracy   — {label:25s}: {pct:6.2f}%  (n={n_scored}, skipped {skipped} missing GT)")
        
        # ────── Rouge-1 Metrics ──────
        for field in ["education_rouge1", "committee_roles_rouge1"]:
            if field in metrics and pd.notna(metrics[field]):
                label = field.replace("_", " ")
                print(f"Rouge-1    — {label:25s}: {metrics[field]:.3f}")
        
        # ────── BERT Score ──────
        for field in ["education_bert"]:
            if field in metrics and pd.notna(metrics[field]):
                label = "education"
                print(f"BERT score — {label:25s}: F1={metrics[field]:.3f}")
        
        # ────── Education Components ──────
        edu_components = {k: v for k, v in metrics.items() if "education_component" in k}
        if edu_components:
            print(f"\n── Education Components (Detailed) ──")
            for key, val in sorted(edu_components.items()):
                if pd.notna(val):
                    label = key.replace("education_component_", "").replace("_", " ")
                    print(f"  {label:20s}: {val*100:6.2f}%")
        
        # ────── Religion Stratification (if available) ──────
        if "religion_by_signal" in style_results:
            print(f"\n── Religion Accuracy by Signal Type ──")
            for signal, score in style_results["religion_by_signal"].items():
                pct = score["accuracy"] * 100
                n = score["n"]
                print(f"  {signal:20s}: {pct:6.2f}%  (n={n})")
        
        print()
    
    print("=" * 90 + "\n")
