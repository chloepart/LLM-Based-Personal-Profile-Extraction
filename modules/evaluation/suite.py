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

try:
    import bert_score
    HAS_BERT_SCORE = False  # Disabled due to matplotlib dependency issue
except ImportError:
    HAS_BERT_SCORE = False

from .evaluator import (
    match_by_fuzzy_name,
    gender_match_score,
    religion_match_score,
    name_match_score,  # ← ADD THIS
    evaluate_text_fields,
    evaluate_education_components,
)
from ..utils.parsing import (
    birthdate_scores,
    parse_education,
    parse_committee_roles,
)
from ..config.config_unified import RELIGION_HIERARCHY

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
    
    # Rename prediction columns to match GT column names for consistent _x/_y suffixing
    # This ensures all paired columns get _x (GT) and _y (pred) suffixes after merge
    pred_rename_map = {
        "religious_affiliation": "religion",  # Match GT column name
    }
    df_pred = df_pred.rename(columns=pred_rename_map)
    
    # Merge by style
    merged_by_style = {}
    
    for style in df_pred["prompt_style"].unique():
        df_style = df_pred[df_pred["prompt_style"] == style]
        
        # Try exact merge first
        merged = df_gt.merge(df_style, on="senator_id", how="inner", suffixes=("_x", "_y"))
        
        # Fall back to fuzzy if needed
        if merged.empty and merge_method == "fuzzy":
            merged = match_by_fuzzy_name(df_gt, df_style)
        
        # Rename prediction-only columns to have _y suffix for consistency
        pred_only_cols = [
            "education",
            "extraction_error", 
            "birth_year_inferred",
            "religious_affiliation_inferred",
            "race_ethnicity"
        ]
        rename_suffix = {col: f"{col}_y" for col in pred_only_cols if col in merged.columns}
        merged = merged.rename(columns=rename_suffix)
        
        merged_by_style[style] = merged
    
    return {
        "df_pred": df_pred,
        "df_gt": df_gt,
        "merged_by_style": merged_by_style,
    }


def get_per_row_scores(
    merged_df: pd.DataFrame,
    style: str,
    religion_hierarchy: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Compute per-row match scores for all fields with consistent NaN/0 semantics.
    
    REUSABLE by both aggregate evaluation (evaluate_all_styles sums these)
    and detailed spreadsheet export (notebook comparison file).
    
    NaN/0 Handling (per evaluator.py architecture):
    ────────────────────────────────────────────
    • gender: 0.0 if pred missing (high-confidence extractable field)
              NaN if GT missing
    • religion: NaN if either missing (low-confidence optional field)
    • name, birthdate: Per-field logic (exact match or normalized scoring)
    • education, committee_roles: ROUGE F1; 0.0 if pred missing and GT not absent
    
    Args:
        merged_df: Merged GT + predictions with _x (GT) and _y (pred) suffixes
        style: Prompt style name (for display only)
        religion_hierarchy: Religion matching taxonomy (defaults RELIGION_HIERARCHY)
        
    Returns:
        DataFrame with columns:
        - senator_id
        - [field]_ground_truth, [field]_predicted, [field]_match_score for each field
        - overall_match_score (mean of all match scores)
    """
    if religion_hierarchy is None:
        religion_hierarchy = RELIGION_HIERARCHY
    
    comparison_data = []
    scorer_rouge = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    
    # Detect which columns exist with _x/_y suffixes
    cols = set(merged_df.columns)
    gt_cols = {col.replace('_x', ''): col for col in cols if col.endswith('_x')}
    pred_cols = {col.replace('_y', ''): col for col in cols if col.endswith('_y')}
    
    for idx, row in merged_df.iterrows():
        comparison_row = {
            "senator_id": row.get("senator_id"),
        }
        
        # ─── 1. NAME (exact match after normalization) ───
        if "full_name" in gt_cols and "full_name" in pred_cols:
            gt_val = row.get(gt_cols["full_name"])
            pred_val = row.get(pred_cols["full_name"])
            comparison_row["name_ground_truth"] = gt_val
            comparison_row["name_predicted"] = pred_val
            score = name_match_score(gt_val, pred_val) if (pd.notna(gt_val) and pd.notna(pred_val)) else float('nan')
            comparison_row["name_match_score"] = score
        
        # ─── 2. GENDER (strict: missing pred → 0.0; missing GT → NaN) ───
        if "gender" in gt_cols and "gender" in pred_cols:
            gt_val = row.get(gt_cols["gender"])
            pred_val = row.get(pred_cols["gender"])
            comparison_row["gender_ground_truth"] = gt_val
            comparison_row["gender_predicted"] = pred_val
            score = gender_match_score(gt_val, pred_val)
            comparison_row["gender_match_score"] = score
        
        # ─── 3. BIRTHDATE (year/month/day granularity, NaN if either missing) ───
        if "birthdate" in gt_cols and "birthdate" in pred_cols:
            gt_val = row.get(gt_cols["birthdate"])
            pred_val = row.get(pred_cols["birthdate"])
            comparison_row["birthdate_ground_truth"] = gt_val
            comparison_row["birthdate_predicted"] = pred_val
            if pd.isna(gt_val) or pd.isna(pred_val):
                score = float('nan')
            else:
                scores = birthdate_scores(gt_val, pred_val)
                score = scores.get("exact", float('nan'))
            comparison_row["birthdate_match_score"] = score
        
        # ─── 4. RELIGION (hierarchical: NaN if either missing) ───
        if "religion" in gt_cols and "religion" in pred_cols:
            gt_val = row.get(gt_cols["religion"])
            pred_val = row.get(pred_cols["religion"])
            comparison_row["religion_ground_truth"] = gt_val
            comparison_row["religion_predicted"] = pred_val
            score = religion_match_score(gt_val, pred_val, religion_hierarchy=religion_hierarchy)
            comparison_row["religion_match_score"] = score
        
        # ─── 5. EDUCATION (ROUGE-1 F-measure text similarity) ───
        if "education" in gt_cols and "education" in pred_cols:
            gt_val = row.get(gt_cols["education"])
            pred_val = row.get(pred_cols["education"])
            comparison_row["education_ground_truth"] = gt_val
            comparison_row["education_predicted"] = pred_val
            
            # Apply text field scoring logic:
            # - Both absent or both present: mutual score (ROUGE or 1.0 if absent)
            # - GT absent, pred present: 1.0 (correctly detected absence)
            # - GT present, pred absent: 0.0 (extraction failure)
            # - GT absent, pred absent: 1.0 (correct absence)
            if pd.isna(gt_val) and pd.isna(pred_val):
                score = 1.0
            elif pd.isna(gt_val) or pd.isna(pred_val):
                score = 0.0
            else:
                try:
                    rouge_result = scorer_rouge.score(str(gt_val), str(pred_val))
                    score = rouge_result['rouge1'].fmeasure
                except Exception:
                    score = 0.0
            comparison_row["education_match_score"] = score
        
        # ─── 6. COMMITTEE_ROLES (ROUGE-1 F-measure text similarity) ───
        if "committee_roles" in gt_cols and "committee_roles" in pred_cols:
            gt_val = row.get(gt_cols["committee_roles"])
            pred_val = row.get(pred_cols["committee_roles"])
            comparison_row["committee_roles_ground_truth"] = gt_val
            comparison_row["committee_roles_predicted"] = pred_val
            
            if pd.isna(gt_val) and pd.isna(pred_val):
                score = 1.0
            elif pd.isna(gt_val) or pd.isna(pred_val):
                score = 0.0
            else:
                try:
                    rouge_result = scorer_rouge.score(str(gt_val), str(pred_val))
                    score = rouge_result['rouge1'].fmeasure
                except Exception:
                    score = 0.0
            comparison_row["committee_roles_match_score"] = score
        
        comparison_data.append(comparison_row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Add metadata columns
    score_cols = [col for col in comparison_df.columns if col.endswith("_match_score")]
    if score_cols:
        comparison_df["overall_match_score"] = comparison_df[score_cols].mean(axis=1)
    
    pred_cols_list = [col for col in comparison_df.columns if col.endswith("_predicted")]
    if pred_cols_list:
        comparison_df["attributes_with_predictions"] = comparison_df[pred_cols_list].notna().sum(axis=1)
    
    return comparison_df


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
        "metrics": {},
        "field_counts": {}  # Track n_gt and n_pred separately per field
    }
    
   # ─────────────────────────────────────────────────────────────────────
    # Track GT and Pred counts for all fields
    # ─────────────────────────────────────────────────────────────────────

    # Initialize field_counts for each field
    for field in ["name", "gender", "birthdate", "religion", "education", "committee_roles"]:
        style_results["field_counts"][field] = {"n_gt": 0, "n_pred": 0}
    
    # Compute GT and pred counts
    if "full_name_x" in merged.columns:
        style_results["field_counts"]["name"]["n_gt"] = int(merged["full_name_x"].notna().sum())
    if "full_name_y" in merged.columns:
        style_results["field_counts"]["name"]["n_pred"] = int(merged["full_name_y"].notna().sum())
    
    if "gender_x" in merged.columns:
        style_results["field_counts"]["gender"]["n_gt"] = int(merged["gender_x"].notna().sum())
    if "gender_y" in merged.columns:
        style_results["field_counts"]["gender"]["n_pred"] = int(merged["gender_y"].notna().sum())
    
    if "birthdate_x" in merged.columns:
        style_results["field_counts"]["birthdate"]["n_gt"] = int(merged["birthdate_x"].notna().sum())
    if "birthdate_y" in merged.columns:
        style_results["field_counts"]["birthdate"]["n_pred"] = int(merged["birthdate_y"].notna().sum())
    
    if "religion_x" in merged.columns:
        style_results["field_counts"]["religion"]["n_gt"] = int(merged["religion_x"].notna().sum())
    if "religion_y" in merged.columns:
        style_results["field_counts"]["religion"]["n_pred"] = int(merged["religion_y"].notna().sum())
    
    if "education_x" in merged.columns:
        style_results["field_counts"]["education"]["n_gt"] = int(merged["education_x"].notna().sum())
    if "education_y" in merged.columns:
        style_results["field_counts"]["education"]["n_pred"] = int(merged["education_y"].notna().sum())
    
    if "committee_roles_x" in merged.columns:
        style_results["field_counts"]["committee_roles"]["n_gt"] = int(merged["committee_roles_x"].notna().sum())
    if "committee_roles_y" in merged.columns:
        style_results["field_counts"]["committee_roles"]["n_pred"] = int(merged["committee_roles_y"].notna().sum())
    
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
    
    # Now consistently using religion_x (GT) and religion_y (pred)
    gt_relig_col = "religion_x" if "religion_x" in merged.columns else None
    pred_relig_col = "religion_y" if "religion_y" in merged.columns else None
    
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
    
    # After merge, GT education is "education_x" (if it exists), pred is "education_y"
    gt_edu_col   = next((c for c in ["education_x"] if c in merged.columns), None)
    pred_edu_col = "education_y" if "education_y" in merged.columns else None

    if gt_edu_col and pred_edu_col:
        # Define field specifications for evaluate_text_fields
        education_fields = [
            ("education", gt_edu_col, pred_edu_col, parse_education)
        ]
        
        # Call evaluate_text_fields for batched ROUGE + BERT scoring
        bert_scorer = bert_score if HAS_BERT_SCORE else None
        text_results = evaluate_text_fields(
            merged, education_fields, scorer_rouge, bert_scorer=bert_scorer
        )
        
        # Map text field results to metrics
        if "education" in text_results:
            ed = text_results["education"]
            style_results["metrics"]["education_rouge1"] = float(ed.get("rouge1"))
            if ed.get("bertscore_f1") is not None:
                style_results["metrics"]["education_bert"] = float(ed["bertscore_f1"])
        
        # Call evaluate_education_components for per-component breakdown
        # Pass a view with standardised column names the evaluator expects
        edu_merged = merged.rename(columns={gt_edu_col: "education_text", pred_edu_col: "education"})
        comp_results = evaluate_education_components(edu_merged)
        
        # Surface all component metrics (n_gt is a count, not a score — stored separately)
        for component_key in ["degree_exact", "institution_fuzzy", "year_exact", "combined_score"]:
            value = comp_results.get(component_key)
            if value is not None and not pd.isna(value):
                style_results["metrics"][f"education_component_{component_key}"] = float(value)
        if comp_results.get("n_gt"):
            style_results["education_component_n_gt"] = int(comp_results["n_gt"])
        

    # ─────────────────────────────────────────────────────────────────────
    # Committee roles evaluation (ROUGE + BERT)
    # ─────────────────────────────────────────────────────────────────────

    gt_comm_col = "committee_roles_x" if "committee_roles_x" in merged.columns else None
    pred_comm_col = "committee_roles_y" if "committee_roles_y" in merged.columns else None
    
    if gt_comm_col and pred_comm_col:
        committee_fields = [
            ("committee_roles", gt_comm_col, pred_comm_col, parse_committee_roles)
        ]
        
        bert_scorer = bert_score if HAS_BERT_SCORE else None
        text_results = evaluate_text_fields(
            merged, committee_fields, scorer_rouge, bert_scorer=bert_scorer
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
                    for gt, pred in zip(subset["religion_x"], subset["religion_y"])
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
        field_counts = style_results.get("field_counts", {})
        total = style_results.get("total_records", 0)
        
        print(f"\nPROMPT STYLE: {style.upper():20s} (n={total})\n")
        
        # ────── Accuracy Metrics ──────
        for field in ["name_exact", "gender_exact", "birthdate_exact", "birthdate_year", "birthdate_month", "religion_hierarchical"]:
            if field in metrics:
                val = metrics[field]
                if pd.notna(val):
                    pct = val * 100
                    # Parse field name for display and get counts
                    if field == "name_exact":
                        label = "full_name (exact)"
                        counts = field_counts.get("name", {})
                    elif field == "gender_exact":
                        label = "gender (exact)"
                        counts = field_counts.get("gender", {})
                    elif "birthdate" in field:
                        suffix = field.split("_")[1]
                        label = f"birthdate_{suffix}"
                        counts = field_counts.get("birthdate", {})
                    elif "religion" in field:
                        label = "religion"
                        counts = field_counts.get("religion", {})
                    else:
                        label = field
                        counts = {}
                    
                    n_gt = counts.get("n_gt", 0)
                    n_pred = counts.get("n_pred", 0)
                    print(f"Accuracy   — {label:25s}: {pct:6.2f}%  (n_gt={n_gt}, n_pred={n_pred})")
        
        # ────── Rouge-1 Metrics ──────
        for field in ["education_rouge1", "committee_roles_rouge1"]:
            if field in metrics and pd.notna(metrics[field]):
                if field == "education_rouge1":
                    field_name = "education"
                    label = "education rouge1"
                else:
                    field_name = "committee_roles"
                    label = "committee roles rouge1"
                
                counts = field_counts.get(field_name, {})
                n_gt = counts.get("n_gt", 0)
                n_pred = counts.get("n_pred", 0)
                print(f"Rouge-1    — {label:25s}: {metrics[field]:.3f}  (n_gt={n_gt}, n_pred={n_pred})")
        
        # ────── BERT Score ──────
        bert_field_labels = {
            "education_bert": "education",
            "committee_roles_bert": "committee roles",
        }
        for field, label in bert_field_labels.items():
            if field in metrics and pd.notna(metrics[field]):
                # Extract field name for lookup in counts
                field_name = "education" if "education" in field else "committee_roles"
                counts = field_counts.get(field_name, {})
                n_gt = counts.get("n_gt", 0)
                n_pred = counts.get("n_pred", 0)
                print(f"BERT score — {label:25s}: F1={metrics[field]:.3f}  (n_gt={n_gt}, n_pred={n_pred})")
        
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
    
    # ────── Cross-Style Comparison ──────
    CROSS_STYLE_METRICS = [
        "name_exact",
        "gender_exact",
        "religion_hierarchical",
        "birthdate_exact",
        "birthdate_year",
        "education_rouge1",
        "education_bert",
        "committee_roles_rouge1",
        "committee_roles_bert",
    ]

    PERCENT_METRICS = {
        "name_exact", "gender_exact", "religion_hierarchical",
        "birthdate_exact", "birthdate_year",
        "education_rouge1", "committee_roles_rouge1",
    }

    rows = {}
    for style, style_results in results.get("by_style", {}).items():
        metrics = style_results.get("metrics", {})
        rows[style] = {m: metrics.get(m) for m in CROSS_STYLE_METRICS}

    if rows:
        df_cross = pd.DataFrame(rows).T  # styles as rows, metrics as columns
        df_cross = df_cross.dropna(axis=1, how="all")  # drop metrics with no data

        # Format for display
        df_display = df_cross.copy()
        for col in df_display.columns:
            if col in PERCENT_METRICS:
                df_display[col] = df_display[col].map(
                    lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—"
                )
            else:
                df_display[col] = df_display[col].map(
                    lambda x: f"{x:.3f}" if pd.notna(x) else "—"
                )

        print("\n" + "─" * 90)
        print(" CROSS-STYLE COMPARISON")
        print("─" * 90)
        print(df_display.to_string())

        # Best style per metric
        print("\n── Best Style Per Metric ──")
        for col in df_cross.columns:
            col_vals = df_cross[col].dropna()
            if not col_vals.empty:
                best_style = col_vals.idxmax()
                best_val = col_vals.max()
                fmt = f"{best_val*100:.2f}%" if col in PERCENT_METRICS else f"{best_val:.3f}"
                print(f"  {col:30s}: {best_style} ({fmt})")
    
    print("=" * 90 + "\n")
