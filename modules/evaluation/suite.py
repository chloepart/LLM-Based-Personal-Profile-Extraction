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
    HAS_BERT_SCORE = True  # Disabled due to matplotlib dependency issue
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
    parse_education_detailed,          
    compare_education_components,
)
from ..config.config_unified import RELIGION_HIERARCHY

import warnings
warnings.filterwarnings("ignore", message="Some weights of RobertaModel were not initialized")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# ============================================================================
# BASELINE COMPARISON NOTES (Liu et al. vs. Senator Adaptation)
# ============================================================================

"""
Baseline Implementation Guide
==============================

This suite compares 5 traditional PIE baselines against LLM-based extraction.
All baselines are adapted from Liu et al. (USENIX Security 2025) with
senator-specific modifications documented below.

BASELINE SCHEMA:
  Liu et al. evaluates 8 PIE categories:
    1. email address     ✅ Aligned
    2. phone number      ✅ Aligned
    3. mailing address   (→ renamed "address")
    4. name              ✅ Aligned
    5. work experience   (→ extracted as "work" text field)
    6. education         ✅ Aligned
    7. affiliation       ✅ Aligned
    8. occupation        ✅ Aligned

  + Senator-specific expansions:
    9. state             (domain adaptation)
    10. party            (domain adaptation)
    11. religion         (domain expansion)
    12. years_found      (auxiliary: timeline extraction)

BASELINE DEVIATIONS FROM LIU ET AL.:

1. REGEX (Baseline 1)
   ├─ Aligned: Email, phone, name (via Liu et al. Table 15 patterns)
   ├─ Extended: Added party extraction regex
   ├─ Excluded: Work/education/affiliation regex (Liu et al. uses HTML <p> tags)
   └─ Reason: Senator bios have inconsistent <p> tag usage; prefer NER

2. KEYWORD SEARCH (Baseline 2)
   ├─ Aligned: Liu et al. strategy (search HTML headers + plaintext labels)
   ├─ Enhanced: Added BeautifulSoup sibling traversal (paper less explicit)
   ├─ Extended: Keyword map includes senator-specific triggers
   │           (e.g., "caucus" for affiliation, "senator" for occupation)
   └─ Reason: Improves recall on unstructured senator bios

3. spaCy NER (Baseline 3) ⭐ FIXED
   ├─ Original problem: Returned raw NER types (PERSON, ORG, GPE, DATE)
   ├─ Fix: Added NER→PIE semantic mapping layer
   ├─ Mapping logic:
   │   ├─ PERSON → name (first occurrence)
   │   ├─ ORG → affiliation (first) or occupation (if contains "Senate")
   │   ├─ GPE → state
   │   └─ DATE → years_found (regex extraction)
   └─ Impact: Now produces comparable output to BERT/TextWash

4. BERT NER (Baseline 4)
   ├─ Aligned: Uses dslim/bert-base-NER (CoNLL-2003 fine-tuned)
   ├─ Note: Liu et al. uses TextWash; dslim is closest open alternative
   ├─ Same NER→PIE mapping as spaCy (PERSON→name, ORG→affiliation, etc.)
   └─ Known limitation: 512-token limit may truncate long bios

5. TextWash NER (Baseline 5) ⭐ NEW
   ├─ Source: Kleinberg et al. (2022), open-source, GPL-3.0
   ├─ Model: Learned PII detection (not just dict-based)
   ├─ Entity types: EMAIL_ADDRESS, PHONE_NUMBER, ADDRESS, PERSON_FIRSTNAME,
   │               PERSON_LASTNAME, ORGANIZATION, OCCUPATION, DATE
   ├─ Advantage: More directly comparable to Liu et al. Appendix results
   ├─ Setup: Requires manual GDrive download + model placement at ./data/en/
   └─ Trade-off: High setup cost but potentially best accuracy

EVALUATION METRICS:
  • accuracy (exact match for email, phone)
  • ROUGE-1 (word overlap for text fields)
  • BERT-score (semantic similarity, requires transformers)

FIELD-LEVEL NOTES:
  • name: Compare against ground truth after lower() normalization
  • education: Parse into degree+institution; compare components
  • address: Loose comparison (ROUGE-1) due to format variation
  • work: Often None in baselines; compare text overlap with LLM extracts
  • party: Not in Liu et al.; senator-specific evaluation only
"""

# ============================================================================


def evaluate_baseline_extraction(
    senator_bio: str,
    ground_truth: dict,
    baseline_names: list = None
) -> dict:
    """
    Quick evaluation of baseline extractors on a single senator bio.
    
    Useful for testing baseline quality before running full suite evaluation.
    
    Args:
        senator_bio: Raw HTML or plaintext bio
        ground_truth: Dict with keys matching PIE schema
                      (name, email, phone, education, affiliation, etc.)
        baseline_names: List of baselines to run. Options:
                        ["regex", "keyword", "spacy", "bert", "textwash"]
                        Defaults to all 5.
    
    Returns:
        Dict mapping baseline_name → extracted_fields
        
    Example:
        >>> from modules.evaluation.baselines import *
        >>> results = evaluate_baseline_extraction(
        ...     senator_bio=bio_text,
        ...     ground_truth={"name": "John Smith", "email": "john@senate.gov"},
        ...     baseline_names=["keyword", "spacy"]
        ... )
        >>> results["spacy"]
        {'name': 'John Smith', 'email': None, 'affiliation': 'U.S. Senate', ...}
    """
    from modules.evaluation.baselines import (
        RegexBaseline, KeywordSearchBaseline, SpaCyBaseline,
        BERTBaseline, TextWashBaseline
    )
    
    baseline_names = baseline_names or ["regex", "keyword", "spacy", "bert"]
    results = {}
    
    # You'll need to pass compiled regex patterns; construct separately
    baselines = {
        "keyword": KeywordSearchBaseline(),
        "spacy": SpaCyBaseline(),
        "bert": BERTBaseline(),
        # "textwash": TextWashBaseline(),  # Only if installed
    }
    
    for name in baseline_names:
        if name not in baselines:
            print(f"⚠️  Baseline '{name}' not available or not initialized")
            continue
        
        try:
            extracted = baselines[name].extract(senator_bio)
            results[name] = extracted
        except Exception as e:
            print(f"❌ {name} extraction failed: {e}")
            results[name] = None
    
    return results

# ============================================================================
# BASELINE EVALUATION & COMPARISON
# ============================================================================

def load_baseline_results(baseline_csv_path):
    """
    Load and return baseline results DataFrame.
    
    Args:
        baseline_csv_path: Path to baselines.csv
        
    Returns:
        DataFrame with baseline results
    """
    return pd.read_csv(baseline_csv_path)


def summarize_baseline_coverage(baseline_df, ground_truth_df=None):
    """
    Summarize extraction coverage per baseline method.
    
    Args:
        baseline_df: DataFrame from baselines.csv
        ground_truth_df: Optional ground truth for comparison
        
    Returns:
        Dict with coverage statistics by baseline
    """
    summary = {}
    
    # Regex coverage
    summary["regex"] = {
        "names_found": (baseline_df["regex_name"].notna()).sum(),
        "emails_found": (baseline_df["regex_email_found"] > 0).sum(),
        "phones_found": (baseline_df["regex_phone_found"] > 0).sum(),
    }
    
    # Keyword coverage
    summary["keyword"] = {
        "names_found": (baseline_df["keyword_name"].notna()).sum(),
        "emails_found": (baseline_df["keyword_email"].notna()).sum(),
        "phones_found": (baseline_df["keyword_phone"].notna()).sum(),
        "education_found": (baseline_df["keyword_education"].notna()).sum(),
    }
    
    # spaCy coverage
    summary["spacy"] = {
        "names_found": (baseline_df["spacy_name"].notna()).sum(),
        "states_found": (baseline_df["spacy_state"].notna()).sum(),
        "affiliations_found": (baseline_df["spacy_affiliation"].notna()).sum(),
        "avg_years_per_senator": baseline_df.get("spacy_years_count", pd.Series([0]*len(baseline_df))).mean(),
        "avg_education_entries": baseline_df.get("spacy_education_entries", pd.Series([0]*len(baseline_df))).mean(),
        "avg_committee_roles": baseline_df.get("spacy_committee_roles_count", pd.Series([0]*len(baseline_df))).mean(),
    }
    
    # BERT coverage
    summary["bert"] = {
        "avg_persons_found": baseline_df["bert_persons_found"].mean(),
        "avg_orgs_found": baseline_df["bert_orgs_found"].mean(),
    }
    
    return summary


def print_baseline_summary(baseline_df, total_senators):
    """
    Print formatted baseline comparison table.
    
    CORRECTED: Removed email/phone reporting since they have no ground truth for evaluation.
    Only shows fields that can be scored against ground truth:
    - NAME (all baselines)
    - EDUCATION (Keyword, spaCy)
    - STATES, AFFILIATIONS (spaCy)
    
    Args:
        baseline_df: DataFrame with baseline results
        total_senators: Total number of senators in dataset
    """
    summary = summarize_baseline_coverage(baseline_df)
    
    print("\n" + "=" * 80)
    print(" BASELINE EXTRACTION COVERAGE (GROUND-TRUTH-SCORABLE FIELDS ONLY)")
    print("=" * 80)
    print(f"\nTotal senators: {total_senators}")
    print("Note: Email & phone excluded (no ground truth available for evaluation)\n")
    
    print("REGEX BASELINE (Scorable fields):")
    print(f"  Names extracted:              {summary['regex']['names_found']:3d}/{total_senators} ({100*summary['regex']['names_found']/total_senators:5.1f}%)")
    # Email/phone removed — not scorable
    
    print("\nKEYWORD SEARCH BASELINE (Scorable fields):")
    print(f"  Names extracted:              {summary['keyword']['names_found']:3d}/{total_senators} ({100*summary['keyword']['names_found']/total_senators:5.1f}%)")
    print(f"  Education found:              {summary['keyword']['education_found']:3d}/{total_senators} ({100*summary['keyword']['education_found']/total_senators:5.1f}%)")
    # Email/phone removed — not scorable
    
    print("\nspaCy NER BASELINE (Scorable fields):")
    print(f"  Names extracted:              {summary['spacy']['names_found']:3d}/{total_senators} ({100*summary['spacy']['names_found']/total_senators:5.1f}%)")
    print(f"  States found:                 {summary['spacy']['states_found']:3d}/{total_senators} ({100*summary['spacy']['states_found']/total_senators:5.1f}%)")
    print(f"  Affiliations:                 {summary['spacy']['affiliations_found']:3d}/{total_senators} ({100*summary['spacy']['affiliations_found']/total_senators:5.1f}%)")
    print(f"  Avg education entries/senator: {summary['spacy']['avg_education_entries']:6.2f}")
    
    print("\nBERT NER BASELINE:")
    print(f"  ⚠ Entity counts only (no GT labels for persons/orgs)")
    print(f"  Avg persons/senator: {summary['bert']['avg_persons_found']:6.2f}")
    print(f"  Avg orgs/senator:    {summary['bert']['avg_orgs_found']:6.2f}")
    
    print("\n" + "=" * 80)
    print("Legend: Only fields with ground truth are scored for accuracy in baseline_accuracy.csv")
    print("=" * 80)# ============================================================================

# BASELINE ACCURACY EVALUATION
# ============================================================================

def get_baseline_accuracy(
    baseline_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    output_dir: Optional[Path] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute accuracy metrics for each baseline method against ground truth.
    
    CORRECTED: Only scores fields with GT. Removed email/phone/BERT counts.
    """
    if verbose:
        print("\n" + "=" * 80)
        print(" COMPUTING BASELINE ACCURACY METRICS (GT-SCORABLE FIELDS ONLY)")
        print("=" * 80)
    
    # Merge baseline results with ground truth
    merged = baseline_df.merge(ground_truth_df, on='senator_id', how='inner', suffixes=('_baseline', '_gt'))
    
    if verbose:
        print(f"\n✓ Merged {len(merged)} senators with ground truth\n")
    
    accuracy_rows = []
    scorer_rouge = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    
    # ──────────────────────────────────────────────────────────────────
    # BASELINE 1: REGEX — Score NAME only
    # ──────────────────────────────────────────────────────────────────
    
    for idx, row in merged.iterrows():
        senator_id = row.get('senator_id')
        regex_name = row.get('regex_name')
        gt_name = row.get('full_name')
        
        if pd.notna(gt_name):
            accuracy = 1.0 if (pd.notna(regex_name) and 
                              regex_name.lower().strip() == str(gt_name).lower().strip()) else 0.0
            
            accuracy_rows.append({
                'senator_id': senator_id,
                'baseline_method': 'Regex',
                'field': 'name',
                'accuracy': accuracy,
                'value_gt': gt_name,
                'value_extracted': regex_name if pd.notna(regex_name) else 'Not extracted',
            })
    
    # ──────────────────────────────────────────────────────────────────
    # BASELINE 2: KEYWORD — Score NAME and EDUCATION
    # ──────────────────────────────────────────────────────────────────
    
    for idx, row in merged.iterrows():
        senator_id = row.get('senator_id')
        gt_name = row.get('full_name')
        
        # Keyword: Name
        keyword_name = row.get('keyword_name')
        if pd.notna(gt_name):
            accuracy = 1.0 if (pd.notna(keyword_name) and 
                              keyword_name.lower().strip() == str(gt_name).lower().strip()) else 0.0
            accuracy_rows.append({
                'senator_id': senator_id,
                'baseline_method': 'Keyword',
                'field': 'name',
                'accuracy': accuracy,
                'value_gt': gt_name,
                'value_extracted': keyword_name if pd.notna(keyword_name) else 'Not extracted',
            })
        
        # Keyword: Education
        keyword_education = row.get('keyword_education')
        gt_education = row.get('education')
        
        if pd.notna(gt_education):
            if pd.notna(keyword_education):
                try:
                    rouge_score = scorer_rouge.score(
                        str(keyword_education).lower(),
                        str(gt_education).lower()
                    )
                    accuracy = rouge_score['rouge1'].fmeasure
                except:
                    accuracy = 0.0
            else:
                accuracy = 0.0
            
            accuracy_rows.append({
                'senator_id': senator_id,
                'baseline_method': 'Keyword',
                'field': 'education',
                'accuracy': accuracy,
                'value_gt': str(gt_education)[:100],
                'value_extracted': str(keyword_education)[:100] if pd.notna(keyword_education) else 'Not extracted',
            })
    
    # ──────────────────────────────────────────────────────────────────
    # BASELINE 3: spaCy NER — Score NAME and EDUCATION
    # ──────────────────────────────────────────────────────────────────
    
    for idx, row in merged.iterrows():
        senator_id = row.get('senator_id')
        gt_name = row.get('full_name')
        
        # spaCy: Name
        spacy_name = row.get('spacy_name')
        if pd.notna(gt_name):
            accuracy = 1.0 if (pd.notna(spacy_name) and 
                              spacy_name.lower().strip() == str(gt_name).lower().strip()) else 0.0
            accuracy_rows.append({
                'senator_id': senator_id,
                'baseline_method': 'spaCy',
                'field': 'name',
                'accuracy': accuracy,
                'value_gt': gt_name,
                'value_extracted': spacy_name if pd.notna(spacy_name) else 'Not extracted',
            })
        
        # spaCy: Education (count-based)
        spacy_edu_count = row.get('spacy_education_entries', 0)
        gt_education = row.get('education')
        
        if pd.notna(gt_education):
            gt_edu_count = 0
            try:
                edu_list = json.loads(str(gt_education)) if isinstance(gt_education, str) else gt_education
                gt_edu_count = len(edu_list) if isinstance(edu_list, list) else 1
            except:
                gt_edu_count = 1 if gt_education else 0
            
            if spacy_edu_count == gt_edu_count:
                accuracy = 1.0
            elif abs(spacy_edu_count - gt_edu_count) == 1:
                accuracy = 0.5
            else:
                accuracy = 0.0 if spacy_edu_count > 0 else (1.0 if gt_edu_count == 0 else 0.0)
            
            accuracy_rows.append({
                'senator_id': senator_id,
                'baseline_method': 'spaCy',
                'field': 'education',
                'accuracy': accuracy,
                'value_gt': f'{gt_edu_count} entries',
                'value_extracted': f'{spacy_edu_count} entries',
            })
    
    # ──────────────────────────────────────────────────────────────────
    # AGGREGATE AND SAVE
    # ──────────────────────────────────────────────────────────────────
    
    df_accuracy = pd.DataFrame(accuracy_rows)  # ← THIS WAS MISSING!
    
    if len(df_accuracy) == 0:
        if verbose:
            print("⚠ No accuracy rows generated. Check data alignment.\n")
        return df_accuracy
    
    summary = df_accuracy.groupby(['baseline_method', 'field'])['accuracy'].agg(['mean', 'std', 'count', 'min', 'max'])
    
    if verbose:
        print("ACCURACY SUMMARY BY BASELINE METHOD & FIELD:")
        print("─" * 80)
        print(summary.to_string())
        print()
        
        baseline_summary = df_accuracy.groupby('baseline_method')['accuracy'].agg(['mean', 'count'])
        print("\nOVERALL BY BASELINE (all scorable fields):")
        print("─" * 40)
        print(baseline_summary.to_string())
        print()
    
    if output_dir:
        output_path = Path(output_dir) / 'baseline_accuracy.csv'
        df_accuracy.to_csv(output_path, index=False)
        if verbose:
            print(f"✓ Saved {len(df_accuracy)} baseline accuracy rows to: {output_path}\n")
    
    return df_accuracy

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
        
        # ─── 5. EDUCATION (ROUGE-1 F-measure + component-level breakdown) ───
        if "education" in gt_cols and "education" in pred_cols:
            gt_val = row.get(gt_cols["education"])
            pred_val = row.get(pred_cols["education"])
            comparison_row["education_ground_truth"] = gt_val
            comparison_row["education_predicted"] = pred_val
            
            # Overall ROUGE score
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
            
            # Component-level breakdown
            try:
                # Parse GT education (JSON string → list of dicts)
                gt_items = []
                if pd.notna(gt_val):
                    if isinstance(gt_val, str):
                        try:
                            parsed = json.loads(gt_val)
                            gt_items = parsed if isinstance(parsed, list) else []
                        except (json.JSONDecodeError, TypeError):
                            gt_items = []
                    elif isinstance(gt_val, list):
                        gt_items = gt_val
                
                # Parse pred education (could be JSON string OR Python repr string)
                pred_items = []
                if pd.notna(pred_val):
                    if isinstance(pred_val, str):
                        # Try JSON first
                        try:
                            parsed = json.loads(pred_val)
                            pred_items = parsed if isinstance(parsed, list) else []
                        except (json.JSONDecodeError, TypeError):
                            # If JSON fails, try ast.literal_eval for Python repr strings
                            try:
                                import ast
                                parsed = ast.literal_eval(pred_val)
                                pred_items = parsed if isinstance(parsed, list) else []
                            except (ValueError, SyntaxError):
                                pred_items = []
                    elif isinstance(pred_val, list):
                        pred_items = pred_val
                
                # Score components
                comp_scores = compare_education_components(gt_items, pred_items)
                
                comparison_row["education_component_degree_match_score"] = comp_scores.get("degree_exact", float('nan'))
                comparison_row["education_component_institution_match_score"] = comp_scores.get("institution_fuzzy", float('nan'))
                comparison_row["education_component_year_match_score"] = comp_scores.get("year_exact", float('nan'))
                comparison_row["education_component_combined_score"] = comp_scores.get("combined_score", float('nan'))
                comparison_row["education_component_n_gt"] = comp_scores.get("n_gt", 0)
            except Exception as e:
                # Silent fallback if component scoring fails
                comparison_row["education_component_degree_match_score"] = float('nan')
                comparison_row["education_component_institution_match_score"] = float('nan')
                comparison_row["education_component_year_match_score"] = float('nan')
                comparison_row["education_component_combined_score"] = float('nan')
                comparison_row["education_component_n_gt"] = 0                        
        
        
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
        # Parse GT education from JSON strings BEFORE component evaluation
        def safe_parse_education_json(val):
            """Convert JSON string to normalized format for parse_education_detailed."""
            if pd.isna(val) or val == "":
                return []
            if isinstance(val, str):
                try:
                    parsed = json.loads(val)
                    # Return the parsed Python list/dict directly, not re-serialized
                    if isinstance(parsed, list):
                        return parsed  # Return actual list, not JSON string
                    return []
                except (json.JSONDecodeError, TypeError):
                    return []
            # Already a list or dict
            return val if isinstance(val, list) else []

        edu_merged = merged.rename(columns={gt_edu_col: "education_text", pred_edu_col: "education"})

        # ← KEY: Parse GT education_text from JSON strings
        edu_merged = edu_merged.copy()
        edu_merged["education_text"] = edu_merged["education_text"].apply(safe_parse_education_json)

        # Pass bert_scorer to enable component-level BERT scoring
        comp_results = evaluate_education_components(
            edu_merged,
            bert_scorer=bert_score if HAS_BERT_SCORE else None
        )
       # Surface all component metrics (exact/fuzzy matching)
        for component_key in ["degree_exact", "institution_fuzzy", "year_exact", "combined_score"]:
            value = comp_results.get(component_key)
            if value is not None and not pd.isna(value):
                style_results["metrics"][f"education_component_{component_key}"] = float(value)

        # Surface component-level BERT scores (if available)
        for bert_key in ["degree_bert", "institution_bert", "year_bert", "combined_component_bert"]:
            if bert_key in comp_results and not pd.isna(comp_results[bert_key]):
                style_results["metrics"][f"education_component_{bert_key}"] = float(comp_results[bert_key])
            # Also surface the count of items evaluated
            count_key = bert_key.replace("_bert", "") + "_n"
            if count_key in comp_results:
                style_results[f"education_component_{count_key}"] = int(comp_results[count_key])
        
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
        "education_component_degree_bert",
        "education_component_institution_bert",  
        "education_component_year_bert",
        "education_component_combined_component_bert",
    ]

    PERCENT_METRICS = {
        "name_exact", "gender_exact", "religion_hierarchical",
        "birthdate_exact", "birthdate_year",
        "education_rouge1", "committee_roles_rouge1",
        "education_component_degree_exact",
        "education_component_institution_fuzzy",
        "education_component_year_exact",
        "education_component_combined_score",
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
