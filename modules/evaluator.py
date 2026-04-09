"""
Evaluation metrics and scoring functions for LLM extraction assessment.

Implements scoring functions following Liu et al. (Section 6.1.4) with
NaN-aware evaluation, partial credit systems, and hierarchical matching.
"""

import unicodedata
import re
from difflib import SequenceMatcher
import json
import pandas as pd


# Name Matching & Normalization

def normalize_name(name):
    """
    Lowercase, strip accents, expand common nickname abbreviations.
    
    Args:
        name: Name string to normalize
    
    Returns:
        Normalized name string
    """
    if not name or (isinstance(name, float) and pd.isna(name)):
        return ""
    
    name = str(name).lower().strip()
    
    # Strip accents using Unicode normalization
    name = "".join(
        c for c in unicodedata.normalize("NFD", name)
        if unicodedata.category(c) != "Mn"
    )
    
    # Expand common nickname abbreviations
    expansions = {
        r"\bdan\b": "daniel",
        r"\btom\b": "thomas",
        r"\bjon\b": "jonathan",
        r"\bjim\b": "james",
        r"\bbob\b": "robert",
        r"\bwill\b": "william",
        r"\bliz\b": "elizabeth",
        r"\bpat\b": "patrick",
        r"\bbert\b": "albert",
        r"\bted\b": "edward",
        r"\bkatie\b": "katherine",
        r"\bcat\b": "catherine",
        r"\btimothy\b": "tim",
        r"\bchristopher\b": "chris",
        r"\banthony\b": "tony",
    }
    
    for pattern, replacement in expansions.items():
        name = re.sub(pattern, replacement, name)
    
    return name


def name_match_score(gt_name, pred_name):
    """
    Calculate name matching score: 1.0 if identical or >90% similar after
    normalization, else 0.0.
    
    Args:
        gt_name: Ground truth name
        pred_name: Predicted name
    
    Returns:
        Float: 1.0 for match, 0.0 for no match
    """
    gt_norm = normalize_name(gt_name)
    pred_norm = normalize_name(pred_name)
    
    if not gt_norm or not pred_norm:
        return 1.0 if gt_norm == pred_norm else 0.0
    if gt_norm == pred_norm:
        return 1.0
    
    return 1.0 if SequenceMatcher(None, gt_norm, pred_norm).ratio() > 0.90 else 0.0


def create_normalized_senator_id(name, state):
    """Create normalized senator ID from name and state."""
    normalized = normalize_name(name)
    name_slug = "_".join(w.capitalize() for w in normalized.split() if w)
    return f"{name_slug}_{state.upper()}"


# Fuzzy Matching

def match_by_fuzzy_name(df_gt, df_pred):
    """
    Match GT ↔ predictions by fuzzy name when senator_id direct merge fails.
    
    Args:
        df_gt: Ground truth DataFrame
        df_pred: Predictions DataFrame
    
    Returns:
        Merged DataFrame with matched rows
    """
    from fuzzywuzzy import fuzz as fuzzy_fuzz, process as fuzzy_process
    
    results = []
    for _, gt_row in df_gt.iterrows():
        gt_id = gt_row.get("senator_id", "")
        gt_name = gt_row.get("full_name") or gt_row.get("name", "")
        
        # Try exact match first
        exact = df_pred[df_pred["senator_id"] == gt_id]
        if not exact.empty:
            for _, pred_row in exact.iterrows():
                results.append({**gt_row, **pred_row})
            continue
        
        # Fall back to fuzzy matching
        gt_norm = normalize_name(gt_name)
        pred_names = df_pred["full_name"].fillna("").astype(str).unique()
        matches = fuzzy_process.extract(
            gt_norm, pred_names, scorer=fuzzy_fuzz.token_sort_ratio, limit=1
        )
        
        if matches and matches[0][1] > 85:
            matched_name = matches[0][0]
            pred_matches = df_pred[df_pred["full_name"] == matched_name]
            for _, pred_row in pred_matches.iterrows():
                results.append({**gt_row, **pred_row})
    
    return pd.DataFrame(results) if results else pd.DataFrame()


# Date Matching

_CURRENT_YEAR = 2025  # used for two-digit year correction


def _two_digit_year_fix(ts, gt_year=None):
    """
    Fix two-digit years that pandas infers incorrectly.
    
    pandas uses a 50-year pivot (≥50 → 1900s, <50 → 2000s) which can
    misinterpret years. This corrects for that.
    """
    if pd.isna(ts):
        return ts
    
    year = ts.year
    # If pandas mapped a two-digit year to future (e.g. 2082), correct it
    if year > _CURRENT_YEAR:
        year -= 100
        ts = ts.replace(year=year)
    
    return ts


def parse_date(val, gt_year=None):
    """
    Parse a date string in any common format; return pd.Timestamp or NaT.
    
    Args:
        val: Date string or value
        gt_year: Ground truth year (for correcting two-digit year inference)
    
    Returns:
        pd.Timestamp or pd.NaT
    """
    if pd.isna(val) or str(val).strip() in ("", "nan", "None"):
        return pd.NaT
    
    for fmt in ("%Y-%m-%d", "%m/%d/%y", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            ts = pd.to_datetime(str(val).strip(), format=fmt)
            return _two_digit_year_fix(ts, gt_year)
        except ValueError:
            continue
    
    ts = pd.to_datetime(str(val), errors="coerce")
    return _two_digit_year_fix(ts, gt_year) if not pd.isna(ts) else ts


def birthdate_scores(gt_val, pred_val):
    """
    Calculate birthdate matching scores (exact, year, month).
    
    Returns NaN when either value is missing to avoid penalizing
    missing ground truth data.
    
    Args:
        gt_val: Ground truth date
        pred_val: Predicted date
    
    Returns:
        Dict with keys: exact, year, month (each 0.0-1.0 or NaN)
    """
    nan_result = {
        "exact": float("nan"),
        "year": float("nan"),
        "month": float("nan")
    }
    
    gt_ts = parse_date(gt_val)
    gt_year = gt_ts.year if not pd.isna(gt_ts) else None
    pred_ts = parse_date(pred_val, gt_year=gt_year)
    
    if pd.isna(gt_ts) or pd.isna(pred_ts):
        return nan_result
    
    return {
        "exact": float(gt_ts == pred_ts),
        "year": float(gt_ts.year == pred_ts.year),
        "month": float(gt_ts.month == pred_ts.month),
    }


# Discrete Field Matching

def gender_match_score(gt_val, pred_val):
    """
    Calculate gender match score (case-insensitive exact match).
    
    Missing value handling:
    - GT missing, prediction present: NaN — ground truth unavailable due to
      data collection gaps (5/100 senators), not because signal is absent
      from the bio. Do not conflate with Liu et al.'s NaN logic for
      genuinely absent profile fields.
    - GT present, prediction missing: 0.0 — extraction failure; gender is
      inferrable from pronouns and titles in virtually every senator bio,
      so a missing prediction is penalized.
    - Both missing: NaN
    - Both present: 1.0 if case-insensitive match, else 0.0
    
    Args:
        gt_val: Ground truth gender
        pred_val: Predicted gender
    
    Returns:
        1.0, 0.0, or float('nan')
    """
    if pd.isna(gt_val) or str(gt_val).strip() == "":
        return float("nan")
    if pd.isna(pred_val) or str(pred_val).strip() == "":
        return 0.0
    
    return float(str(gt_val).strip().lower() == str(pred_val).strip().lower())


# Religion Matching

def get_religion_category(religion_str, religion_hierarchy=None):
    """
    Normalize religion string and return its category from hierarchy.
    
    Args:
        religion_str: Religion string
        religion_hierarchy: Dict mapping religions to categories
    
    Returns:
        Category string or None if missing
    """
    if religion_hierarchy is None:
        religion_hierarchy = {}
    
    if pd.isna(religion_str) or str(religion_str).strip() == "":
        return None
    
    norm = str(religion_str).strip().lower()
    
    # Direct lookup
    if norm in religion_hierarchy:
        return religion_hierarchy[norm]
    
    # Substring matching for common phrases
    for key, category in religion_hierarchy.items():
        if key in norm or norm in key:
            return category
    
    # Default: treat as its own category
    return norm


def religion_match_score(gt_val, pred_val, religion_hierarchy=None):
    """
    Hierarchical religion matching:
    - 1.0 for exact match (after lowercasing)
    - 0.7 for parent-child match (e.g., Methodist vs Christian)
    - 0.0 for unrelated religions
    - NaN when either GT or pred is missing
    
    Args:
        gt_val: Ground truth religion
        pred_val: Predicted religion
        religion_hierarchy: Dict mapping religions to categories
    
    Returns:
        Float: 1.0, 0.7, 0.0, or NaN
    """
    if religion_hierarchy is None:
        religion_hierarchy = {}
    
    if pd.isna(gt_val) or str(gt_val).strip() == "":
        return float("nan")
    if pd.isna(pred_val) or str(pred_val).strip() == "":
        return float("nan")
    
    gt_norm = str(gt_val).strip().lower()
    pred_norm = str(pred_val).strip().lower()
    
    # Exact match
    if gt_norm == pred_norm:
        return 1.0
    
    # Hierarchical match
    gt_cat = get_religion_category(gt_norm, religion_hierarchy)
    pred_cat = get_religion_category(pred_norm, religion_hierarchy)
    
    if gt_cat and pred_cat:
        if gt_cat == pred_cat:
            # Same category but different names — partial credit
            return 0.7
    
    # No match
    return 0.0


# Text Parsing & Formatting

def parse_education(edu_str):
    """
    Parse education field (JSON list or pipe-delimited) into readable text.
    Input formats:
      - JSON: [{'degree': 'B.A.', 'institution': 'Harvard', 'year': 2020}, ...]
      - Pipe: "B.A.|Harvard|2020|M.A.|Stanford|2022"
      - String: Any text
    Output: Normalized string combining degree, institution, year
    """
    if pd.isna(edu_str) or not str(edu_str).strip():
        return ""
    
    edu_str = str(edu_str).strip()
    
    # Try to parse as JSON list
    try:
        if edu_str.startswith("["):
            items = json.loads(edu_str)
            parts = []
            for item in items:
                if isinstance(item, dict):
                    degree = item.get("degree", "").strip() if item.get("degree") else ""
                    institution = item.get("institution", "").strip() if item.get("institution") else ""
                    year = item.get("year", "")
                    
                    entry = " ".join(filter(None, [degree, institution, str(year) if year else ""]))
                    if entry:
                        parts.append(entry)
            return " ".join(parts) if parts else edu_str
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Try pipe-delimited format
    if "|" in edu_str:
        return edu_str.replace("|", " ")
    
    return edu_str


def parse_committee_roles(roles_str):
    """
    Parse committee roles field (JSON list, pipe-delimited, or string).
    Normalize to readable text by joining all roles.
    """
    if pd.isna(roles_str) or not str(roles_str).strip():
        return ""
    
    roles_str = str(roles_str).strip()
    
    # Try to parse as JSON list
    try:
        if roles_str.startswith("["):
            items = json.loads(roles_str)
            if isinstance(items, list):
                return " ".join(str(item).strip() for item in items if item)
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Try pipe-delimited
    if "|" in roles_str:
        return roles_str.replace("|", " ")
    
    return roles_str


def parse_education_detailed(edu_str):
    """
    Parse education into structured list of components: [(degree, institution, year), ...]
    Used for component-level comparison (degree exact, institution fuzzy, year exact).
    """
    import ast
    
    if pd.isna(edu_str) or not str(edu_str).strip():
        return []
    
    edu_str = str(edu_str).strip()
    items = []
    
    # Try JSON format (handles both double-quoted and single-quoted JSON)
    if edu_str.startswith("["):
        # First try standard JSON
        try:
            parsed = json.loads(edu_str)
        except (json.JSONDecodeError, TypeError):
            # Fall back to ast.literal_eval for single-quoted JSON
            try:
                parsed = ast.literal_eval(edu_str)
            except (ValueError, SyntaxError):
                parsed = None
        
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    degree = item.get("degree")
                    institution = item.get("institution")
                    year = item.get("year")
                    
                    # Only add if at least one field is non-None
                    if degree or institution or year:
                        items.append((
                            str(degree).strip() if degree else "",
                            str(institution).strip() if institution else "",
                            str(year).strip() if year else ""
                        ))
            return items if items else []
    
    # Try pipe format: "degree1|institution1|year1|degree2|institution2|year2"
    if "|" in edu_str:
        parts = [p.strip() for p in edu_str.split("|")]
        for i in range(0, len(parts), 3):
            if i+2 < len(parts):
                items.append((parts[i], parts[i+1], parts[i+2]))
            elif i+1 < len(parts):
                items.append((parts[i], parts[i+1], ""))
            elif i < len(parts):
                items.append((parts[i], "", ""))
        return items
    
    # Fallback: treat entire string as one entry with no degree/year
    return [("", edu_str, "")]


def compare_education_components(gt_items, pred_items):
    """
    Compare education components: degree (exact), institution (fuzzy), year (exact).
    Return: Dict with keys: degree_exact, institution_fuzzy, year_exact, combined_score
    """
    if not gt_items or not pred_items:
        return {
            "degree_exact": float("nan"),
            "institution_fuzzy": float("nan"),
            "year_exact": float("nan"),
            "combined_score": float("nan"),
        }
    
    def normalize_inst(s):
        return s.lower().replace(",", "").replace("university", "u").strip()
    
    # Match by institution first (handles reordering)
    matched = {}
    for gt_idx, gt_item in enumerate(gt_items):
        best_match_idx = -1
        best_score = 0
        
        gt_inst = normalize_inst(gt_item[1])
        for pred_idx, pred_item in enumerate(pred_items):
            if pred_idx in matched.values():
                continue
            pred_inst = normalize_inst(pred_item[1])
            
            # Simple overlap score
            if gt_inst and pred_inst:
                score = SequenceMatcher(None, gt_inst, pred_inst).ratio()
                if score > best_score:
                    best_score = score
                    best_match_idx = pred_idx
        
        if best_match_idx >= 0:
            matched[gt_idx] = best_match_idx
    
    # Compute component scores
    degree_scores = []
    institution_scores = []
    year_scores = []
    
    for gt_idx, pred_idx in matched.items():
        gt_deg, gt_inst, gt_year = gt_items[gt_idx]
        pred_deg, pred_inst, pred_year = pred_items[pred_idx]
        
        # Exact match on degree
        if gt_deg and pred_deg:
            degree_scores.append(float(gt_deg.lower() == pred_deg.lower()))
        
        # Fuzzy match on institution
        if gt_inst and pred_inst:
            gt_inst_norm = normalize_inst(gt_inst)
            pred_inst_norm = normalize_inst(pred_inst)
            ratio = SequenceMatcher(None, gt_inst_norm, pred_inst_norm).ratio()
            institution_scores.append(1.0 if ratio > 0.8 else 0.0)
        
        # Exact match on year
        if gt_year and pred_year:
            year_scores.append(float(gt_year == pred_year))
    
    # Aggregate
    result = {
        "degree_exact": sum(degree_scores) / len(degree_scores) if degree_scores else float("nan"),
        "institution_fuzzy": sum(institution_scores) / len(institution_scores) if institution_scores else float("nan"),
        "year_exact": sum(year_scores) / len(year_scores) if year_scores else float("nan"),
    }
    
    # Combined score: average of all components
    all_scores = [s for s in [result["degree_exact"], result["institution_fuzzy"], result["year_exact"]] if not pd.isna(s)]
    result["combined_score"] = sum(all_scores) / len(all_scores) if all_scores else float("nan")
    
    return result


# Text Field & Component Evaluation

def evaluate_text_fields(merged_df, fields, rouge_scorer, bert_scorer=None):
    """
    Compute Rouge-1 and optionally BERTScore F1 for one or more text fields.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Already-merged GT + predictions DataFrame.
    fields : list of (field_name, gt_col, pred_col, parse_fn)
    rouge_scorer : rouge_score.RougeScorer instance
    bert_scorer : if None, BERTScore is skipped

    Returns
    -------
    dict: { field_name: { "rouge1": float, "bertscore_f1": float | None, "n": int } }
    """
    import bert_score as bs

    results = {}
    for field, gt_col, pred_col, parse_fn in fields:
        if gt_col not in merged_df.columns or pred_col not in merged_df.columns:
            continue

        gts   = [str(r).strip() for r in merged_df[gt_col].fillna("")]
        preds = [parse_fn(str(r)) for r in merged_df[pred_col].fillna("")]
        pairs = [(g, p) for g, p in zip(gts, preds) if g]

        if not pairs:
            continue

        gt_list, pred_list = zip(*pairs)

        rouge1 = sum(
            rouge_scorer.score(g, p)["rouge1"].fmeasure
            for g, p in zip(gt_list, pred_list)
        ) / len(gt_list)

        bert_f1 = None
        if bert_scorer is not None:
            _, _, F = bs.score(list(pred_list), list(gt_list), lang="en", verbose=False)
            bert_f1 = F.mean().item()

        results[field] = {"rouge1": rouge1, "bertscore_f1": bert_f1, "n": len(gt_list)}

    return results


def evaluate_education_components(merged_df):
    """
    Compute component-level education scores (degree, institution, year).

    Returns
    -------
    dict: { "degree_exact": float, "institution_fuzzy": float,
            "year_exact": float, "combined_score": float, "n": int }
    """
    buckets = {"degree_exact": [], "institution_fuzzy": [], "year_exact": [], "combined_score": []}

    for _, row in merged_df.iterrows():
        try:
            gt_items   = parse_education_detailed(row.get("education_text", "") or "")
            pred_items = parse_education_detailed(row.get("education", "") or "")
            result     = compare_education_components(gt_items, pred_items)
            for key, val in result.items():
                if not pd.isna(val):
                    buckets[key].append(val)
        except Exception:
            pass

    return {
        k: (sum(v) / len(v) if v else float("nan"))
        for k, v in buckets.items()
    } | {"n": len(buckets["combined_score"])}


__all__ = [
    "normalize_name",
    "name_match_score",
    "create_normalized_senator_id",
    "match_by_fuzzy_name",
    "parse_date",
    "birthdate_scores",
    "gender_match_score",
    "get_religion_category",
    "religion_match_score",
    "parse_education",
    "parse_committee_roles",
    "parse_education_detailed",
    "compare_education_components",
    "evaluate_text_fields",
    "evaluate_education_components",
]
