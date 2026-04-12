"""
Consolidated parsing utilities for structured field extraction.

Handles education, committee roles, and other structured data from both
LLM-extracted JSON and ground truth pipe-delimited formats.
"""

import ast
from difflib import SequenceMatcher
import pandas as pd


class EducationParser:
    """Single source of truth for education parsing (consolidates 3 variants)."""
    
    @staticmethod
    def parse(s, format_type="auto"):
        """
        Parse education data into structured list of (degree, institution, year) tuples.
        Handles both JSON format (LLM output) and pipe-delimited format (GT).
        
        Args:
            s: Education string or list in JSON or pipe-delimited format
            format_type: "auto" (try both), "json", or "pipe"
        
        Returns:
            List of dicts with keys: degree, institution, year
        """
        if not s or (isinstance(s, float) and pd.isna(s)):
            return []
        
        s_str = str(s).strip()
        
        if format_type in ("auto", "json"):
            # Try JSON format first (LLM output)
            if s_str.startswith("["):
                try:
                    edu_list = ast.literal_eval(s_str)
                    education_items = []
                    for edu in edu_list:
                        if isinstance(edu, dict):
                            degree = str(edu.get("degree", "")).strip() if edu.get("degree") else None
                            institution = str(edu.get("institution", "")).strip() if edu.get("institution") else None
                            year = str(edu.get("year", "")).strip() if edu.get("year") else None
                            if institution:  # Only include if we have at least institution
                                education_items.append({
                                    "degree": degree,
                                    "institution": institution,
                                    "year": year
                                })
                    return education_items
                except Exception:
                    pass
        
        if format_type in ("auto", "pipe"):
            # Try pipe-delimited format (GT: "degree|institution|year|degree2|institution2|year2")
            if "|" in s_str:
                try:
                    parts = [p.strip() for p in s_str.split("|")]
                    education_items = []
                    # Process in groups of 3 (degree, institution, year)
                    for i in range(0, len(parts), 3):
                        if i + 1 < len(parts):  # At least need degree and institution
                            degree = parts[i] if i < len(parts) and parts[i] else None
                            institution = parts[i + 1] if i + 1 < len(parts) and parts[i + 1] else None
                            year = parts[i + 2] if i + 2 < len(parts) and parts[i + 2] else None
                            
                            # Skip placeholder years like "no year given"
                            if year and "no year" in year.lower():
                                year = None
                            
                            if institution:  # Only include if we have at least institution
                                education_items.append({
                                    "degree": degree,
                                    "institution": institution,
                                    "year": year
                                })
                    return education_items
                except Exception:
                    pass
        
        return []
    
    @staticmethod
    def get_institutions_only(education_items):
        """Extract just institution names (for backward compatibility)."""
        return [e["institution"] for e in education_items if e.get("institution")]


def parse_education_detailed(s, format_type="auto"):
    """Backward compatibility wrapper for EducationParser."""
    return EducationParser.parse(s, format_type=format_type)


def parse_education(s):
    """Simple education parsing (institutions only) for backward compatibility."""
    if not s or (isinstance(s, float) and pd.isna(s)):
        return ""
    try:
        s_str = str(s).strip()
        if s_str.startswith("["):
            edu_list = ast.literal_eval(s_str)
            institutions = [
                str(edu["institution"])
                for edu in edu_list
                if isinstance(edu, dict) and edu.get("institution")
            ]
            return "|".join(institutions)
    except Exception:
        pass
    return str(s)


def parse_committee_roles(s):
    """Parse committee roles from JSON or string format."""
    if not s or (isinstance(s, float) and pd.isna(s)):
        return ""
    try:
        s_str = str(s).strip()
        if s_str.startswith("["):
            roles = ast.literal_eval(s_str)
            return "|".join(roles) if isinstance(roles, list) else str(s)
    except Exception:
        pass
    return str(s)


class DegreeNormalizer:
    """Normalize degree strings for comparison."""
    
    EXACT_MATCHES = {
        "MBA": "MBA",
        "JD": "JD",
        "BS": "BS",
        "BA": "BA",
        "MA": "MA",
        "MS": "MS",
        "MIA": "MIA",    # Master of International Affairs
        "PHD": "PHD",
    }
    
    SUBSTRING_MAPPINGS = [
        ("BACHELOR OF SCIENCE", "BS"),
        ("BACHELOR OF ARTS", "BA"),
        ("BACHELORS SCIENCE", "BS"),
        ("BACHELORS ARTS", "BA"),
        ("BACHELOR DEGREE", "BA"),
        ("BACHELORS DEGREE", "BA"),
        ("JURIS DOCTOR", "JD"),
        ("JURIS DOCTORATE", "JD"),
        ("MASTER OF BUSINESS", "MBA"),  # MBA must come before MA
        ("MASTER OF SCIENCE", "MS"),
        ("MASTER OF ARTS", "MA"),
        ("DOCTOR OF PHILOSOPHY", "PHD"),
        ("DOCTORATE", "PHD"),
    ]
    
    @classmethod
    def normalize(cls, deg):
        """Normalize degree strings (B.S. → BS, B.A. → BA, etc.)."""
        if not deg or pd.isna(deg):
            return ""
        
        deg = str(deg).upper().strip()
        # Remove periods and extra spaces
        deg = deg.replace(".", "").replace(",", "").replace("'", "")
        
        # If exact match, return immediately
        if deg in cls.EXACT_MATCHES:
            return cls.EXACT_MATCHES[deg]
        
        # Substring/phrase mappings (longer patterns first to avoid partial matches)
        for pattern, result in cls.SUBSTRING_MAPPINGS:
            if pattern in deg:
                return result
        
        return deg


def normalize_degree(deg):
    """Backward compatibility wrapper for DegreeNormalizer."""
    return DegreeNormalizer.normalize(deg)


class SchoolNormalizer:
    """Normalize school names for fuzzy matching, with canonical alias resolution."""

    # Canonical alias map: common variants → canonical form used for comparison.
    # Keys are lowercased; values are the normalized canonical string.
    ALIASES = {
        # Abbreviations
        "mit":          "massachusetts institute technology",
        "ucla":         "california los angeles",
        "usc":          "southern california",
        "unc":          "north carolina chapel hill",
        "unc chapel hill": "north carolina chapel hill",
        "nc state":     "north carolina state",
        "ohio state":   "ohio state",
        "uva":          "virginia",
        "uw":           "washington",
        "cu":           "columbia",
        "bu":           "boston",
        "bc":           "boston college",
        "nyu":          "new york",
        "lsu":          "louisiana state",
        "tcu":          "texas christian",
        "smu":          "southern methodist",
        "tamu":         "texas a&m",
        "texas a&m":    "texas a&m",
        "a&m":          "texas a&m",
        "psu":          "penn state",
        "penn state":   "penn state",
        "fsu":          "florida state",
        "asu":          "arizona state",
        "csu":          "colorado state",
        "vcu":          "virginia commonwealth",
        "gmu":          "george mason",
        "gwu":          "george washington",
        "georgetown":   "georgetown",

        # Ivy League variants
        "harvard":      "harvard",
        "yale":         "yale",
        "princeton":    "princeton",
        "columbia":     "columbia",
        "penn":         "pennsylvania",
        "upenn":        "pennsylvania",
        "dartmouth":    "dartmouth",
        "brown":        "brown",
        "cornell":      "cornell",

        # Common misspellings / shorthand
        "u of michigan":    "michigan",
        "u michigan":       "michigan",
        "u of florida":     "florida",
        "u florida":        "florida",
        "u of texas":       "texas austin",
        "ut austin":        "texas austin",
        "u of chicago":     "chicago",
        "u chicago":        "chicago",
        "u of illinois":    "illinois",
        "u illinois":       "illinois",
    }

    # Stop words stripped before comparison.
    # "institute" intentionally excluded — kept so "Massachusetts Institute of Technology"
    # strips to "massachusetts institute technology", matching the MIT alias canonical form.
    STOP_WORDS = {
        "university", "college", "school", "institution",
        "of", "the", "at", "and", "&",
    }

    @classmethod
    def normalize(cls, school):
        """
        Normalize a school name:
          1. Lowercase + strip punctuation
          2. Resolve alias if found
          3. Strip stop words
        """
        if not school or pd.isna(school):
            return ""

        s = str(school).lower().strip()
        s = s.replace(".", "").replace(",", "")

        # Alias lookup (exact key match after lowercasing)
        if s in cls.ALIASES:
            return cls.ALIASES[s]

        # Partial alias lookup (for cases like "university of michigan" → key "u of michigan" won't match)
        # Strip stop words first, then check aliases
        tokens = [t for t in s.split() if t not in cls.STOP_WORDS]
        stripped = " ".join(tokens)

        if stripped in cls.ALIASES:
            return cls.ALIASES[stripped]

        return stripped


def normalize_school(school):
    """Backward compatibility wrapper for SchoolNormalizer."""
    return SchoolNormalizer.normalize(school)

def _score_single_education_pair(gt_item, pred_item):
    """
    Score one GT education entry against one predicted entry.
    Returns dict: {degree_exact, institution_fuzzy, year_exact, combined}
    Each component is 0.0, 1.0, or NaN if both sides are missing.
    """
    # ── Degree ───────────────────────────────────────────────────────────────
    gt_deg  = DegreeNormalizer.normalize(gt_item.get("degree") or "")
    pred_deg = DegreeNormalizer.normalize(pred_item.get("degree") or "")

    if gt_deg and pred_deg:
        degree_score = float(gt_deg == pred_deg)
    elif not gt_deg and not pred_deg:
        degree_score = float("nan")
    else:
        degree_score = 0.0  # One side present, other missing → extraction failure

    # ── Institution ──────────────────────────────────────────────────────────
    gt_school   = SchoolNormalizer.normalize(gt_item.get("institution") or "")
    pred_school = SchoolNormalizer.normalize(pred_item.get("institution") or "")

    if gt_school and pred_school:
        institution_score = (
            1.0 if (
                gt_school == pred_school
                or gt_school in pred_school
                or pred_school in gt_school
                or SequenceMatcher(None, gt_school, pred_school).ratio() > 0.80
            ) else 0.0
        )
    elif not gt_school and not pred_school:
        institution_score = float("nan")
    else:
        institution_score = 0.0

    # ── Year ─────────────────────────────────────────────────────────────────
    gt_year   = str(gt_item.get("year") or "").strip()
    pred_year = str(pred_item.get("year") or "").strip()

    if gt_year and pred_year:
        year_score = float(gt_year == pred_year)
    elif not gt_year and not pred_year:
        year_score = float("nan")
    else:
        year_score = 0.0

    # ── Combined: mean of non-NaN components ────────────────────────────────
    components = [s for s in [degree_score, institution_score, year_score]
                  if not (isinstance(s, float) and s != s)]  # exclude NaN
    combined = sum(components) / len(components) if components else float("nan")

    return {
        "degree_exact":      degree_score,
        "institution_fuzzy": institution_score,
        "year_exact":        year_score,
        "combined":          combined,
    }


def compare_education_components(gt_items, pred_items):
    """
    Compare all GT education entries against predictions using best-match alignment.

    Strategy:
      For each GT degree, find the best-matching prediction (highest combined score).
      Average those best-match scores across all GT degrees.
      This penalizes missed degrees (GT entry with no good pred match → 0.0)
      while not penalizing the model for extracting extra degrees.

    Args:
        gt_items:   List of dicts from parse_education_detailed (ground truth)
        pred_items: List of dicts from parse_education_detailed (model output)

    Returns:
        dict: {
            degree_exact:      float,  # avg across GT degrees
            institution_fuzzy: float,  # avg across GT degrees
            year_exact:        float,  # avg across GT degrees
            combined_score:    float,  # avg combined per GT degree
            n_gt:              int,    # number of GT degrees evaluated
        }
    """
    nan_result = {
        "degree_exact":      float("nan"),
        "institution_fuzzy": float("nan"),
        "year_exact":        float("nan"),
        "combined_score":    float("nan"),
        "n_gt":              0,
    }

    if not gt_items or not pred_items:
        return nan_result

    degree_scores      = []
    institution_scores = []
    year_scores        = []
    combined_scores    = []

    for gt_item in gt_items:
        # Score this GT entry against every prediction, keep the best
        candidate_scores = [
            _score_single_education_pair(gt_item, pred_item)
            for pred_item in pred_items
        ]

        best = max(candidate_scores, key=lambda x: (
            x["combined"] if not (isinstance(x["combined"], float) and x["combined"] != x["combined"])
            else -1.0
        ))

        # If no prediction came close, combined will be 0.0 — that's correct (penalize miss)
        degree_scores.append(best["degree_exact"] if not _is_nan(best["degree_exact"]) else 0.0)
        institution_scores.append(best["institution_fuzzy"] if not _is_nan(best["institution_fuzzy"]) else 0.0)
        year_scores.append(best["year_exact"] if not _is_nan(best["year_exact"]) else 0.0)
        combined_scores.append(best["combined"] if not _is_nan(best["combined"]) else 0.0)

    def _avg(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    return {
        "degree_exact":      _avg(degree_scores),
        "institution_fuzzy": _avg(institution_scores),
        "year_exact":        _avg(year_scores),
        "combined_score":    _avg(combined_scores),
        "n_gt":              len(gt_items),
    }


# ============================================================================
# DATE PARSING (consolidated from evaluator.py)
# ============================================================================

_CURRENT_YEAR = 2025  # Used for two-digit year correction


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


__all__ = [
    "EducationParser",
    "parse_education_detailed",
    "parse_education",
    "parse_committee_roles",
    "DegreeNormalizer",
    "normalize_degree",
    "SchoolNormalizer",
    "normalize_school",
    "compare_education_components",
    "parse_date",
    "birthdate_scores",
]
