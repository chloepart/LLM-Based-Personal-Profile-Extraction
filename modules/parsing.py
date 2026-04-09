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
    """Normalize school names for fuzzy matching."""
    
    @staticmethod
    def normalize(school):
        """Normalize school names for fuzzy matching."""
        if not school or pd.isna(school):
            return ""
        school = str(school).lower().strip()
        # Remove common suffixes
        for suffix in ["university", "college", "school", "institute", "of", "the"]:
            school = school.replace(suffix, "").strip()
        return school


def normalize_school(school):
    """Backward compatibility wrapper for SchoolNormalizer."""
    return SchoolNormalizer.normalize(school)


def compare_education_components(gt_items, pred_items):
    """
    Compare education detail by detail: degree, institution, year.
    Returns dict with per-component match counts and a combined score.
    """
    if not gt_items or not pred_items:
        return {
            "degree_exact": float("nan"),
            "institution_fuzzy": float("nan"),
            "year_exact": float("nan"),
            "combined_score": float("nan")
        }
    
    # Simple matching: compare first degree from each
    degree_matches = 0
    school_matches = 0
    year_matches = 0
    
    # Compare degrees
    if gt_items[0].get("degree") and pred_items[0].get("degree"):
        gt_deg = DegreeNormalizer.normalize(gt_items[0]["degree"])
        pred_deg = DegreeNormalizer.normalize(pred_items[0]["degree"])
        degree_matches = float(gt_deg == pred_deg)
    
    # Compare institutions (fuzzy)
    if gt_items[0].get("institution") and pred_items[0].get("institution"):
        gt_school = SchoolNormalizer.normalize(gt_items[0]["institution"])
        pred_school = SchoolNormalizer.normalize(pred_items[0]["institution"])
        # Fuzzy match via substring or SequenceMatcher
        school_matches = 1.0 if (gt_school in pred_school or pred_school in gt_school or
                                 SequenceMatcher(None, gt_school, pred_school).ratio() > 0.80) else 0.0
    
    # Compare years
    if gt_items[0].get("year") and pred_items[0].get("year"):
        gt_year = str(gt_items[0]["year"]).strip()
        pred_year = str(pred_items[0]["year"]).strip()
        year_matches = float(gt_year == pred_year)
    
    # Combined: average of the three
    components = [degree_matches, school_matches, year_matches]
    combined = sum(components) / len(components) if components else float("nan")
    
    return {
        "degree_exact": degree_matches,
        "institution_fuzzy": school_matches,
        "year_exact": year_matches,
        "combined_score": combined
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
]
