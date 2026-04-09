"""
Data merging and signal detection utilities.
Consolidates functions for merging external data sources and detecting religious signals.
"""

import pandas as pd
import re
from typing import Optional, Tuple


def merge_pew(senators_df: pd.DataFrame, pew_path: str) -> Optional[pd.Series]:
    """
    Merge Pew religion data into senator dataframe.
    
    Args:
        senators_df: DataFrame with senator data (expects 'name' column)
        pew_path: Path to Pew religion CSV file
        
    Returns:
        Series of religion values aligned with senators_df, or None if merge fails
    """
    try:
        pew_df = pd.read_csv(pew_path)
    except Exception as e:
        print(f"Error loading Pew data: {e}")
        return None
    
    # Expected columns: name, religion
    if 'name' not in pew_df.columns or 'religion' not in pew_df.columns:
        print("Pew data missing 'name' or 'religion' columns")
        return None
    
    # Merge on name (case-insensitive)
    senators_df['name_lower'] = senators_df['name'].str.lower() if 'name' in senators_df.columns else ""
    pew_df['name_lower'] = pew_df['name'].str.lower()
    
    merged = senators_df.merge(
        pew_df[['name_lower', 'religion']],
        how='left',
        on='name_lower'
    )
    
    # Drop temporary column
    if 'name_lower' in merged.columns:
        merged.drop('name_lower', axis=1, inplace=True)
    
    return merged['religion'] if 'religion' in merged.columns else None


def detect_religion_signal(text: str) -> Tuple[bool, str]:
    """
    Detect whether religious affiliation is explicitly mentioned in text.
    
    Returns:
        (is_explicit: bool, signal_type: str)
        signal_type can be: "explicit", "organizational", "language", "none"
    """
    if not text or not isinstance(text, str):
        return False, "none"
    
    text_lower = text.lower()
    
    # Explicit mention patterns
    explicit_patterns = [
        r'\b(is|are|was|were)\s+(a|an)?\s*(christian|catholic|jewish|muslim|buddhist|hindu|atheist|agnostic)',
        r'\b(believes?|faith|religion)',
        r'\b(member|chair)\s+of\s+(church|synagogue|mosque|temple|congregation)',
    ]
    
    for pattern in explicit_patterns:
        if re.search(pattern, text_lower):
            return True, "explicit"
    
    # Organizational signals
    org_patterns = [
        r'\b(church|synagogue|mosque|temple|congregation|ministry)',
        r'\b(christian|baptist|methodist|catholic|jewish|muslim)',
    ]
    
    for pattern in org_patterns:
        if re.search(pattern, text_lower):
            return True, "organizational"
    
    # Language/value signals
    language_patterns = [
        r'\b(grace|prayer|faith|spirit|blessing|spiritual)',
        r'\b(values?|principle|moral|ethical)',
    ]
    
    for pattern in language_patterns:
        if re.search(pattern, text_lower):
            return True, "language"
    
    return False, "none"


def normalize_birthdate(date_str: str) -> Optional[str]:
    """
    Normalize birthdate to YYYY-MM-DD format.
    
    Args:
        date_str: Date string (various formats supported)
        
    Returns:
        Normalized date in YYYY-MM-DD or None
    """
    if not date_str or pd.isna(date_str):
        return None
    
    date_str = str(date_str).strip()
    
    # Common date format patterns
    patterns = [
        (r'(\d{4})-(\d{1,2})-(\d{1,2})', lambda m: f"{m[1]}-{m[2]:0>2s}-{m[3]:0>2s}"),
        (r'(\d{1,2})/(\d{1,2})/(\d{4})', lambda m: f"{m[3]}-{m[1]:0>2s}-{m[2]:0>2s}"),
        (r'(\w+)\s+(\d{1,2}),?\s+(\d{4})', lambda m: f"{m[3]}-{_month_to_num(m[1]):0>2s}-{m[2]:0>2s}"),
    ]
    
    for pattern, formatter in patterns:
        match = re.match(pattern, date_str)
        if match:
            try:
                groups = [m if isinstance(m, str) else str(m) for m in match.groups()]
                return formatter(groups)
            except Exception:
                pass
    
    return None


def _month_to_num(month: str) -> str:
    """Convert month name to number."""
    months = {
        'january': '01', 'february': '02', 'march': '03', 'april': '04',
        'may': '05', 'june': '06', 'july': '07', 'august': '08',
        'september': '09', 'october': '10', 'november': '11', 'december': '12'
    }
    return months.get(month.lower()[:3], '01')
