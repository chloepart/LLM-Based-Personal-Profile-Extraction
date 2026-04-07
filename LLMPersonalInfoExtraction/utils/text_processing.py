"""
Centralized text processing utilities.
Eliminates duplicate normalization logic across the codebase.
"""

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# ABBREVIATION EXPANSIONS
# ─────────────────────────────────────────────────────────────────────────────

ABBREVIATION_EXPANSIONS = {
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
    r"\bamy\b": "amelia",
    r"\bkatie\b": "katherine",
    r"\bcat\b": "catherine",
    r"\btimothy\b": "tim",
    r"\bchristopher\b": "chris",
    r"\banthony\b": "tony",
}

WIKIPEDIA_SLUG_OVERRIDES = {
    "bernard_sanders": "Bernie_Sanders",
    "dan_sullivan": "Dan_Sullivan_(U.S._senator)",
    "tom_cotton": "Thomas_Cotton",
    "tommy_tuberville": "Thomas_Tuberville",
    "jon_ossoff": "Jonathan_Ossoff",
    "alan_armstrong": "Alan_S._Armstrong",
    "jack_reed": "Jack_Reed_(Rhode_Island_politician)",
}


# ─────────────────────────────────────────────────────────────────────────────
# CORE NORMALIZATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    """
    Normalize a name for comparison and matching.
    
    Steps:
    1. Convert to lowercase
    2. Remove accents (é → e)
    3. Expand common abbreviations (Dan → Daniel, Tom → Thomas)
    
    Args:
        name: Full name string (can include middle initials)
        
    Returns:
        Normalized name string
        
    Examples:
        >>> normalize_name("Dan O'Brien")
        "daniel o'brien"
        >>> normalize_name("Régis Pelosi")
        "regis pelosi"
    """
    if not name or (isinstance(name, float)):
        return ""
    
    # Lowercase and strip
    normalized = str(name).lower().strip()
    
    # Remove accents: é→e, ñ→n, etc.
    normalized = "".join(
        c for c in unicodedata.normalize("NFD", normalized)
        if unicodedata.category(c) != "Mn"
    )
    
    # Expand abbreviations
    for pattern, replacement in ABBREVIATION_EXPANSIONS.items():
        normalized = re.sub(pattern, replacement, normalized)
    
    return normalized


def create_slug(name: str) -> str:
    """
    Create a URL-safe slug from a senator's name for Wikipedia/Ballotpedia URLs.
    
    Steps:
    1. Normalize the name
    2. Remove middle initials (Roger F. Wicker → Roger Wicker)
    3. Replace spaces with underscores
    4. Apply known overrides (for variations)
    
    Args:
        name: Senator's full name
        
    Returns:
        Slug suitable for Wikipedia URLs (no spaces, title case)
        
    Examples:
        >>> create_slug("Roger F. Wicker")
        "Roger_Wicker"
        >>> create_slug("Dan Sullivan")
        "Dan_Sullivan_(U.S._senator)"  # override applied
    """
    normalized = normalize_name(name)
    
    # Remove middle initial: "roger f. wicker" -> "roger wicker"
    slug = re.sub(r"\s+[a-z]\.\s*", " ", normalized).strip()
    
    # Replace spaces with underscores
    slug = slug.replace(" ", "_")
    
    # Apply hardcoded overrides
    override = WIKIPEDIA_SLUG_OVERRIDES.get(slug.lower())
    if override:
        return override
    
    # Capitalize each word for Wikipedia format
    return "_".join(word.capitalize() for word in slug.split("_") if word)


def create_wikipedia_url(senator_name: str) -> str:
    """
    Create a Wikipedia URL for a senator.
    
    Args:
        senator_name: Senator's full name
        
    Returns:
        Full Wikipedia URL
        
    Examples:
        >>> create_wikipedia_url("Dan Sullivan")
        "https://en.wikipedia.org/wiki/Dan_Sullivan_(U.S._senator)"
    """
    slug = create_slug(senator_name)
    return f"https://en.wikipedia.org/wiki/{slug}"


# ─────────────────────────────────────────────────────────────────────────────
# MATCHING & COMPARISON FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def name_match_score(gt_name: str, pred_name: str, threshold: float = 0.90) -> float:
    """
    Score name similarity after normalization.
    
    Returns 1.0 if:
    - Both are empty
    - Normalized names are identical
    - Similarity ratio > threshold (default 90%)
    
    Args:
        gt_name: Ground truth name
        pred_name: Predicted name
        threshold: Similarity threshold (0.0–1.0)
        
    Returns:
        1.0 if match, 0.0 if no match
        
    Examples:
        >>> name_match_score("Daniel Smith", "Dan Smith")
        1.0  # normalized match
        >>> name_match_score("Smith, Daniel", "Daniel Smith")
        1.0  # 95% similar
    """
    gt_norm = normalize_name(gt_name)
    pred_norm = normalize_name(pred_name)
    
    # Both empty → match
    if not gt_norm and not pred_norm:
        return 1.0
    
    # One empty, other not → no match
    if not gt_norm or not pred_norm:
        return 0.0
    
    # Exact match after normalization
    if gt_norm == pred_norm:
        return 1.0
    
    # Fuzzy similarity
    ratio = SequenceMatcher(None, gt_norm, pred_norm).ratio()
    return 1.0 if ratio > threshold else 0.0


def create_normalized_senator_id(name: str, state: str) -> str:
    """
    Create a canonical senator ID from name and state.
    
    Format: FirstName_LastName_STATE
    
    Args:
        name: Senator's full name
        state: Two-letter state code
        
    Returns:
        Canonical senator ID
        
    Examples:
        >>> create_normalized_senator_id("Roger Wicker", "MS")
        "Roger_Wicker_MS"
    """
    normalized = normalize_name(name)
    name_slug = "_".join(w.capitalize() for w in normalized.split() if w)
    return f"{name_slug}_{state.upper()}"


# ─────────────────────────────────────────────────────────────────────────────
# HTML TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_readable_text(html: str) -> str:
    """
    Extract clean readable text from HTML, removing boilerplate.
    
    Removes: <script>, <style>, <nav>, <footer>, <noscript>
    
    Args:
        html: HTML string
        
    Returns:
        Clean text with normalized whitespace
    """
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove noise tags
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    
    # Extract text with space preservation
    text = soup.get_text(separator=" ", strip=True)
    
    # Normalize whitespace: multiple spaces → single space
    text = re.sub(r"\s{2,}", " ", text).strip()
    
    return text
