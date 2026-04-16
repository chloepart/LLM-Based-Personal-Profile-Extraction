"""
Enhanced baseline extractors for comprehensive senator profile extraction.

Implements regex, keyword search, spaCy NER, and BERT-based NER baselines
with field-specific extraction strategies for 6 senator profile fields:
  - name (Regex, Keyword)
  - gender (Regex)
  - birthdate (Regex)
  - education (Keyword, spaCy, BERT)
  - religion (Keyword, spaCy, BERT)
  - committee_roles (Keyword, spaCy, BERT)

All baselines ingest raw HTML and return structured predictions in a unified schema.
Religion values are mapped to canonical forms via RELIGION_HIERARCHY from config.
Scoring logic is delegated to evaluator.py for consistent hierarchical matching.

Architecture:
  1. HTMLProcessor.extract_readable_text() → plain text from HTML
  2. Each baseline processes text to extract fields
  3. All outputs normalized to unified schema (see UnifiedExtraction class)
  4. Runner script batches extraction over senate_html/ directory
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup
import pandas as pd

# Import core utilities
from modules.config.config import REGEX_PATTERNS, RELIGION_HIERARCHY
from modules.data.html import HTMLProcessor
from modules.utils.parsing import EducationParser, DegreeNormalizer
from modules.standardizer import SenatorDataStandardizer


# ============================================================================
# UNIFIED EXTRACTION SCHEMA
# ============================================================================

@dataclass
class UnifiedExtraction:
    """
    Unified output schema for all baseline extractors.
    
    All baselines return this dataclass to ensure consistent format for
    downstream comparison and scoring.
    """
    baseline: str  # e.g., "regex", "keyword", "spacy", "bert"
    senator_id: Optional[str] = None  # e.g., "Bernie_Sanders_VT"
    name: Optional[str] = None
    gender: Optional[str] = None  # "Male" or "Female" or None
    birthdate: Optional[str] = None  # YYYY-MM-DD format or None
    education: Optional[List[Dict[str, Any]]] = None  # [{"degree": str, "institution": str, "year": int}]
    religion: Optional[str] = None  # Canonical form from RELIGION_HIERARCHY
    committee_roles: Optional[List[str]] = None  # ["Committee Name", "Committee Name"]
    
    def to_dict(self):
        """Convert to dict for CSV export."""
        result = asdict(self)
        # Serialize complex fields
        result['education'] = str(result['education']) if result['education'] else None
        result['committee_roles'] = '|'.join(result['committee_roles']) if result['committee_roles'] else None
        return result


# ============================================================================
# SUPPORT FUNCTIONS
# ============================================================================

def extract_html_text(html: str, max_length: int = 10000) -> str:
    """
    Extract readable text from raw HTML using HTMLProcessor.
    
    Args:
        html: Raw HTML string
        max_length: Truncate to this many characters (for spaCy/BERT token limits)
    
    Returns:
        Cleaned plain text
    """
    return HTMLProcessor.extract_readable_text(html, max_length=max_length)


def normalize_degree(degree_str: str) -> str:
    """
    Normalize degree abbreviations to canonical form without periods.
    
    Examples:
        "B.A." → "BA"
        "Ph.D." → "PhD"
        "M.B.A." → "MBA"
        "J.D." → "JD"
        "BA" → "BA" (already normalized)
    """
    if not degree_str:
        return degree_str
    
    # Mapping of period-based formats to normalized formats
    mappings = {
        "b.a.": "BA", "b.s.": "BS", "b.e.": "BE",
        "m.a.": "MA", "m.s.": "MS", "m.b.a.": "MBA",
        "j.d.": "JD", "l.l.b.": "LLB", "l.l.m.": "LLM",
        "m.d.": "MD", "ph.d.": "PhD", "ed.d.": "EdD",
        "d.d.s.": "DDS", "d.v.m.": "DVM",
    }
    
    normalized = degree_str.strip().lower()
    if normalized in mappings:
        return mappings[normalized]
    
    # Remove periods if not in mapping (fallback)
    return degree_str.strip().replace(".", "").upper()


def standardize_religion(value: Optional[str], religion_hierarchy: Dict[str, str]) -> Optional[str]:
    """
    Standardize extracted religion value to canonical hierarchy form.
    
    Args:
        value: Extracted religion string (may be variant spelling/capitalization)
        religion_hierarchy: Mapping from variants to canonical forms (from config)
    
    Returns:
        Canonical religion form or None if not in hierarchy
    
    Example:
        standardize_religion("baptist", RELIGION_HIERARCHY) → "christian"
        standardize_religion("jewish", RELIGION_HIERARCHY) → "jewish"
    """
    if not value or not isinstance(value, str):
        return None
    
    # Normalize to lowercase for lookup
    normalized_value = value.strip().lower()
    
    # Direct lookup in hierarchy
    if normalized_value in religion_hierarchy:
        return religion_hierarchy[normalized_value]
    
    # Attempt substring match (for multi-word variants like "united methodist")
    for key, canonical in religion_hierarchy.items():
        if key in normalized_value or normalized_value in key:
            return canonical
    
    return None


# ============================================================================
# BASELINE 1: REGEX EXTRACTION
# ============================================================================

class RegexBaseline:
    """
    Regex-based extraction for senator names, gender, and birthdates.
    
    Fields extracted:
      - name: Senator names via pattern "Senator FirstName LastName"
      - gender: Pronouns and explicit gender references (he/him/his vs she/her/hers)
      - birthdate: Dates in "born Month DD, YYYY" or similar patterns
    
    Operates on plain text (HTML pre-processed).
    """
    
    def __init__(self, regex_patterns: Optional[Dict[str, Any]] = None):
        """
        Initialize with regex patterns.
        
        Args:
            regex_patterns: Dict of compiled regex patterns (from config.REGEX_PATTERNS)
                           If None, uses config.REGEX_PATTERNS
        """
        self.patterns = regex_patterns or REGEX_PATTERNS
        
        # Additional patterns not in config (for gender, birthdate)
        self.gender_male_re = re.compile(
            r'\b(he|him|his|male|senator\s+[^,]*?\s+\(R-|born.*?\).*?served|his career)\b', re.IGNORECASE
        )
        self.gender_female_re = re.compile(
            r'\b(she|her|hers|female|senator\s+[^,]*?\s+\(D-|her career)\b', re.IGNORECASE
        )
        # Birthdate patterns:
        # 1. "born January 15, 1965"
        # 2. "born in 1965"
        # 3. "Born 1965"
        self.birthdate_re = re.compile(
            r'born\s+(?:in\s+)?([A-Z][a-z]*\s+)?(\d{1,2})?[,\s]*(\d{4})',
            re.IGNORECASE
        )
    
    def extract(self, text: str, senator_id: Optional[str] = None) -> UnifiedExtraction:
        """
        Extract fields using regex patterns.
        
        Args:
            text: Plain text (pre-processed from HTML)
            senator_id: Optional senator identifier
        
        Returns:
            UnifiedExtraction with extracted fields
        """
        result = UnifiedExtraction(
            baseline="regex",
            senator_id=senator_id,
        )
        
        # Extract name
        name_re = self.patterns.get('NAME_RE')
        if name_re:
            match = name_re.search(text)
            if match:
                result.name = match.group(1).strip()
        
        # Extract gender (simple heuristic: first match wins)
        male_matches = self.gender_male_re.findall(text[:2000])  # First 2000 chars
        female_matches = self.gender_female_re.findall(text[:2000])
        
        if len(male_matches) > len(female_matches):
            result.gender = "Male"
        elif len(female_matches) > 0:
            result.gender = "Female"
        
        # Extract birthdate
        birthdate_match = self.birthdate_re.search(text)
        if birthdate_match:
            month_str, day_str, year_str = birthdate_match.groups()
            try:
                from datetime import datetime
                
                # Parse month if present (e.g., "January"), else default to None
                month_num = None
                if month_str and month_str.strip():
                    try:
                        month_num = datetime.strptime(month_str.strip(), "%B").month
                    except Exception:
                        month_num = None
                
                # Format date
                if month_num and day_str:
                    # Full date: YYYY-MM-DD
                    result.birthdate = f"{year_str}-{month_num:02d}-{day_str.zfill(2)}"
                elif month_num:
                    # Partial date: YYYY-MM-01 (month without day)
                    result.birthdate = f"{year_str}-{month_num:02d}-01"
                else:
                    # Year only: YYYY-01-01 (as placeholder)
                    result.birthdate = f"{year_str}-01-01"
            except Exception:
                result.birthdate = None
        
        return result


# ============================================================================
# BASELINE 2: KEYWORD SEARCH EXTRACTION
# ============================================================================

# Enhanced keyword map for senator bios (extends default)
SENATOR_KEYWORD_MAP = {
    "name": ["senator", "name"],
    "education": ["education", "degree", "university", "college", "school", "studied",
                  "graduate", "undergraduate", "alma mater", "attended"],
    "religion": ["religion", "faith", "church", "denomination", "jewish", "catholic",
                 "baptist", "methodist", "presbyterian", "episcopal", "mormon", "atheist",
                 "agnostic", "unitarian", "evangelical", "pentecostal", "orthodox"],
    "committee_roles": ["committee", "subcommittee", "member", "ranking member", "chair",
                       "caucus", "congressional"],
}


class KeywordSearchBaseline:
    """
    Keyword-search extraction for education, religion, and committee roles.
    
    Strategy:
      1. Parse HTML structure: identify heading/section elements
      2. Extract keywords from headings, match with content sections
      3. Return text following matched headings as extracted values
    
    Falls back to plain-text label search when HTML structure unavailable.
    """
    
    def __init__(self, keyword_map: Optional[Dict[str, List[str]]] = None):
        """
        Args:
            keyword_map: Dict mapping field name → list of trigger keywords
                        Defaults to SENATOR_KEYWORD_MAP
        """
        self.keyword_map = keyword_map or SENATOR_KEYWORD_MAP
    
    def _search_html(self, html: str, field: str, keywords: List[str]) -> Optional[str]:
        """
        Look for heading/title element containing keywords, return following text.
        """
        soup = BeautifulSoup(html, "html.parser")
        heading_tags = ["h1", "h2", "h3", "h4", "h5", "h6", "dt", "th", "label", "strong", "b"]
        
        for tag in soup.find_all(heading_tags):
            tag_text = tag.get_text(separator=" ", strip=True).lower()
            if any(kw in tag_text for kw in keywords):
                # Extract text from next sibling
                sibling = tag.find_next_sibling()
                if sibling:
                    return sibling.get_text(separator=" ", strip=True)
                # Or next non-empty line in parent
                parent_sib = tag.parent.find_next_sibling() if tag.parent else None
                if parent_sib:
                    return parent_sib.get_text(separator=" ", strip=True)
        
        return None
    
    def _search_plaintext(self, text: str, keywords: List[str]) -> Optional[str]:
        """
        Scan plain text for label patterns (e.g., "Education: ...") or keyword headers.
        """
        lines = text.splitlines()
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            for kw in keywords:
                # Pattern 1: "Keyword: value"
                if re.match(rf"^{re.escape(kw)}\s*[:：]", line_lower):
                    after_colon = re.split(r"[:：]", line, maxsplit=1)
                    if len(after_colon) > 1 and after_colon[1].strip():
                        return after_colon[1].strip()
                    # Pattern 2: "Keyword" on current line, value on next
                    for j in range(i + 1, min(i + 4, len(lines))):
                        if lines[j].strip():
                            return lines[j].strip()
                
                # Pattern 2: Keyword as section header (at line start)
                if line_lower.startswith(kw):
                    # Return next non-empty line
                    for j in range(i + 1, min(i + 4, len(lines))):
                        if lines[j].strip():
                            return lines[j].strip()
        
        return None
    
    def extract(self, html: str, senator_id: Optional[str] = None) -> UnifiedExtraction:
        """
        Extract fields using keyword search on HTML structure and plain text.
        
        Args:
            html: Raw HTML string
            senator_id: Optional senator identifier
        
        Returns:
            UnifiedExtraction with name, education, religion, committee_roles
        """
        result = UnifiedExtraction(
            baseline="keyword",
            senator_id=senator_id,
        )
        
        # Detect if input is HTML or plain text
        is_html = bool(re.search(r"<[a-zA-Z][\s\S]*?>", html))
        text_fallback = extract_html_text(html) if is_html else html
        
        # Extract name: from plain text (usually first full sentence with senator title)
        name_match = re.search(r"Senator\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s", text_fallback)
        if name_match:
            result.name = name_match.group(1).strip()
        
        # Extract education: look for degree patterns and nearby institutions
        education = self._extract_degrees_from_text(text_fallback)
        if education:
            result.education = education
        
        # Extract religion: search for religion keywords in biography section (first 3000 chars)
        # Avoid false positives from office addresses
        bio_section = text_fallback[:3000]
        religion_keywords_lower = [kw.lower() for kw in self.keyword_map.get("religion", [])]
        for kw in religion_keywords_lower:
            # Stricter matching: look for religion keywords but exclude "Church" in "Church St"
            if kw == "church":
                # Skip if "Church St" appears nearby (office address)
                if "Church St" in text_fallback:
                    continue
            if kw in bio_section.lower():
                # Found a religion keyword in bio section
                religion = standardize_religion(kw, RELIGION_HIERARCHY)
                if religion:
                    result.religion = religion
                    break
        
        # Extract committee roles: look for committee patterns
        # Pattern: "Ranking Member of", "Member of", "serves on", followed by committee name
        committee_pattern = r"(?:Ranking Member of|Member of|serves on|on)\s+(?:the\s+)?(?:Senate\s+)?([A-Z][a-z\s&,]*(?:Committee|Subcommittee))"
        committee_matches = re.finditer(committee_pattern, text_fallback)
        committees = [m.group(1).strip() for m in committee_matches]
        if committees:
            result.committee_roles = list(set(committees))
        
        return result
    
    def _extract_degrees_from_text(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """Extract degree entries from education text."""
        degree_pattern = r"(B\.A\.|B\.S\.|M\.A\.|M\.S\.|M\.B\.A\.|M\.D\.|Ph\.D\.|J\.D\.|D\.D\.S\.|LL\.B\.)"
        matches = re.finditer(degree_pattern, text, re.IGNORECASE)
        
        education = []
        seen = set()
        
        for match in matches:
            degree = normalize_degree(match.group(1))
            degree_pos = match.start()
            
            # Look for institution name: search backwards first (institution usually before degree)
            search_start = max(0, degree_pos - 200)
            search_end = min(len(text), degree_pos + 100)
            context_before = text[search_start:degree_pos]
            context_after = text[degree_pos:search_end]
            
            institution = None
            
            # Pattern 1: Look for "University of X", "X University", "X College" (excludes "High School")
            inst_pattern = r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:University|College|Institute|Academy|School of))"
            
            # First try backwards
            inst_matches_before = list(re.finditer(inst_pattern, context_before))
            if inst_matches_before:
                institution = inst_matches_before[-1].group(1).strip()  # Use last (closest to degree)
            
            # If not found backwards, try forwards
            if not institution:
                inst_match_after = re.search(inst_pattern, context_after)
                if inst_match_after:
                    institution = inst_match_after.group(1).strip()
            
            # Extract year from context
            year_pattern = r"(\d{4})"
            year_match_before = re.search(year_pattern, context_before)
            year_match_after = re.search(year_pattern, context_after)
            
            year = None
            if year_match_after:
                year = int(year_match_after.group(1))
            elif year_match_before:
                year = int(year_match_before.group(1))
            
            # Avoid duplicates
            entry_key = (degree, institution, year)
            if entry_key not in seen:
                education.append({
                    "degree": degree,
                    "institution": institution,
                    "year": year
                })
                seen.add(entry_key)
        
        return education if education else None
    
    def _extract_religion_from_text(self, text: str) -> Optional[str]:
        """Extract religion denomination from text (deprecated - integrated into extract())."""
        return None
    
    def _extract_committees_from_text(self, text: str) -> Optional[List[str]]:
        """Extract committee names from text (deprecated - integrated into extract())."""
        return None


# ============================================================================
# BASELINE 3: SPACY NER EXTRACTION
# ============================================================================

class SpaCyBaseline:
    """
    spaCy NER baseline for education, religion, and committee roles.
    
    Uses named entity recognition to identify:
      - PERSON entities (for name fallback)
      - ORG entities (institutions, organizations, committees)
      - DATE entities (graduation years)
    
    Domain-specific heuristics:
      - ORG + "Committee" keyword → committee role
      - ORG + education context → institution name
      - DATE entities → year extraction
    """
    
    def __init__(self, nlp_model=None):
        """
        Args:
            nlp_model: Loaded spaCy model. If None, loads en_core_web_sm.
        """
        if nlp_model is None:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise RuntimeError(
                    "spaCy model 'en_core_web_sm' not found. "
                    "Install with: python -m spacy download en_core_web_sm"
                )
        else:
            self.nlp = nlp_model
    
    def extract(self, text: str, senator_id: Optional[str] = None) -> UnifiedExtraction:
        """
        Extract fields using spaCy NER with domain heuristics.
        
        Args:
            text: Plain text (pre-processed from HTML)
            senator_id: Optional senator identifier
        
        Returns:
            UnifiedExtraction with education, religion, committee_roles
        """
        result = UnifiedExtraction(
            baseline="spacy",
            senator_id=senator_id,
        )
        
        # Process text with spaCy (limit to 10000 chars for token limits)
        doc = self.nlp(text[:10000])
        
        # Collect entities by type
        persons = [e.text for e in doc.ents if e.label_ == "PERSON"]
        orgs = [e.text for e in doc.ents if e.label_ == "ORG"]
        dates = [e.text for e in doc.ents if e.label_ == "DATE"]
        
        # Extract education (degree + institution + year)
        result.education = self._extract_education(text, orgs, dates)
        
        # Extract religion (heuristic: look for religion-related ORGs or keywords)
        result.religion = self._extract_religion(text, orgs)
        
        # Extract committee roles (ORG entities with "Committee" keyword)
        result.committee_roles = self._extract_committee_roles(orgs)
        
        return result
    
    def _extract_education(self, text: str, orgs: List[str], dates: List[str]) -> Optional[List[Dict[str, Any]]]:
        """Extract education entries using degree patterns and ORG entities."""
        degree_pattern = r"(B\.A\.|B\.S\.|M\.A\.|M\.S\.|M\.B\.A\.|M\.D\.|Ph\.D\.|J\.D\.|D\.D\.S\.|LL\.B\.)"
        matches = list(re.finditer(degree_pattern, text, re.IGNORECASE))
        
        education = []
        for match in matches:
            degree = normalize_degree(match.group(1))
            degree_pos = match.start()
            
            # Look for institution (ORG entity) within ±200 chars
            search_start = max(0, degree_pos - 200)
            search_end = min(len(text), degree_pos + 200)
            search_region = text[search_start:search_end]
            
            institution = None
            for org in orgs:
                if org.lower() in search_region.lower():
                    institution = org
                    break
            
            # Look for year in search region
            year_pattern = r"(19|20)\d{2}"
            year_match = re.search(year_pattern, search_region)
            year = year_match.group(0) if year_match else None
            
            education.append({
                "degree": degree,
                "institution": institution,
                "year": int(year) if year else None
            })
        
        return education if education else None
    
    def _extract_religion(self, text: str, orgs: List[str]) -> Optional[str]:
        """Extract religion using keyword matching in text and ORGs."""
        religion_keywords = ["jewish", "catholic", "baptist", "methodist", "presbyterian",
                            "episcopal", "lutheran", "evangelical", "pentecostal", "orthodox",
                            "mormon", "lds", "muslim", "atheist", "agnostic", "unitarian"]
        
        # Search for religion keywords in text
        for kw in religion_keywords:
            if kw in text.lower():
                return standardize_religion(kw, RELIGION_HIERARCHY)
        
        # Also check ORG entities (some may contain religious organization names)
        for org in orgs:
            for kw in religion_keywords:
                if kw in org.lower():
                    return standardize_religion(kw, RELIGION_HIERARCHY)
        
        return None
    
    def _extract_committee_roles(self, orgs: List[str]) -> Optional[List[str]]:
        """Extract committee roles from ORG entities containing 'Committee'."""
        committee_keywords = ["committee", "subcommittee", "caucus"]
        committees = [org for org in orgs 
                     if any(kw in org.lower() for kw in committee_keywords)]
        
        return list(set(committees)) if committees else None


# ============================================================================
# BASELINE 4: BERT NER EXTRACTION
# ============================================================================

class BERTBaseline:
    """
    BERT-based NER baseline for education, religion, and committee roles.
    
    Uses 'dslim/bert-base-NER' (or TextWash if available) for fine-grained
    entity detection. Post-processes NER outputs with domain heuristics
    to extract structured education, religion, and committee info.
    """
    
    DEFAULT_MODEL = "dslim/bert-base-NER"
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Args:
            model_name: HuggingFace model name or local path.
                       Defaults to 'dslim/bert-base-NER'.
        """
        try:
            from transformers import pipeline
        except ImportError:
            raise RuntimeError(
                "transformers is required for BERTBaseline. "
                "Install with: pip install transformers torch"
            )
        
        self.model_name = model_name or self.DEFAULT_MODEL
        self._pipe = pipeline(
            "ner",
            model=self.model_name,
            aggregation_strategy="simple",
        )
    
    def extract(self, text: str, senator_id: Optional[str] = None) -> UnifiedExtraction:
        """
        Extract fields using BERT NER.
        
        Args:
            text: Plain text (pre-processed from HTML)
            senator_id: Optional senator identifier
        
        Returns:
            UnifiedExtraction with education, religion, committee_roles
        """
        result = UnifiedExtraction(
            baseline="bert",
            senator_id=senator_id,
        )
        
        # Run NER pipeline (truncated to ~2000 chars ≈ 512 tokens)
        entities = self._pipe(text[:2000])
        
        # Aggregate entities by type
        persons = self._aggregate(entities, "PER")
        orgs = self._aggregate(entities, "ORG")
        locations = self._aggregate(entities, "LOC")
        misc = self._aggregate(entities, "MISC")
        
        # Extract education (heuristic: ORG + degree keywords)
        result.education = self._extract_education_from_entities(text, orgs)
        
        # Extract religion (heuristic: MISC + religion keywords)
        result.religion = self._extract_religion_from_entities(text, orgs, misc)
        
        # Extract committee roles (ORG + "Committee" keyword)
        result.committee_roles = self._extract_committees_from_entities(orgs)
        
        return result
    
    def _aggregate(self, entities: List[Dict], label: str) -> List[str]:
        """Return deduplicated text spans for a given NER label."""
        return list({e["word"] for e in entities if e["entity_group"] == label})
    
    def _extract_education_from_entities(self, text: str, orgs: List[str]) -> Optional[List[Dict[str, Any]]]:
        """Extract education using ORG entities as institutions."""
        degree_pattern = r"(B\.A\.|B\.S\.|M\.A\.|M\.S\.|M\.B\.A\.|M\.D\.|Ph\.D\.|J\.D\.|D\.D\.S\.|LL\.B\.)"
        matches = list(re.finditer(degree_pattern, text, re.IGNORECASE))
        
        education = []
        for match in matches:
            degree = normalize_degree(match.group(1))
            degree_pos = match.start()
            
            # Look for ORG within ±200 chars
            search_start = max(0, degree_pos - 200)
            search_end = min(len(text), degree_pos + 200)
            search_region = text[search_start:search_end]
            
            institution = None
            for org in orgs:
                if org.lower() in search_region.lower():
                    institution = org
                    break
            
            year_pattern = r"(19|20)\d{2}"
            year_match = re.search(year_pattern, search_region)
            year = int(year_match.group(0)) if year_match else None
            
            education.append({
                "degree": degree,
                "institution": institution,
                "year": year
            })
        
        return education if education else None
    
    def _extract_religion_from_entities(self, text: str, orgs: List[str], misc: List[str]) -> Optional[str]:
        """Extract religion using MISC entities and keyword heuristics."""
        religion_keywords = ["jewish", "catholic", "baptist", "methodist", "presbyterian",
                            "episcopal", "lutheran", "evangelical", "pentecostal", "orthodox",
                            "mormon", "lds", "muslim", "atheist", "agnostic", "unitarian"]
        
        # Search MISC and ORG entities for religion keywords
        all_entities = misc + orgs
        for entity in all_entities:
            entity_lower = entity.lower()
            for kw in religion_keywords:
                if kw in entity_lower:
                    return standardize_religion(kw, RELIGION_HIERARCHY)
        
        # Fallback: search raw text for religion keywords
        for kw in religion_keywords:
            if kw in text.lower():
                return standardize_religion(kw, RELIGION_HIERARCHY)
        
        return None
    
    def _extract_committees_from_entities(self, orgs: List[str]) -> Optional[List[str]]:
        """Extract committee roles from ORG entities."""
        committee_keywords = ["committee", "subcommittee", "caucus"]
        committees = [org for org in orgs 
                     if any(kw in org.lower() for kw in committee_keywords)]
        
        return list(set(committees)) if committees else None


# ============================================================================
# CONVENIENCE WRAPPER FUNCTIONS
# ============================================================================

def extract_via_regex(html: str, senator_id: Optional[str] = None) -> UnifiedExtraction:
    """Convenience wrapper for RegexBaseline."""
    text = extract_html_text(html)
    return RegexBaseline().extract(text, senator_id=senator_id)


def extract_via_keyword(html: str, senator_id: Optional[str] = None) -> UnifiedExtraction:
    """Convenience wrapper for KeywordSearchBaseline."""
    return KeywordSearchBaseline().extract(html, senator_id=senator_id)


def extract_via_spacy(html: str, senator_id: Optional[str] = None) -> UnifiedExtraction:
    """Convenience wrapper for SpaCyBaseline."""
    text = extract_html_text(html)
    return SpaCyBaseline().extract(text, senator_id=senator_id)


def extract_via_bert(html: str, senator_id: Optional[str] = None) -> UnifiedExtraction:
    """Convenience wrapper for BERTBaseline."""
    text = extract_html_text(html)
    return BERTBaseline().extract(text, senator_id=senator_id)


# ============================================================================
__all__ = [
    # Classes
    "UnifiedExtraction",
    "RegexBaseline",
    "KeywordSearchBaseline",
    "SpaCyBaseline",
    "BERTBaseline",
    # Support functions
    "extract_html_text",
    "normalize_degree",
    "standardize_religion",
    # Convenience functions
    "extract_via_regex",
    "extract_via_keyword",
    "extract_via_spacy",
    "extract_via_bert",
    # Constants
    "SENATOR_KEYWORD_MAP",
]
