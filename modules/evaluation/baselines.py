"""
Traditional baseline extractors for comparison with LLM approaches.

Implements regex, keyword search, spaCy NER, and BERT-based NER baselines
as described in Liu et al. (Section 6.1.3).

Note on mlscraper: Liu et al. include mlscraper as a fifth baseline, but it
requires structured HTML and a training phase with labeled examples, making
it incompatible with plain-text senator bios. It is excluded here and should
be noted as a deviation from the original paper in the methodology section.
"""

import re
from bs4 import BeautifulSoup
import spacy


# ---------------------------------------------------------------------------
# U.S. State Names and Abbreviations
# ---------------------------------------------------------------------------

US_STATES = frozenset({
    # Full state names
    "Alabama", "Alaska", "Arizona", "Arkansas", "California",
    "Colorado", "Connecticut", "Delaware", "Florida", "Georgia",
    "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
    "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri",
    "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
    "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
    "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
    "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
    # Two-letter abbreviations
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
})


# ---------------------------------------------------------------------------
# Baseline 1: Regex
# ---------------------------------------------------------------------------

class RegexBaseline:
    """
    Regex-based extraction baseline (Liu et al. Section 6.1.3).

    Domain note: 'party' and 'years_found' are senator-specific adaptations
    not present in Liu et al.'s original 8-category schema.
    """

    def __init__(self, regex_patterns, name_re=None, party_re=None,
                 email_re=None, phone_re=None):
        """
        Initialize with compiled regex patterns.

        Args:
            regex_patterns: Dict with pattern keys or individual patterns.
            name_re:   Compiled regex for name extraction.
            party_re:  Compiled regex for party/affiliation extraction.
            email_re:  Compiled regex for email extraction.
            phone_re:  Compiled regex for phone extraction.
        """
        if isinstance(regex_patterns, dict):
            self.name_re  = regex_patterns.get('NAME_RE')          or name_re
            self.party_re = regex_patterns.get('PARTY_KEYWORD_RE') or party_re
            self.email_re = regex_patterns.get('EMAIL_RE')         or email_re
            self.phone_re = regex_patterns.get('PHONE_RE')         or phone_re
        else:
            self.name_re  = name_re
            self.party_re = party_re
            self.email_re = email_re
            self.phone_re = phone_re

    def extract(self, text):
        """
        Extract structured fields using regex patterns.

        Args:
            text: Input text to extract from.

        Returns:
            Dict with keys: full_name, party, email, phone.
        """
        name_m  = self.name_re.search(text)  if self.name_re  else None
        party_m = self.party_re.search(text) if self.party_re else None

        return {
            "full_name":   name_m.group(1)              if name_m  else None,
            "party":       party_m.group(0).title()     if party_m else None,
            "email":       self.email_re.findall(text)  if self.email_re else None,
            "phone":       self.phone_re.findall(text)  if self.phone_re else None,
        }


# ---------------------------------------------------------------------------
# Baseline 2: Keyword Search
# ---------------------------------------------------------------------------

# Default keyword map for senator bios.
# Keys are the output field names; values are the trigger keywords to search
# for in HTML title/heading elements or plain-text labels.
DEFAULT_KEYWORD_MAP = {
    "name":       ["name"],
    "email":      ["email", "contact", "e-mail"],
    "phone":      ["phone", "telephone", "tel", "fax"],
    "address":    ["address", "office", "location", "mailing"],
    "work":       ["work", "experience", "career", "employment", "position",
                   "served", "service"],
    "education":  ["education", "degree", "university", "college", "school",
                   "studied", "graduate", "undergraduate"],
    "affiliation":["affiliation", "party", "caucus", "committee", "member"],
    "occupation": ["occupation", "profession", "role", "job", "senator",
                   "representative"],
    "religion":   ["religion", "faith", "church", "denomination"],
    "state":      ["state", "represents", "district"],
}


class KeywordSearchBaseline:
    """
    Keyword-search extraction baseline (Liu et al. Section 6.1.3).

    Strategy (mirroring the paper):
      1. Parse HTML with BeautifulSoup to identify title/heading elements
         whose text contains a known keyword.
      2. Extract the text that immediately follows the matched heading.
      3. Fall back to a plain-text label search when no HTML structure is
         found (covers pre-parsed senator bio strings).
    """

    def __init__(self, keyword_map=None):
        """
        Args:
            keyword_map: Dict mapping field name -> list of trigger keywords.
                         Defaults to DEFAULT_KEYWORD_MAP.
        """
        self.keyword_map = keyword_map or DEFAULT_KEYWORD_MAP

    # ------------------------------------------------------------------
    def _search_html(self, html, field, keywords):
        """
        Look for a heading/title element whose text contains one of the
        keywords, then return the text content of the next sibling element.
        """
        soup = BeautifulSoup(html, "html.parser")
        heading_tags = ["h1", "h2", "h3", "h4", "h5", "h6",
                        "dt", "th", "label", "strong", "b", "li"]

        for tag in soup.find_all(heading_tags):
            tag_text = tag.get_text(separator=" ", strip=True).lower()
            if any(kw in tag_text for kw in keywords):
                # Try next sibling first, then parent's next sibling
                sibling = tag.find_next_sibling()
                if sibling:
                    return sibling.get_text(separator=" ", strip=True)
                parent_sib = tag.parent.find_next_sibling() if tag.parent else None
                if parent_sib:
                    return parent_sib.get_text(separator=" ", strip=True)
        return None

    def _search_plaintext(self, text, keywords):
        """
        Scan plain text line by line; if a line contains a keyword acting as
        a label (keyword + ':' or keyword at line start), return the text on
        that line after the colon, or the next non-empty line.
        """
        lines = text.splitlines()
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            for kw in keywords:
                # Label pattern:  "Education: ..." or "Education\n..."
                if re.match(rf"^{re.escape(kw)}\s*[:：]", line_lower):
                    after_colon = re.split(r"[:：]", line, maxsplit=1)
                    if len(after_colon) > 1 and after_colon[1].strip():
                        return after_colon[1].strip()
                    # value is on the next line
                    for j in range(i + 1, min(i + 4, len(lines))):
                        if lines[j].strip():
                            return lines[j].strip()
        return None

    # ------------------------------------------------------------------
    def extract(self, text):
        """
        Extract fields using keyword search.

        Args:
            text: Raw HTML source or plain text of a senator bio.

        Returns:
            Dict mapping each field name to the extracted string (or None).
        """
        is_html = bool(re.search(r"<[a-zA-Z][\s\S]*?>", text))
        results = {}

        for field, keywords in self.keyword_map.items():
            if is_html:
                value = self._search_html(text, field, keywords)
            else:
                value = self._search_plaintext(text, keywords)
            results[field] = value

        return results


# ---------------------------------------------------------------------------
# Baseline 3: spaCy NER with semantic PIE mapping
# ---------------------------------------------------------------------------

class SpaCyBaseline:
    """
    spaCy NER baseline (Liu et al. Section 6.1.3) with senator-specific PIE mapping.
    
    vanilla=True replicates Liu et al. Section 6.1.3 naive NER mapping for baseline
    comparison; vanilla=False (default) uses senator-specific semantic mapping for
    domain analysis.
    
    Improvement (vanilla=False): Maps raw NER entities (PERSON, ORG, GPE, DATE) to
    PIE fields using domain heuristics. This makes output directly comparable to
    TextWash and BERT baselines.
    
    Domain adaptations (senator bios):
      - GPE → state assignment
      - First PERSON → name extraction
      - DATE entities → year extraction for work/education timeline
      - ORG entities + text patterns → education (degree + institution)
      - ORG entities + "Committee" keyword → committee roles
    """

    def __init__(self, nlp_model=None):
        """
        Args:
            nlp_model: Loaded spaCy model. If None, loads en_core_web_sm.
        """
        if nlp_model is None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise RuntimeError(
                    "spaCy model 'en_core_web_sm' not found. "
                    "Install with: python -m spacy download en_core_web_sm"
                )
        else:
            self.nlp = nlp_model

    def _extract_years_from_dates(self, date_entities):
        """
        Extract years from DATE entity text (for work/education timeline).
        
        Parses text like "1995", "1995-2000", "January 2005", etc.
        
        Args:
            date_entities: List of DATE entity texts from spaCy.
            
        Returns:
            List of extracted year strings.
        """
        years = []
        year_pattern = r"(19|20)\d{2}"
        
        for date_text in date_entities:
            matches = re.findall(year_pattern, date_text)
            if matches:
                years.extend(matches)
        
        return years[:5]  # Limit to 5 years

    def _extract_education(self, text, org_entities):
        """
        Extract education entries (degree + institution + year).
        
        Strategy:
          1. Look for degree keywords (B.A., M.D., Ph.D., J.D., etc.)
          2. Associate with nearby ORG entities (institutions)
          3. Extract graduation year if available
        
        Args:
            text: Original text for pattern matching.
            org_entities: List of ORG entities from spaCy.
            
        Returns:
            List of dicts with keys: degree, institution, year
        """
        education = []
        
        # Degree patterns (common U.S. degrees)
        degree_pattern = r"(B\.A\.|B\.S\.|M\.A\.|M\.S\.|M\.B\.A\.|M\.D\.|Ph\.D\.|J\.D\.|D\.D\.S\.|D\.V\.M\.|LL\.B\.|Bachelor|Master|Doctorate|Ph\.D)"
        degree_matches = re.finditer(degree_pattern, text, re.IGNORECASE)
        
        for degree_match in degree_matches:
            degree_text = degree_match.group(1)
            start_pos = degree_match.start()
            
            # Look for institution (ORG entity) near this degree
            # Search within ±200 characters
            search_start = max(0, start_pos - 200)
            search_end = min(len(text), start_pos + 200)
            search_region = text[search_start:search_end]
            
            # Find ORG entities in this region
            # Approximate: find capitalized phrases that might be institution names
            institution = None
            for org in org_entities:
                # Check if org appears in search region (with some tolerance for variations)
                if org.lower() in search_region.lower():
                    institution = org
                    break
            
            # Extract year if present nearby
            year = None
            year_pattern_local = r"(19|20)\d{2}"
            year_in_region = re.search(year_pattern_local, search_region)
            if year_in_region:
                year = year_in_region.group(0)
            
            education.append({
                "degree": degree_text,
                "institution": institution,
                "year": year
            })
        
        return education if education else None

    def _extract_committee_roles(self, text, org_entities):
        """
        Extract committee roles and memberships.
        
        Strategy:
          1. Look for ORG entities containing "Committee"
          2. Extract full committee name
          3. Look for role keywords (Chair, Ranking Member, Member)
        
        Args:
            text: Original text for pattern matching.
            org_entities: List of ORG entities from spaCy.
            
        Returns:
            List of committee role strings.
        """
        committee_roles = []
        
        # Filter ORGs that contain "Committee" or similar keywords
        committee_keywords = ["committee", "subcommittee", "caucus"]
        committees = [org for org in org_entities 
                     if any(kw in org.lower() for kw in committee_keywords)]
        
        if not committees:
            return None
        
        # For each committee, try to find associated role
        role_pattern = r"(Chair|Ranking Member|Member|Co-Chair)"
        
        for committee in committees:
            # Look for role keywords near committee name in text
            committee_pos = text.lower().find(committee.lower())
            if committee_pos != -1:
                # Search within window of 150 chars before/after committee mention
                search_start = max(0, committee_pos - 150)
                search_end = min(len(text), committee_pos + len(committee) + 150)
                context = text[search_start:search_end]
                
                role_match = re.search(role_pattern, context, re.IGNORECASE)
                if role_match:
                    role = role_match.group(1)
                    committee_roles.append(f"{role} of {committee}")
                else:
                    committee_roles.append(f"Member of {committee}")
            else:
                committee_roles.append(committee)
        
        return committee_roles if committee_roles else None

    def extract(self, text):
        """
        Extract entities using spaCy with domain-specific mapping for senator bios.

        Args:
            text: Input text to extract from.

        Returns:
            Dict mapping field names to extracted values:
              - name: First detected PERSON entity
              - state: First detected GPE (U.S. state)
              - affiliation: First detected ORG entity (excluding committee/subcommittee/caucus)
              - occupation: None (NER-based extraction does not infer occupation)
              - education: List of dicts with degree, institution, year
              - committee_roles: List of committee role strings
        """
        doc = self.nlp(text[:10000])  # spaCy token-limit safety
        
        # Collect entities by type
        persons = [e.text for e in doc.ents if e.label_ == "PERSON"]
        orgs = [e.text for e in doc.ents if e.label_ == "ORG"]
        gpe = [e.text for e in doc.ents if e.label_ == "GPE"]
        dates = [e.text for e in doc.ents if e.label_ == "DATE"]
        
        # Extract education and committee roles using domain heuristics
        education = self._extract_education(text, orgs)
        committee_roles = self._extract_committee_roles(text, orgs)
        
        # Filter orgs for affiliation: exclude committee-related terms
        committee_keywords = ["committee", "subcommittee", "caucus"]
        filtered_orgs = [org for org in orgs 
                        if not any(kw in org.lower() for kw in committee_keywords)]
        
        # Filter gpe to only U.S. states (exclude cities and countries)
        filtered_gpe = [g for g in gpe if g.strip().title() in US_STATES]
        
        # Return comprehensive extraction results
        return {
            "name": persons[0] if persons else None,
            "state": filtered_gpe[0] if filtered_gpe else None,
            "affiliation": filtered_orgs[0] if filtered_orgs else None,
            "occupation": None,
            "education": education,
            "committee_roles": committee_roles,
        }

# ---------------------------------------------------------------------------
# Baseline 4: BERT-based NER  (Kleinberg et al. TextWash, Liu et al. §6.1.3)
# ---------------------------------------------------------------------------

class BERTBaseline:
    """
    BERT-based NER baseline (Liu et al. Section 6.1.3).

    Liu et al. use the TextWash fine-tuned BERT extractor (Kleinberg et al.,
    2022).  That model is not publicly available via a standard pip install,
    so this implementation uses the closest freely available alternative:
    'dslim/bert-base-NER' from HuggingFace, which is also fine-tuned on
    CoNLL-2003 NER and is the de-facto open-source BERT NER baseline used in
    replication studies.

    Requires:  pip install transformers torch
    """

    # HuggingFace model ID. Swap to a local TextWash checkpoint if available.
    DEFAULT_MODEL = "dslim/bert-base-NER"

    def __init__(self, model_name=None):
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
        # aggregation_strategy='simple' merges subword tokens into full words
        self._pipe = pipeline(
            "ner",
            model=self.model_name,
            aggregation_strategy="simple",
        )

    # ------------------------------------------------------------------
    def _aggregate(self, entities, label):
        """Return deduplicated text spans for a given NER label."""
        return list({e["word"] for e in entities if e["entity_group"] == label})

    def extract(self, text):
        """
        Extract named entities using BERT NER.

        Args:
            text: Input text (plain text, not raw HTML).
                  Truncated to 512 tokens internally by the model; pass
                  pre-processed text for best results.

        Returns:
            Dict with raw NER buckets (persons_detected, orgs_detected, loc_detected,
            misc_detected) and scalar PIE fields (name, affiliation, occupation,
            education, committee_roles) for compatibility with evaluation scoring.
        """
        # HuggingFace pipeline handles tokenisation and truncation
        entities = self._pipe(text[:2000])  # ~512 tokens ≈ 2000 chars

        # Raw NER entity buckets (for interpretability)
        persons_detected = self._aggregate(entities, "PER")[:5]
        orgs_detected = self._aggregate(entities, "ORG")[:10]
        loc_detected = self._aggregate(entities, "LOC")[:5]
        misc_detected = self._aggregate(entities, "MISC")[:5]
        
        # Scalar PIE fields (for scoring consistency with other baselines)
        # name: first person entity
        name = persons_detected[0] if persons_detected else None
        
        # affiliation: first org entity that does NOT contain committee keywords
        committee_keywords = ["committee", "subcommittee", "caucus"]
        filtered_orgs = [org for org in orgs_detected 
                        if not any(kw in org.lower() for kw in committee_keywords)]
        affiliation = filtered_orgs[0] if filtered_orgs else None
        
        # committee_roles: org entities that DO contain committee keywords
        committee_roles = [org for org in orgs_detected 
                          if any(kw in org.lower() for kw in committee_keywords)]
        committee_roles = committee_roles if committee_roles else None
        
        # occupation and education not extractable from raw NER buckets
        occupation = None
        education = None
        
        # Return merged dict: raw NER buckets + scalar PIE fields
        return {
            "persons_detected": persons_detected,
            "orgs_detected": orgs_detected,
            "loc_detected": loc_detected,
            "misc_detected": misc_detected,
            "name": name,
            "affiliation": affiliation,
            "occupation": occupation,
            "education": education,
            "committee_roles": committee_roles,
        }

# ---------------------------------------------------------------------------
# Baseline 5: TextWash NER (Kleinberg et al. 2022, open-source)
# ---------------------------------------------------------------------------

class TextWashBaseline:
    """
    TextWash NER baseline (Kleinberg et al., 2022).
    
    Open-source learned PII detection model. Directly comparable to Liu et al.
    Appendix results. Outperforms spaCy/BERT by using probabilistic entity
    classification trained specifically on PII.
    
    Setup:
      1. pip install textwash  (or clone from GitHub)
      2. Download models: https://drive.google.com/file/d/1YBccngYE3lvod87TI6UIhBzrN7nY9vHS/view?usp=sharing
      3. Extract to ./data/ such that ./data/en/ contains model files
    
    Reference: Kleinberg et al., "Textwash: A Comprehensive Text Anonymisation 
    Benchmark with Realistic PII", 2022.
    """

    # Maps TextWash entity types to PIE field names
    ENTITY_MAPPING = {
        "EMAIL_ADDRESS": "email",
        "PHONE_NUMBER": "phone",
        "ADDRESS": "address",
        "PERSON_FIRSTNAME": "name_first",
        "PERSON_LASTNAME": "name_last",
        "ORGANIZATION": "affiliation",
        "OCCUPATION": "occupation",
        "DATE": "date",
    }

    def __init__(self, language="en", data_dir=None):
        """
        Args:
            language: Language code ("en" for English, "nl" for Dutch).
            data_dir: Path to TextWash data directory containing model folders.
                      If None, uses ./data/
        """
        try:
            from textwash.anonymizer import Anonymizer
        except ImportError:
            raise RuntimeError(
                "TextWash not installed. Install from:\n"
                "  pip install textwash\n"
                "or\n"
                "  git clone https://github.com/ben-aaron188/textwash.git\n\n"
                "Then download models from: "
                "https://drive.google.com/file/d/1YBccngYE3lvod87TI6UIhBzrN7nY9vHS"
                " and extract to ./data/"
            )

        self.language = language
        self.data_dir = data_dir or "./data"
        
        try:
            self.anonymizer = Anonymizer(
                language=language,
                model_path=self.data_dir
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize TextWash. Verify model path {self.data_dir}: {e}"
            )

    def extract(self, text):
        """
        Extract entities using TextWash learned PII detection.

        Args:
            text: Input text to extract from.

        Returns:
            Dict mapping your PIE field names to extracted values.
        """
        try:
            # Get TextWash entity predictions (returns dict with entity types as keys)
            predictions = self.anonymizer.get_entities(text)
        except Exception as e:
            # Fallback to empty dict on extraction error
            print(f"Warning: TextWash extraction failed: {e}")
            predictions = {}

        # Map TextWash output → PIE fields using ENTITY_MAPPING
        result = {}
        for textwash_type, pie_field in self.ENTITY_MAPPING.items():
            # predictions[textwash_type] is a list of detected spans
            entities = predictions.get(textwash_type, [])
            
            if entities:
                # For name fields, return first occurrence; for others, join
                if pie_field in ["name_first", "name_last"]:
                    result[pie_field] = entities[0] if entities else None
                elif pie_field in ["email", "phone", "address"]:
                    # Return single best match (first)
                    result[pie_field] = entities[0] if entities else None
                else:
                    # affiliation, occupation: use first occurrence
                    result[pie_field] = entities[0] if entities else None
            else:
                result[pie_field] = None

        return result

# ---------------------------------------------------------------------------
# Convenience functions (notebook interface)
# ---------------------------------------------------------------------------

def regex_extract(text, regex_patterns):
    """Convenience wrapper around RegexBaseline.extract()."""
    return RegexBaseline(regex_patterns).extract(text)


def keyword_extract(text, keyword_map=None):
    """Convenience wrapper around KeywordSearchBaseline.extract()."""
    return KeywordSearchBaseline(keyword_map).extract(text)


def spacy_extract(text, nlp_model=None):
    """Convenience wrapper around SpaCyBaseline.extract()."""
    return SpaCyBaseline(nlp_model).extract(text)


def bert_extract(text, model_name=None):
    """Convenience wrapper around BERTBaseline.extract()."""
    return BERTBaseline(model_name).extract(text)


# ---------------------------------------------------------------------------
__all__ = [
    # Classes
    "RegexBaseline",
    "KeywordSearchBaseline",
    "SpaCyBaseline",
    "BERTBaseline",
    # Convenience functions
    "regex_extract",
    "keyword_extract",
    "spacy_extract",
    "bert_extract",
    # Constants
    "DEFAULT_KEYWORD_MAP",
]
