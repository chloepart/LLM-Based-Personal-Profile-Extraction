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
# Baseline 1: Regex
# ---------------------------------------------------------------------------

class RegexBaseline:
    """
    Regex-based extraction baseline (Liu et al. Section 6.1.3).

    Domain note: 'party' and 'years_found' are senator-specific adaptations
    not present in Liu et al.'s original 8-category schema.
    """

    def __init__(self, regex_patterns, name_re=None, party_re=None,
                 year_re=None, email_re=None, phone_re=None):
        """
        Initialize with compiled regex patterns.

        Args:
            regex_patterns: Dict with pattern keys or individual patterns.
            name_re:   Compiled regex for name extraction.
            party_re:  Compiled regex for party/affiliation extraction.
            year_re:   Compiled regex for year extraction.
            email_re:  Compiled regex for email extraction.
            phone_re:  Compiled regex for phone extraction.
        """
        if isinstance(regex_patterns, dict):
            self.name_re  = regex_patterns.get('NAME_RE')          or name_re
            self.party_re = regex_patterns.get('PARTY_KEYWORD_RE') or party_re
            self.year_re  = regex_patterns.get('YEAR_RE')          or year_re
            self.email_re = regex_patterns.get('EMAIL_RE')         or email_re
            self.phone_re = regex_patterns.get('PHONE_RE')         or phone_re
        else:
            self.name_re  = name_re
            self.party_re = party_re
            self.year_re  = year_re
            self.email_re = email_re
            self.phone_re = phone_re

    def extract(self, text):
        """
        Extract structured fields using regex patterns.

        Args:
            text: Input text to extract from.

        Returns:
            Dict with keys: full_name, party, email, phone, years_found.
        """
        name_m  = self.name_re.search(text)  if self.name_re  else None
        party_m = self.party_re.search(text) if self.party_re else None
        years   = self.year_re.findall(text) if self.year_re  else []

        return {
            "full_name":   name_m.group(1)              if name_m  else None,
            "party":       party_m.group(0).title()     if party_m else None,
            "email":       self.email_re.findall(text)  if self.email_re else None,
            "phone":       self.phone_re.findall(text)  if self.phone_re else None,
            "years_found": years[:5]                    if years   else None,
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
# Baseline 3: spaCy NER
# ---------------------------------------------------------------------------

class SpaCyBaseline:
    """spaCy NER baseline (Liu et al. Section 6.1.3)."""

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

    def extract(self, text):
        """
        Extract named entities using spaCy.

        Args:
            text: Input text to extract from.

        Returns:
            Dict with keys: persons_detected, orgs_detected, gpe_detected,
            dates_detected.  All are deduplicated lists.
        """
        doc = self.nlp(text[:10000])  # spaCy token-limit safety

        return {
            "persons_detected": list({e.text for e in doc.ents if e.label_ == "PERSON"})[:5],
            "orgs_detected":    list({e.text for e in doc.ents if e.label_ == "ORG"})[:10],
            "gpe_detected":     list({e.text for e in doc.ents if e.label_ == "GPE"})[:5],
            "dates_detected":   list({e.text for e in doc.ents if e.label_ == "DATE"})[:5],
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
            Dict with keys: persons_detected, orgs_detected, loc_detected,
            misc_detected.
        """
        # HuggingFace pipeline handles tokenisation and truncation
        entities = self._pipe(text[:2000])  # ~512 tokens ≈ 2000 chars

        return {
            "persons_detected": self._aggregate(entities, "PER")[:5],
            "orgs_detected":    self._aggregate(entities, "ORG")[:10],
            "loc_detected":     self._aggregate(entities, "LOC")[:5],
            "misc_detected":    self._aggregate(entities, "MISC")[:5],
        }


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
