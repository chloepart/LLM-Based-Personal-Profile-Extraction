"""
LLM Pipeline Modules
Consolidated utilities for configuration, name processing, HTML extraction,
parsing, baselines, and evaluation.
"""

from .config import (
    PATHS,
    TASK1_DIRECT,
    TASK1_PSEUDOCODE,
    TASK1_ICL,
    PROMPT_STYLE_MAP,
    ABLATION_STYLES,
    EDUCATION_PROMPT,
    PipelineConfig,
    T1_FIELDS,
    T1_FIELDS_CMP,
    GT_FIELDS,
    REGEX_PATTERNS,
    RELIGION_HIERARCHY,
)

from .name_utils import NameNormalizer

from .html_processing import (
    HTMLProcessor,
    WikipediaExtractor,
    extract_readable_text,
    extract_infobox,
)

# NEW: Parsing utilities (consolidates 3 variants into single module)
from .parsing import (
    EducationParser,
    parse_education_detailed,
    parse_education,
    parse_committee_roles,
    DegreeNormalizer,
    normalize_degree,
    SchoolNormalizer,
    normalize_school,
    compare_education_components,
)

# NEW: Baseline extractors (regex + spaCy NER + keyword search + BERT NER)
from .baselines import (
    RegexBaseline,
    SpaCyBaseline,
    KeywordSearchBaseline,
    BERTBaseline,
    regex_extract,
    spacy_extract,
    keyword_extract,
    bert_extract,
    DEFAULT_KEYWORD_MAP,
)

# NEW: Evaluation metrics and scoring functions
from .evaluator import (
    normalize_name,
    name_match_score,
    create_normalized_senator_id,
    match_by_fuzzy_name,
    parse_date,
    birthdate_scores,
    gender_match_score,
    get_religion_category,
    religion_match_score,
)

__all__ = [
    # config
    "PATHS",
    "TASK1_DIRECT",
    "TASK1_PSEUDOCODE",
    "TASK1_ICL",
    "PROMPT_STYLE_MAP",
    "ABLATION_STYLES",
    "EDUCATION_PROMPT",
    "PipelineConfig",
    "T1_FIELDS",
    "T1_FIELDS_CMP",
    "GT_FIELDS",
    "REGEX_PATTERNS",
    "RELIGION_HIERARCHY",
    # name_utils
    "NameNormalizer",
    # html_processing
    "HTMLProcessor",
    "WikipediaExtractor",
    "extract_readable_text",
    "extract_infobox",
    # parsing (NEW)
    "EducationParser",
    "parse_education_detailed",
    "parse_education",
    "parse_committee_roles",
    "DegreeNormalizer",
    "normalize_degree",
    "SchoolNormalizer",
    "normalize_school",
    "compare_education_components",
    # baselines (NEW)
    "RegexBaseline",
    "SpaCyBaseline",
    "KeywordSearchBaseline",
    "BERTBaseline",
    "regex_extract",
    "spacy_extract",
    "keyword_extract",
    "bert_extract",
    "DEFAULT_KEYWORD_MAP",
    # evaluator (NEW)
    "normalize_name",
    "name_match_score",
    "create_normalized_senator_id",
    "match_by_fuzzy_name",
    "parse_date",
    "birthdate_scores",
    "gender_match_score",
    "get_religion_category",
    "religion_match_score",
]
