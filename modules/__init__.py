"""
LLM Pipeline Modules
Consolidated utilities for configuration, name processing, HTML extraction,
parsing, baselines, and evaluation.
"""

from .config_unified import (
    PATHS,
    TASK1_DIRECT,
    TASK1_PSEUDOCODE,
    TASK1_ICL,
    PROMPT_STYLE_MAP,
    ABLATION_STYLES,
    EDUCATION_PROMPT,
    APIConfig,
    RateLimitConfig,
    AblationConfig,
    PipelineConfig,
    T1_FIELDS,
    T1_FIELDS_CMP,
    GT_FIELDS,
    REGEX_PATTERNS,
    RELIGION_HIERARCHY,
)

from .logging_config import setup_logging, get_logger

from .name_utils import NameNormalizer

from .html_processing import (
    HTMLProcessor,
    WikipediaExtractor,
    extract_readable_text,
    extract_infobox,
)

# NEW: HTML utilities (consolidated from static methods)
from .html_utils import (
    extract_readable_text as extract_readable_text_fn,
    extract_infobox as extract_infobox_fn,
    extract_wikipedia_profile,
)

# NEW: Data merging and signal detection
from .data_merge import (
    merge_pew,
    detect_religion_signal,
    normalize_birthdate,
)

from .api import (
    call_groq,
    run_pipeline,
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
    parse_date,
    birthdate_scores,
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
    evaluate_text_fields,
    evaluate_education_components,
)

# NEW: Evaluation suite with caching (Phase 4)
from .evaluation_suite import (
    load_and_merge_results,
    evaluate_all_styles,
    print_evaluation_summary,
)

# NEW: Session initialization (Phase 2)
from .session_init import (
    initialize_pipeline_session,
)

# NEW: Pipeline orchestration (Phase 3)
from .pipeline_runner import (
    run_main_pipeline,
    run_baselines,
)

__all__ = [
    # config_unified
    "PATHS",
    "TASK1_DIRECT",
    "TASK1_PSEUDOCODE",
    "TASK1_ICL",
    "PROMPT_STYLE_MAP",
    "ABLATION_STYLES",
    "EDUCATION_PROMPT",
    "APIConfig",
    "RateLimitConfig",
    "AblationConfig",
    "PipelineConfig",
    "T1_FIELDS",
    "T1_FIELDS_CMP",
    "GT_FIELDS",
    "REGEX_PATTERNS",
    "RELIGION_HIERARCHY",
    # logging_config
    "setup_logging",
    "get_logger",
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
        # evaluator (expanded)
    "normalize_name",
    "name_match_score",
    "create_normalized_senator_id",
    "match_by_fuzzy_name",
    "parse_date",
    "birthdate_scores",
    "gender_match_score",
    "get_religion_category",
    "religion_match_score",
    "evaluate_text_fields",
    "evaluate_education_components",
    # evaluation_suite (Phase 4)
    "load_and_merge_results",
    "evaluate_all_styles",
    "print_evaluation_summary",
    # session_init (Phase 2)
    "initialize_pipeline_session",
    # pipeline_runner (Phase 3)
    "run_main_pipeline",
    "run_baselines",
]
