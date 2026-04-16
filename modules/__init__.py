"""
LLM Pipeline Modules – Organized Structure

Submodules:
  - api/        : LLM provider integrations (Groq, OpenAI, etc.)
  - config/     : Configuration management, prompts, and API settings
  - data/       : Data import, HTML processing, and ground truth
  - evaluation/ : Evaluation metrics, scoring, and baselines
  - extraction/ : Session initialization and pipeline orchestration
  - utils/      : Utilities for parsing and name normalization

All exports are re-exported here for backward compatibility.
"""

# Re-export from submodules
from .api import call_groq, run_pipeline
from .config import (
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
    setup_logging,
    get_logger,
)
from .data import (
    HTMLProcessor,
    WikipediaExtractor,
    extract_readable_text,
    extract_infobox,
    extract_wikipedia_profile,
    merge_pew,
    detect_religion_signal,
    normalize_birthdate,
)
from .evaluation import (
    normalize_name,
    name_match_score,
    create_normalized_senator_id,
    match_by_fuzzy_name,
    gender_match_score,
    get_religion_category,
    religion_match_score,
    evaluate_text_fields,
    evaluate_education_components,
    load_and_merge_results,
    get_per_row_scores,
    evaluate_all_styles,
    print_evaluation_summary,
    load_baseline_results,
    print_baseline_summary,
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
from .extraction import (
    initialize_pipeline_session,
    run_main_pipeline,
    run_baselines,
)
from .utils import (
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
    NameNormalizer,
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
    # api
    "call_groq",
    "run_pipeline",
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
    "get_per_row_scores",
    "evaluate_all_styles",
    "print_evaluation_summary",
    "load_baseline_results",
    "print_baseline_summary",
    # session_init (Phase 2)
    "initialize_pipeline_session",
    # pipeline_runner (Phase 3)
    "run_main_pipeline",
    "run_baselines",
]
