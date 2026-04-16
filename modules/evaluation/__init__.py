"""Evaluation metrics and scoring - accuracy computation, baselines, comparisons"""

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
from .suite import (
    load_and_merge_results,
    get_per_row_scores,
    evaluate_all_styles,
    print_evaluation_summary,
    load_baseline_results,
    print_baseline_summary,
)
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

__all__ = [
    # evaluator
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
    # suite
    "load_and_merge_results",
    "get_per_row_scores",
    "evaluate_all_styles",
    "print_evaluation_summary",
    "load_baseline_results",
    "print_baseline_summary",
    # baselines
    "RegexBaseline",
    "SpaCyBaseline",
    "KeywordSearchBaseline",
    "BERTBaseline",
    "regex_extract",
    "spacy_extract",
    "keyword_extract",
    "bert_extract",
    "DEFAULT_KEYWORD_MAP",
]
