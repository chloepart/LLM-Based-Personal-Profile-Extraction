"""
Centralized configuration for the Senate LLM extraction pipeline.
Single source of truth for paths, API settings, delays, and patterns.
"""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION PATHS (relative to workspace root)
# ─────────────────────────────────────────────────────────────────────────────

PATHS = {
    # Input data
    "html_data": Path("../external_data/senate_html"),
    "synthetic_data": Path("../data/synthetic"),
    "icl_data": Path("../data/icl"),
    "config_models": Path("../configs/model_configs"),
    "config_tasks": Path("../configs/task_configs"),
    "senators_index": Path("../external_data/senate_html/senators_index.csv"),
    "pew_religion": Path("../external_data/pew_religion.csv"),
    
    # Ground truth
    "ground_truth_dir": Path("../external_data/ground_truth"),
    "ground_truth_csv": Path("../external_data/ground_truth/senate_ground_truth.csv"),
    "scrape_errors_log": Path("../external_data/ground_truth/scrape_errors.log"),
    
    # Output
    "results_dir": Path("../outputs/senate_results"),
    "results_raw": Path("../outputs/senate_results/results_raw.json"),
    "results_csv": Path("../outputs/senate_results/task1_pii.csv"),
    "results_log": Path("../outputs/log"),
}


# ─────────────────────────────────────────────────────────────────────────────
# RATE LIMITING & DELAYS (seconds)
# Purpose: Avoid API quota exhaustion; respect web server limits
# ─────────────────────────────────────────────────────────────────────────────

DELAYS = {
    # Between LLM API calls (same provider)
    "lLM_call": 0.5,
    
    # Between different prompt styles when running all three
    "between_prompt_styles": 3.0,
    
    # Between HTTP requests (Wikipedia, Ballotpedia)
    "http_request": 1.0,
    
    # After successful HTTP request (to respect server load)
    "after_http_success": 0.5,
    
    # After validation error
    "after_error": 2.0,
    
    # After quota exhaustion error (before retry)
    "quota_exhaust_base": 2.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# REGEX PATTERNS (precompiled)
# ─────────────────────────────────────────────────────────────────────────────

PATTERNS = {
    # Email extraction
    "email": re.compile(
        r"(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"
        '"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|'
        '\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)'
        '+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.'
        '){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:'
        '(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])',
        re.IGNORECASE
    ),
    
    # Phone number extraction
    "phone": re.compile(r"(?:\+\d+\s?)?(?:\d{3}[\-\s]?\d{3,}[\-\s]?\d{4})"),
    
    # Year extraction (1940–2029)
    "year": re.compile(r"\b(19[4-9]\d|20[0-2]\d)\b"),
    
    # Senator name extraction (greedy: "Senator John Smith")
    "senator_name": re.compile(r"\bSenator\s+([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)"),
    
    # Political party keywords
    "party": re.compile(r"\b(Republican|Democrat|Democratic|Independent)\b", re.IGNORECASE),
    
    # Date patterns (MM/DD/YYYY or YYYY-MM-DD)
    "date_mdy": re.compile(r"\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])[/-](\d{4})\b"),
    "date_ymd": re.compile(r"\b(\d{4})[/-](0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])\b"),
}


# ─────────────────────────────────────────────────────────────────────────────
# LLM PROVIDER & MODEL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """Configuration for LLM API calls."""
    provider: str  # "groq", "gemini", "openai", etc.
    model_name: str
    temperature: float = 0.3  # Low temperature for deterministic extraction
    max_tokens: int = 2048
    max_retries: int = 5
    
    def __str__(self):
        return f"{self.provider}/{self.model_name} (T={self.temperature}, max_tok={self.max_tokens})"


# Default configs for different models
DEFAULT_CONFIGS = {
    "groq_8b": LLMConfig(
        provider="groq",
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=2048,
    ),
    "groq_70b": LLMConfig(
        provider="groq",
        model_name="llama-3.1-70b-versatile",
        temperature=0.3,
        max_tokens=2048,
    ),
    "gemini_1_5": LLMConfig(
        provider="gemini",
        model_name="gemini-1.5-pro",
        temperature=0.3,
        max_tokens=2048,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION FIELD DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

TASK1_FIELDS = [
    "full_name",
    "birthdate",
    "birth_year_inferred",
    "gender",
    "race_ethnicity",
    "education",
    "committee_roles",
    "religious_affiliation",
    "religious_affiliation_inferred",
]

# Fields that support exact match scoring
EXACT_MATCH_FIELDS = ["full_name", "gender", "birthdate"]

# Fields that support ROUGE/text overlap scoring
TEXT_OVERLAP_FIELDS = ["education", "committee_roles", "religious_affiliation"]

# Fields supporting BERT semantic similarity scoring
SEMANTIC_FIELDS = ["education", "religious_affiliation"]


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT STYLES
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_STYLES = ["direct", "pseudocode", "icl"]  # Canonical selection


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION & METRICS CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

EVAL_CONFIG = {
    # Year matching: allow +/- N years tolerance
    "year_tolerance": 2,
    
    # Name matching: similarity threshold
    "name_similarity_threshold": 0.90,
    
    # Education matching: if component parts match > N%, consider overall match
    "component_match_threshold": 0.70,
    
    # Sample size for diagnostics
    "diagnostic_sample_size": 5,
}


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_path(key: str) -> Path:
    """
    Get a configured path by key.
    
    Args:
        key: Path key from PATHS dict
        
    Returns:
        Path object
        
    Raises:
        KeyError if key not found
    """
    if key not in PATHS:
        raise KeyError(f"Unknown path key: {key}. Available: {list(PATHS.keys())}")
    return PATHS[key]


def ensure_paths_exist():
    """Create all configured output directories if they don't exist."""
    for key, path in PATHS.items():
        if "dir" in key and isinstance(path, Path):
            path.mkdir(parents=True, exist_ok=True)


def get_config(model_key: str = "groq_70b") -> LLMConfig:
    """
    Get LLM configuration by key.
    
    Args:
        model_key: Key in DEFAULT_CONFIGS
        
    Returns:
        LLMConfig object
    """
    if model_key not in DEFAULT_CONFIGS:
        raise KeyError(f"Unknown model config: {model_key}. Available: {list(DEFAULT_CONFIGS.keys())}")
    return DEFAULT_CONFIGS[model_key]
