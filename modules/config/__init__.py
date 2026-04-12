"""Configuration management - settings, prompts, and API configuration"""

from .config import PATHS
from .config_unified import (
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
from .logging import setup_logging, get_logger

__all__ = [
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
    "setup_logging",
    "get_logger",
]
