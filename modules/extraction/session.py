"""
Session initialization for Senate LLM Extraction Pipeline.

Encapsulates setup: API configuration, paths, spacy model, pipeline config.
Single point of control for reproducible session state across notebook reruns.
"""

import json
import os
import warnings
from pathlib import Path
import spacy
import pandas as pd

from ..config.config_unified import PipelineConfig, ABLATION_STYLES


def initialize_pipeline_session(
    config_path: Path = None,
    prompt_styles: list = None,
    spacy_model: str = "en_core_web_sm",
    suppress_warnings: bool = True
) -> dict:
    """
    One-call initialization of the entire pipeline session.
    
    Args:
        config_path: Path to Groq config JSON (default: configs/model_configs/groq_config_extraction.json)
        prompt_styles: List of prompt styles to use (default: ["direct", "pseudocode", "icl"])
        spacy_model: SpaCy model to load (default: "en_core_web_sm")
        suppress_warnings: Suppress UserWarning messages (default: True)
        
    Returns:
        Dictionary with keys:
            - session_config: PipelineConfig instance
            - html_files: List of HTML file paths
            - nlp: Loaded SpaCy model
            - api_client: Groq API client
            - output_dir: Output directory path
            - html_dir: HTML directory path
    """
    
    if suppress_warnings:
        warnings.filterwarnings("ignore", category=UserWarning)
    
    # Default config path
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "model_configs" / "groq_config_extraction.json"
    else:
        config_path = Path(config_path)
    
    # Default prompt styles
    if prompt_styles is None:
        prompt_styles = ["direct", "pseudocode", "icl"]
    
    # ─────────────────────────────────────────────────────────────────────
    # Load configuration
    # ─────────────────────────────────────────────────────────────────────
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Create pipeline config from JSON
    pipeline_config = PipelineConfig.from_groq_config_json(config_path=config_path)
    pipeline_config.prompt_styles = prompt_styles
    
    # Build prompt map
    pipeline_config.prompt_map = {
        "direct": ABLATION_STYLES["direct"],
        "pseudocode": ABLATION_STYLES["pseudocode"],
        "icl": ABLATION_STYLES["icl"]
    }
    
    # ─────────────────────────────────────────────────────────────────────
    # Load data files
    # ─────────────────────────────────────────────────────────────────────
    
    html_dir = pipeline_config.html_dir
    html_files = sorted(html_dir.glob("*.html")) if html_dir.exists() else []
    
    # ─────────────────────────────────────────────────────────────────────
    # Load SpaCy model
    # ─────────────────────────────────────────────────────────────────────
    
    try:
        nlp = spacy.load(spacy_model)
    except OSError:
        raise RuntimeError(
            f"SpaCy model '{spacy_model}' not found. "
            f"Install it with: python -m spacy download {spacy_model}"
        )
    
    # ─────────────────────────────────────────────────────────────────────
    # Return session state
    # ─────────────────────────────────────────────────────────────────────
    
    return {
        "session_config": pipeline_config,
        "pipeline_config": pipeline_config,  # Alias for backward compatibility
        "html_files": html_files,
        "nlp": nlp,
        "api_client": pipeline_config.api_client,
        "output_dir": pipeline_config.output_dir,
        "html_dir": pipeline_config.html_dir,
        "config_path": config_path,
    }


def print_session_summary(session: dict) -> None:
    """
    Print a formatted summary of the initialized session.
    
    Args:
        session: Dictionary returned by initialize_pipeline_session()
    """
    config = session["session_config"]
    
    print("\n" + "=" * 70)
    print("PIPELINE SESSION INITIALIZED")
    print("=" * 70)
    print(f"  Model              : {config.model}")
    print(f"  Prompt styles      : {', '.join(config.prompt_styles)}")
    print(f"  Temperature        : {config.api_config.temperature}")
    print(f"  Max tokens         : {config.api_config.max_output_tokens}")
    print(f"  HTML files found   : {len(session['html_files'])}")
    print(f"  Output directory   : {config.output_dir}")
    print(f"  Config file        : {session['config_path']}")
    print("=" * 70 + "\n")
