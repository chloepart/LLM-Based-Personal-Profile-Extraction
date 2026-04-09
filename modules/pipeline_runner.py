"""
Pipeline execution and orchestration for Senate LLM Extraction.

Handles:
- Main extraction pipeline with rate limiting and resume safety
- Baseline execution (regex, spacy, keyword)
- Results aggregation and CSV output
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm.notebook import tqdm

from .api import run_pipeline, call_groq
from .html_processing import extract_readable_text
from .baselines import regex_extract, spacy_extract, keyword_extract
from .config_unified import T1_FIELDS, REGEX_PATTERNS


def run_main_pipeline(
    html_files: List[Path],
    session_config: Any,
    output_dir: Path = None,
    resume: bool = True,
    rate_limit_seconds: float = 6
) -> Dict[str, Any]:
    """
    Run main LLM extraction pipeline on all HTML files.
    
    Features:
    - Resume-safe: skips already-completed senators
    - Rate limiting: respects API quotas
    - Incremental saves: results saved after each senator
    - Progress bar: tqdm integration for notebook
    
    Args:
        html_files: List of HTML file paths to process
        session_config: PipelineConfig instance with API settings
        output_dir: Output directory (defaults to session_config.output_dir)
        resume: Whether to skip already-completed files (default: True)
        rate_limit_seconds: Delay between API calls (default: 6)
        
    Returns:
        Dictionary with keys:
            - results: List of extraction results (one per senator)
            - output_path: Path to saved results JSON
            - done_count: Number of senators processed
            - total_count: Total senators available
    """
    
    if output_dir is None:
        output_dir = session_config.output_dir
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_path = output_dir / "results_raw.json"
    
    # ─────────────────────────────────────────────────────────────────────
    # Resume from saved results
    # ─────────────────────────────────────────────────────────────────────
    
    if resume and raw_path.exists():
        with open(raw_path) as f:
            results = json.load(f)
        done_ids = {r["senator_id"] for r in results}
        print(f"✓ Resuming: {len(done_ids)} senators already processed")
    else:
        results, done_ids = [], set()
    
    remaining = [f for f in html_files if f.stem not in done_ids]
    
    print(f"Senators remaining: {len(remaining)}/{len(html_files)}")
    print(f"Rate limit: {rate_limit_seconds}s between calls")
    print(f"Styles: {', '.join(session_config.prompt_styles)}\n")
    
    # ─────────────────────────────────────────────────────────────────────
    # Main extraction loop
    # ─────────────────────────────────────────────────────────────────────
    
    for html_file in tqdm(remaining, desc="Processing senators"):
        senator_id = html_file.stem
        
        # Read and clean HTML
        html = html_file.read_text(encoding="utf-8", errors="ignore")
        text = extract_readable_text(html)
        
        # Run pipeline (handles all prompt styles)
        result = run_pipeline(text, session_config)
        
        # Append result
        results.append({"senator_id": senator_id, **result})
        
        # Save incrementally (critical for resume safety)
        with open(raw_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # Rate limiting
        time.sleep(rate_limit_seconds)
    
    print(f"\n✓ Pipeline complete: {len(results)} senators processed")
    
    return {
        "results": results,
        "output_path": raw_path,
        "done_count": len(results),
        "total_count": len(html_files),
    }


def flatten_task1_results(
    results: List[Dict],
    output_dir: Path = None,
    task1_fields: List[str] = None
) -> pd.DataFrame:
    """
    Flatten Task 1 results to CSV format.
    
    Converts nested prompt_style results into one row per (senator, style) pair.
    
    Args:
        results: List of raw results from run_main_pipeline()
        output_dir: Directory to save CSV (saves if provided)
        task1_fields: Fields to include (default: T1_FIELDS)
        
    Returns:
        DataFrame with columns: senator_id, prompt_style, extraction_error, + all task1 fields
    """
    
    if task1_fields is None:
        task1_fields = T1_FIELDS
    
    task1_rows = []
    
    for r in results:
        t1_data = r.get("task1_pii", {})
        prompt_style = r.get("prompt_style", "unknown")
        
        # Handle both single-style and all-styles results
        if prompt_style == "all_styles":
            # All three styles were run
            for style_name, style_result in t1_data.items():
                row = {
                    "senator_id": r["senator_id"],
                    "prompt_style": style_name,
                    "extraction_error": style_result.get("error")
                }
                for field in task1_fields:
                    row[field] = style_result.get(field)
                task1_rows.append(row)
        else:
            # Single style was run
            row = {
                "senator_id": r["senator_id"],
                "prompt_style": prompt_style,
                "extraction_error": t1_data.get("error")
            }
            for field in task1_fields:
                row[field] = t1_data.get(field)
            task1_rows.append(row)
    
    df_t1 = pd.DataFrame(task1_rows)
    
    # Save if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_path = output_dir / "task1_pii.csv"
        df_t1.to_csv(output_path, index=False)
        print(f"✓ Saved {len(df_t1)} rows to {output_path}")
    
    return df_t1


def run_baselines(
    html_files: List[Path],
    nlp,
    regex_patterns: Dict = None,
    output_dir: Path = None
) -> pd.DataFrame:
    """
    Run all baseline extractors on HTML files.
    
    Executes:
    - Regex baseline: pattern matching for email, phone, names
    - SpaCy baseline: neural entity recognition for PERSON, ORG, GEO
    - Keyword baseline: keyword search (on first 10 for speed)
    
    Args:
        html_files: List of HTML file paths
        nlp: Loaded SpaCy model (from session.nlp)
        regex_patterns: Regex patterns dict (default: REGEX_PATTERNS)
        output_dir: Directory to save results (saves if provided)
        
    Returns:
        DataFrame with baseline results
    """
    
    if regex_patterns is None:
        regex_patterns = REGEX_PATTERNS
    
    baseline_rows = []
    
    # ─────────────────────────────────────────────────────────────────────
    # Run regex + spaCy on all files
    # ─────────────────────────────────────────────────────────────────────
    
    print(f"\nRunning baselines on {len(html_files)} profiles...")
    
    for hf in tqdm(html_files, desc="Baselines"):
        html = hf.read_text(encoding="utf-8", errors="ignore")
        text = extract_readable_text(html)
        
        # Regex extraction
        regex_r = regex_extract(text, regex_patterns)
        
        # SpaCy extraction
        spacy_r = spacy_extract(text, nlp)
        
        baseline_rows.append({
            "senator_id": hf.stem,
            "regex_name": regex_r["full_name"],
            "regex_email_found": 1 if regex_r["email"] else 0,
            "regex_phone_found": 1 if regex_r["phone"] else 0,
            "spacy_top_person": spacy_r["persons_detected"][0] if spacy_r["persons_detected"] else None,
            "spacy_orgs_count": len(spacy_r["orgs_detected"]),
        })
    
    df_bl = pd.DataFrame(baseline_rows)
    
    # Save if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_path = output_dir / "baselines.csv"
        df_bl.to_csv(output_path, index=False)
        print(f"✓ Saved baseline results to {output_path}\n")
    
    return df_bl


def compare_baseline_coverage(df_llm: pd.DataFrame, df_baseline: pd.DataFrame) -> Dict[str, float]:
    """
    Compare LLM extraction coverage vs baseline methods.
    
    Args:
        df_llm: DataFrame from flatten_task1_results()
        df_baseline: DataFrame from run_baselines()
        
    Returns:
        Dictionary with coverage percentages for each method
    """
    
    # Merge on senator_id
    merged = df_llm.merge(df_baseline, on="senator_id")
    
    coverage = {
        "llm_name": merged["full_name"].notna().mean(),
        "regex_name": merged["regex_name"].notna().mean(),
        "spacy_person": merged["spacy_top_person"].notna().mean(),
    }
    
    print("\n" + "=" * 70)
    print(" BASELINE COVERAGE COMPARISON (Liu et al. Table 4–5)")
    print("=" * 70)
    print(f"  LLM   : {coverage['llm_name']:.1%}")
    print(f"  Regex : {coverage['regex_name']:.1%}")
    print(f"  spaCy : {coverage['spacy_person']:.1%}")
    print("=" * 70 + "\n")
    
    return coverage
