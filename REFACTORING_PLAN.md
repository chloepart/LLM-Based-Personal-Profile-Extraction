# Refactoring Action Plan

## Phase 1: Foundation (2-3 hours) — Do This First

### 1.1 Create Shared HTML Utilities Module

**Create**: `LLMPersonalInfoExtraction/utils/html_utils.py`

```python
"""
HTML extraction and validation utilities.
Consolidates duplicated functions from audit_pipeline_inputs.py, 
rescrape_flagged.py, and senate_llm_pipeline.ipynb
"""

from bs4 import BeautifulSoup
from typing import Tuple

def extract_readable_text(html: str, excluded_tags=None) -> str:
    """
    Extract clean readable text from HTML, removing script/style/nav blocks.
    
    Args:
        html: Raw HTML string
        excluded_tags: List of tags to remove (default: script, style, nav, footer, noscript)
    
    Returns:
        Clean text with single spaces between elements
    """
    if excluded_tags is None:
        excluded_tags = ["script", "style", "nav", "footer", "noscript"]
    
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(excluded_tags):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def validate_html_content(html: str, min_chars: int = 300) -> Tuple[bool, int]:
    """
    Check if HTML has sufficient non-boilerplate content.
    
    Args:
        html: Raw HTML string
        min_chars: Minimum acceptable cleaned text length
    
    Returns:
        Tuple of (is_valid, text_length)
    """
    text = extract_readable_text(html)
    return len(text) >= min_chars, len(text)


def detect_error_page(html: str) -> str:
    """
    Detect if HTML is a 404/error page.
    
    Returns:
        Error phrase found, or None if no error detected
    """
    text_lower = extract_readable_text(html).lower()
    
    error_phrases = [
        "404 error", "page not found", "requested page not found",
        "error page", "this page does not exist", "no longer available",
        "has moved",
    ]
    
    for phrase in error_phrases:
        if phrase in text_lower:
            return phrase
    return None


def normalize_filename_to_name(stem: str) -> str:
    """
    Convert filename stem to readable name.
    
    Example: "Bernie_Moreno_OH" → "Bernie Moreno"
    
    Args:
        stem: Filename stem (without .html extension)
    
    Returns:
        Properly formatted name string
    """
    parts = stem.split("_")
    # Drop last part if it's a 2-letter state code
    if parts and parts[-1].isupper() and len(parts[-1]) == 2:
        parts = parts[:-1]
    return " ".join(parts)
```

**Then update**:
- [experiments/audit_pipeline_inputs.py](experiments/audit_pipeline_inputs.py#L31) → `from LLMPersonalInfoExtraction.utils.html_utils import extract_readable_text, ...`
- [experiments/rescrape_flagged.py](experiments/rescrape_flagged.py#L43) → `from LLMPersonalInfoExtraction.utils.html_utils import extract_readable_text, ...`
- [experiments/senate_llm_pipeline.ipynb](experiments/senate_llm_pipeline.ipynb) → Add import at top, use functions

---

### 1.2 Create Path Manager

**Create**: `LLMPersonalInfoExtraction/config/paths.py`

```python
"""
Centralized path management.
Resolves all paths relative to project root, independent of caller location.
"""

from pathlib import Path

class PathManager:
    """Singleton path resolver for the entire project."""
    
    # Detect project root
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    
    @classmethod
    def data_dir(cls) -> Path:
        """Data directory containing all input data."""
        path = cls.PROJECT_ROOT / "data"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @classmethod
    def senate_html_dir(cls) -> Path:
        """Directory containing senate HTML files."""
        path = cls.PROJECT_ROOT / "external_data" / "senate_html"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @classmethod
    def ground_truth_dir(cls) -> Path:
        """Directory containing ground truth data."""
        path = cls.PROJECT_ROOT / "external_data" / "ground_truth"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @classmethod
    def config_dir(cls) -> Path:
        """Directory containing configuration files."""
        path = cls.PROJECT_ROOT / "configs"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @classmethod
    def model_config(cls, provider: str) -> Path:
        """Get path to model configuration file."""
        return cls.config_dir() / "model_configs" / f"{provider}_config.json"
    
    @classmethod
    def task_config(cls, dataset: str) -> Path:
        """Get path to task configuration file."""
        return cls.config_dir() / "task_configs" / f"{dataset}.json"
    
    @classmethod
    def output_dir(cls) -> Path:
        """Root output directory."""
        path = cls.PROJECT_ROOT / "outputs"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @classmethod
    def result_dir(cls, provider: str, model_name: str) -> Path:
        """Get path to results directory for specific model."""
        # Extract short name from full model name
        model_short = model_name.split("/")[-1]
        path = cls.output_dir() / "result" / f"{provider}_{model_short}"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @classmethod
    def log_dir(cls, provider: str, model_name: str) -> Path:
        """Get path to log directory for specific model."""
        model_short = model_name.split("/")[-1]
        path = cls.output_dir() / "log" / f"{provider}_{model_short}"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @classmethod
    def ground_truth_csv(cls) -> Path:
        """Get path to ground truth CSV."""
        return cls.ground_truth_dir() / "senate_ground_truth.csv"
    
    @classmethod
    def get_senator_html(cls, senator_id: str) -> Path:
        """Get path to HTML file for specific senator."""
        return cls.senate_html_dir() / f"{senator_id}.html"
```

**Usage in code**:
```python
# Instead of:
# html_path = "../external_data/senate_html/John_Doe_NY.html"

# Use:
from LLMPersonalInfoExtraction.config.paths import PathManager
html_path = PathManager.get_senator_html("John_Doe_NY")

# Instead of:
# with open(f'../outputs/result/{provider}_{model}.json') as f:

# Use:
result_dir = PathManager.result_dir(provider, model)
with open(result_dir / "results.json") as f:
```

---

### 1.3 Add Logging Setup

**Create**: `LLMPersonalInfoExtraction/logging_config.py`

```python
"""
Centralized logging configuration.
Replace all print() statements with logger calls.
"""

import logging
import sys
from pathlib import Path

def setup_logging(
    name: str = 'PIE',
    level: int = logging.INFO,
    log_file: Path = None,
    verbose: bool = False
) -> logging.Logger:
    """
    Initialize logger with console and optional file output.
    
    Args:
        name: Logger name
        level: Logging level (INFO, DEBUG, WARNING, ERROR)
        log_file: Optional file path for log output
        verbose: If True, sets DEBUG level
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else level)
    logger.propagate = False
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Convenience module-level logger
logger = setup_logging()
```

**Usage**:
```python
import logging
from LLMPersonalInfoExtraction.logging_config import logger

# Instead of:
# print(f'Processing senator {i+1}/{total}')
logger.info(f'Processing senator {i+1}/{total}')

# Instead of:
# print('ERROR: Could not parse API response')
logger.error('Could not parse API response')
```

---

## Phase 2: Refactor Large Cell (2-3 hours)

### 2.1 Extract Evaluation Metrics to Module

**Create**: `LLMPersonalInfoExtraction/metrics/eval_metrics.py`

Move Cell 39 (~654 lines) to this module and split into:

```python
"""
Evaluation metrics module.
Implements metrics from Liu et al. Section 6.1.4

Includes:
- Name matching with normalization
- Date validation and scoring
- Education parsing (multiple formats)
- Religion categorization
- Ground truth merging
"""

import ast
import re
import unicodedata
from typing import Dict, List, Optional, Literal, Tuple
from difflib import SequenceMatcher
import pandas as pd

class Normalizers:
    """Text normalization utilities."""
    
    NICKNAME_EXPANSIONS = {
        r"\bdan\b": "daniel",
        r"\btom\b": "thomas",
        r"\bjon\b": "jonathan",
        # ... (all expansions)
    }
    
    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize name: lowercase, strip accents, expand nicknames."""
        if not name or (isinstance(name, float) and pd.isna(name)):
            return ""
        name = str(name).lower().strip()
        # Remove accents
        name = "".join(
            c for c in unicodedata.normalize("NFD", name)
            if unicodedata.category(c) != "Mn"
        )
        # Expand nicknames
        for pattern, replacement in Normalizers.NICKNAME_EXPANSIONS.items():
            name = re.sub(pattern, replacement, name)
        return name


class MetricsScorer:
    """Scoring metrics for personal information extraction."""
    
    @staticmethod
    def name_match_score(gt_name: str, pred_name: str) -> float:
        """
        Score name match: 1.0 if identical or >90% similar after normalization.
        """
        gt_norm = Normalizers.normalize_name(gt_name)
        pred_norm = Normalizers.normalize_name(pred_name)
        
        if not gt_norm or not pred_norm:
            return 1.0 if gt_norm == pred_norm else 0.0
        if gt_norm == pred_norm:
            return 1.0
        
        similarity = SequenceMatcher(None, gt_norm, pred_norm).ratio()
        return 1.0 if similarity > 0.90 else 0.0
    
    @staticmethod
    def date_match_score(gt_date: str, pred_date: str) -> float:
        """
        Score date match: 1.0 if exact, 0.5 if year only, 0.0 if mismatch.
        Handles missing values and various date formats.
        """
        # ... (date normalization and validation logic)
        pass
    
    @staticmethod
    def gender_match_score(gt_gender: str, pred_gender: str) -> float:
        """Exact match for gender field."""
        if pd.isna(gt_gender) and pd.isna(pred_gender):
            return 1.0
        return 1.0 if str(gt_gender).lower() == str(pred_gender).lower() else 0.0


class EducationParser:
    """Parse education from multiple input formats."""
    
    @staticmethod
    def parse(
        edu_input: Optional[object],
        format: Literal['auto', 'json', 'pipe', 'list'] = 'auto'
    ) -> List[Dict[str, Optional[str]]]:
        """
        Parse education in multiple formats.
        
        Formats:
        - 'json': '[{"degree":"B.A.","institution":"Stanford","year":1982}]'
        - 'pipe': 'B.A.|Stanford|1982|J.D.|Harvard|1986'
        - 'list': Python list of dicts
        - 'auto': Try all formats
        
        Returns:
            List of {degree, institution, year} dicts or empty list
        """
        # ... (try each format, return parsed list)
        pass


class ReligionCategorizer:
    """Categorize religion strings into hierarchy."""
    
    HIERARCHY = {
        # Catholic
        "catholic": "catholic",
        "roman catholic": "catholic",
        # ... (20+ entries)
    }
    
    @staticmethod
    def categorize(religion_str: Optional[str]) -> Optional[str]:
        """
        Normalize religion string and return its category.
        Returns None if empty/missing/none.
        """
        if pd.isna(religion_str) or not str(religion_str).strip():
            return None
        
        norm = str(religion_str).strip().lower()
        return ReligionCategorizer.HIERARCHY.get(norm, None)
```

**Then in notebook**: Delete Cell 39, replace with:

```python
from LLMPersonalInfoExtraction.metrics.eval_metrics import (
    MetricsScorer, EducationParser, ReligionCategorizer, Normalizers
)

# All metrics now available as clean functions
name_score = MetricsScorer.name_match_score(gt, pred)
education_list = EducationParser.parse(edu_str)
religion_cat = ReligionCategorizer.categorize(religion_str)
```

---

## Phase 3: Split Notebook (2 hours)

### Recommended Notebook Split

Instead of 1 notebook with 65 cells:

**Notebook 1: `1_data_collection.ipynb`** (8 cells)
- Load senators from CSV
- Setup API configuration
- Validate HTML files exist and quality check
- Output: `senators_validated.csv`, `config_loaded.json`

**Notebook 2: `2_ground_truth_builder.ipynb`** (12 cells)
- Load validated senators
- Wikipedia scrape (birthdate, gender, race)
- Ballotpedia scrape (committee roles)
- Pew fuzzy match (religion)
- Manual review process
- Output: `senate_ground_truth.csv`

**Notebook 3: `3_llm_extraction.ipynb`** (10 cells)
- Load ground truth & HTML files
- Setup LLM client
- Run extraction with resume logic
- Parse results to structured format
- Output: `extractions_raw.json`, `extractions_parsed.csv`

**Notebook 4: `4_analysis_evaluation.ipynb`** (15 cells)
- Load predictions & ground truth
- Apply evaluation metrics (from new module)
- Generate detailed scores
- Summary statistics
- Output: `evaluation_results.csv`, `metrics_summary.json`

**Notebook 5: `5_final_report.ipynb`** (10 cells)
- Load evaluation results
- Create visualizations
- Generate comparison tables
- Export HTML/PDF report

---

## Phase 4: CLI Consolidation (1.5 hours)

**Create**: `LLMPersonalInfoExtraction/cli/__init__.py`

Consolidate extraction + evaluation entry points

---

## Phase 5: Data Validation (1.5 hours)

**Create**: `LLMPersonalInfoExtraction/schemas/senator.py`

Add Pydantic models for type safety

---

## Phase 6: Testing (2 hours)

**Create**: `tests/` directory with pytest suite

---

## Implementation Checklist

### Phase 1 ✅ (Priority)
- [ ] Create `html_utils.py`
- [ ] Update 3 files to use new module
- [ ] Create `PathManager`
- [ ] Update `main.py`, `evaluate.py` to use paths
- [ ] Create logging setup
- [ ] Replace critical print() with logger

### Phase 2 (High Priority)
- [ ] Extract metrics to module
- [ ] Reduce Cell 39 (654→100 lines)
- [ ] Verify notebook still runs

### Phase 3 (Medium Priority)
- [ ] Copy notebook to template
- [ ] Split into 5 focused notebooks
- [ ] Test each notebook independently

### Phase 4+ (Nice to Have)
- [ ] Refactor CLI
- [ ] Add Pydantic schemas
- [ ] Add pytest suite
- [ ] Add CI/CD pipeline

---

## Testing After Each Phase

```bash
# Phase 1: Ensure imports work
python -c "from LLMPersonalInfoExtraction.utils.html_utils import extract_readable_text; print('✓')"
python -c "from LLMPersonalInfoExtraction.config.paths import PathManager; print('✓')"

# Phase 2: Ensure notebook still runs
jupyter nbconvert --to notebook --execute senate_llm_pipeline.ipynb

# Phase 3: Test split notebooks
jupyter nbconvert --to notebook --execute 1_data_collection.ipynb
jupyter nbconvert --to notebook --execute 2_ground_truth_builder.ipynb
```

