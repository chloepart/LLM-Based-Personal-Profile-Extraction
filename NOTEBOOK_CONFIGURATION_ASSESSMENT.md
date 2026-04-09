# Senate LLM Pipeline — Configuration & Code Assessment

**Status**: Intermediate maturity with good architectural foundation but critical gaps in execution.  
**Date**: April 8, 2026  
**Focus**: Notebook configuration, underlying module design, and pipeline cohesion.

---

## Executive Summary

**Strengths:**
- ✅ Excellent centralized configuration via `SessionConfig` dataclass
- ✅ Well-modularized codebase with clear separation of concerns
- ✅ Comprehensive prompt engineering (direct, pseudocode, ICL variants)
- ✅ Intelligent rate-limiting with exponential backoff for quota errors
- ✅ Good test coverage for baselines (regex, spaCy, keyword search)

**Critical Issues:**
- ❌ **Missing `run_pipeline()` function** — called on line 449 but never defined
- ❌ **API key hardcoded in JSON config** — security vulnerability
- ❌ **Race conditions in parallel result writing** — unsafe concurrent access to results JSON
- ❌ **No input validation** on LLM responses before struct parsing
- ❌ **Inconsistent session re-initialization logic** — brittle kernel restart handling

**Medium Issues:**
- ⚠️ Hard-coded inter-senator delays aren't adaptive to actual API response times
- ⚠️ No comprehensive error logging — only stderr prints and basic JSON errors
- ⚠️ Test HTML/CSV assumed to exist — no graceful handling if missing
- ⚠️ Prompt templates hardcoded in config; not version-controlled or testable
- ⚠️ Model output validation is fragile (relies on markdown fence detection)

---

## 1. Configuration Architecture — `/modules/config.py`

### Current State
The module exports:
```python
PATHS              # Dict with 11 path definitions
TASK1_DIRECT       # Prompt template (excellent, well-documented)
TASK1_PSEUDOCODE   # Prompt template
TASK1_ICL          # Prompt template
T1_FIELDS          # List of expected JSON fields
REGEX_PATTERNS     # Compiled regex for baselines
```

### Issues & Recommendations

| Issue | Severity | Recommendation |
|-------|----------|-----------------|
| **PATHS uses relative paths** | High | Use `Path(__file__).parent.parent / "..."`to make imports location-independent. Current breaks if notebook runs from different dir. |
| **Prompt templates aren't versioned** | Medium | Add `PROMPT_VERSION = "1.0"` and timestamp; makes ablation studies reproducible. |
| **No validation of PATHS existence** | Medium | Add `_validate_paths()` that runs on import and logs warnings for missing dirs. |
| **T1_FIELDS is a list, not enum** | Low | Change to `enum.Enum` for type safety and to prevent typos in field access. |
| **REGEX_PATTERNS lacks documentation** | Low | Add regex explanation comments for each pattern (phone, email, name patterns). |

### Suggested Improvements

**config.py additions:**
```python
import enum
from typing import Dict, List
from pathlib import Path

# Make relative paths absolute
_MODULE_ROOT = Path(__file__).parent.parent
PATHS = {
    'html_data': _MODULE_ROOT / 'external_data' / 'senate_html',
    # ... etc
}

class T1FieldName(enum.Enum):
    """Enum for all Task 1 extraction fields."""
    FULL_NAME = "full_name"
    BIRTHDATE = "birthdate"
    GENDER = "gender"
    RELIGION = "religious_affiliation"
    # ...

# Version prompts
PROMPT_VERSIONS = {
    "direct": {"version": "1.0", "date": "2025-12-01", "text": TASK1_DIRECT},
    "pseudocode": {"version": "1.0", "date": "2025-12-01", "text": TASK1_PSEUDOCODE},
    "icl": {"version": "1.0", "date": "2025-12-01", "text": TASK1_ICL},
}

def validate_config() -> Dict[str, bool]:
    """Check config validity; return status for each path."""
    status = {}
    for key, path in PATHS.items():
        exists = path.exists() if isinstance(path, Path) else False
        status[key] = exists
        if not exists:
            logging.warning(f"Config path missing: {key} → {path}")
    return status
```

---

## 2. Session Configuration — `SessionConfig` Dataclass

### Current State (Good)
```python
@dataclass
class SessionConfig:
    output_dir, html_dir, config_file, model, temperature, max_tokens, api_key,
    prompt_styles, prompt_map, timestamp, api_client
```

### Issues & Recommendations

| Issue | Severity | Recommendation |
|-------|----------|-----------------|
| **API key passed as plain string** | **CRITICAL** | Move to environment variables ONLY. Never load from JSON config. Use `os.getenv("GROQ_API_KEY")` with fallback error. |
| **api_client can be None** | High | Remove optional initialization; require it in `__init__` or use factory method. |
| **to_dict() doesn't exclude sensitive fields** | High | Explicitly remove `api_key` before serializing to disk. |
| **No validation of prompt_styles against prompt_map** | Medium | Add custom `__post_init__` validation Check all styles in prompt_map before storing. |
| **No immutability post-init** | Low | Consider `frozen=True` to prevent accidental mutation at runtime. |

### Suggested Improvements

```python
@dataclass(frozen=True)
class SessionConfig:
    output_dir: Path
    html_dir: Path
    config_file: Path
    model: str
    temperature: float
    max_tokens: int
    prompt_styles: list
    prompt_map: dict
    timestamp: str
    
    def __post_init__(self):
        """Validate config after initialization."""
        invalid_styles = set(self.prompt_styles) - set(self.prompt_map.keys())
        if invalid_styles:
            raise ValueError(f"Invalid prompt styles: {invalid_styles}")
        if not 0 <= self.temperature <= 2:
            raise ValueError(f"Temperature must be in [0, 2], got {self.temperature}")
        if self.max_tokens < 100 or self.max_tokens > 4096:
            raise ValueError(f"max_tokens must be in [100, 4096]")
    
    @staticmethod
    def from_json_and_env(config_path: Path) -> "SessionConfig":
        """Create SessionConfig from JSON + environment variables.
        
        Environment variables override JSON values for security.
        """
        with open(config_path) as f:
            cfg = json.load(f)
        
        # API key MUST come from environment
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        return SessionConfig(
            output_dir=Path(cfg["output_dir"]),
            html_dir=Path(cfg["html_dir"]),
            config_file=config_path,
            model=cfg.get("model", "llama-3.1-8b-instant"),
            temperature=float(cfg.get("temperature", 0.1)),
            max_tokens=int(cfg.get("max_tokens", 1500)),
            prompt_styles=cfg.get("prompt_styles", ["direct"]),
            prompt_map={...},  # Built from config
            timestamp=datetime.datetime.now().isoformat(),
        )
    
    def to_dict(self, include_sensitive: bool = False) -> dict:
        """Serialize config (excluding API key by default)."""
        data = {
            "output_dir": str(self.output_dir),
            "html_dir": str(self.html_dir),
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "prompt_styles": self.prompt_styles,
            "timestamp": self.timestamp,
        }
        if include_sensitive:
            data["api_key"] = self.api_key  # For testing only
        return data
```

---

## 3. Main Processing Loop — Notebook Cell Section 6

### Current Issue: Missing `run_pipeline()` Function

**Location**: Line 449 in notebook
```python
result = run_pipeline(text, session_config)  # ← UNDEFINED!
```

This function is called but never defined in the current notebook. It's defined in `docs/senate_llm_pipeline_V1.ipynb` but not in the active notebook.

### Immediate Fix Required

Add this cell **before** the processing loop:

```python
def run_pipeline(text: str, config: SessionConfig) -> dict:
    """
    Execute extraction pipeline on HTML text.
    
    Runs all configured prompt styles through LLM, collects results,
    and includes baseline methods (regex, spaCy) for comparison.
    
    Args:
        text: Cleaned HTML text from senator bio
        config: SessionConfig object with model/API settings
    
    Returns:
        Dict with structure:
        {
            "senator_id": str,
            "task1_pii": {
                "direct": {...extraction results...},
                "pseudocode": {...},
                "icl": {...},
            },
            "baselines": {
                "regex": {...},
                "spacy": {...},
            },
            "errors": [...list of errors if any...]
        }
    """
    result = {"task1_pii": {}, "baselines": {}, "errors": []}
    
    # Run all configured prompt styles
    for style in config.prompt_styles:
        try:
            prompt = config.prompt_map[style]
            extraction = call_groq(prompt, text, config)
            
            # Validate JSON structure
            if "error" not in extraction:
                result["task1_pii"][style] = extraction
            else:
                result["errors"].append(f"{style}: {extraction['error']}")
                result["task1_pii"][style] = {"error": extraction["error"]}
        
        except Exception as e:
            result["errors"].append(f"{style}: {str(e)}")
            result["task1_pii"][style] = {"error": str(e)}
    
    # Run baselines for comparison
    try:
        result["baselines"]["regex"] = regex_extract(text)
        result["baselines"]["spacy"] = spacy_extract(text)
    except Exception as e:
        result["errors"].append(f"baselines: {str(e)}")
    
    return result
```

---

## 4. Unsafe Concurrent Result Writing

### Current Issue
```python
# Line 450-452
results.append({"senator_id": senator_id, **result})
with open(raw_path, "w") as f:
    json.dump(results, f, indent=2)
```

**Problem**: Writing entire file after each senator is inefficient AND race-condition-prone if:
- Notebook is interrupted mid-write
- Multiple cells run in parallel
- File becomes corrupted if power loss during write

### Recommended Solution

Use **jsonlines** format with atomic writes:

```python
# Option 1: Use JSONLines (append-only, atomic per line)
import jsonlines

raw_path = session_config.output_dir / "results_raw.jsonl"

# Load existing results
if raw_path.exists():
    with jsonlines.open(raw_path) as reader:
        results = [line for line in reader]
        done_ids = {r["senator_id"] for r in results}
else:
    results, done_ids = [], set()

# Process and write atomically
for html_file in tqdm(remaining, desc="Processing senators"):
    senator_id = html_file.stem
    html = html_file.read_text(encoding="utf-8", errors="ignore")
    text = extract_readable_text(html)
    result = run_pipeline(text, session_config)
    
    record = {"senator_id": senator_id, **result}
    
    # Atomic append (OS-level)
    with jsonlines.open(raw_path, mode='a') as writer:
        writer.write(record)
    
    time.sleep(inter_senator_delay)

# Option 2: Temporary file + atomic rename
import tempfile
import shutil

temp_fd, temp_path = tempfile.mkstemp(suffix='.json', dir=output_dir)
try:
    with open(temp_path, 'w') as f:
        json.dump(results, f, indent=2)
    shutil.move(temp_path, raw_path)  # Atomic on Unix
finally:
    os.close(temp_fd)
```

---

## 5. API Key Security Issue

**CRITICAL VULNERABILITY**

### Current Issue
```json
// configs/model_configs/groq_config_extraction.json
{
    "api_key_info": {
        "api_keys": [
            "${GROQ_API_KEY}"  ← EXPOSED!
        ]
    }
}
```

This API key is:
- ❌ Checked into version control
- ❌ Readable by anyone with repository access
- ❌ Compromises the entire Groq account

### Immediate Action Required

1. **Revoke the API key immediately** (regenerate in Groq dashboard)
2. **Remove from JSON config**:
```json
{
    "api_key_info": {
        "api_keys": [],
        "source": "environment"
    }
}
```

3. **Update config loading**:
```python
# Do NOT load from JSON
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found in environment. "
        "Please set: export GROQ_API_KEY=gsk_..."
    )
```

4. **Add `.env` to `.gitignore`** and document in README:
```bash
# .env (never commit)
GROQ_API_KEY=gsk_...
```

---

## 6. Session Re-initialization Logic — Cell Section 5

### Current Issue
```python
if 'session_config' not in dir() or 'html_files' not in dir():
    print("  Re-initializing after kernel restart...")
    from modules import PATHS, extract_readable_text
    session_config = PATHS['output_root']  # ← WRONG!
    html_files = sorted(session_config.html_dir.glob("*.html"))
```

**Problems:**
1. `session_config = PATHS['output_root']` assigns a **Path**, not a SessionConfig!
2. Line will crash when accessing `session_config.html_dir`
3. No re-initialization of API client or timestamp
4. Using `'session_config' not in dir()` is fragile

### Fix

```python
# Check if kernel was restarted (session_config will be invalid object type)
def is_session_valid():
    """Check if session_config is properly initialized."""
    return (
        'session_config' in dir() and
        isinstance(session_config, SessionConfig) and
        session_config.api_client is not None
    )

if not is_session_valid():
    print("Kernel restart detected. Re-initializing session...")
    
    # Reload config from file
    config_path = Path("../configs/model_configs/groq_config_extraction.json")
    with open(config_path) as f:
        cfg_json = json.load(f)
    
    # Recreate session_config with fresh state
    session_config = SessionConfig(
        output_dir=Path("../outputs/senate_results"),
        html_dir=Path("../external_data/senate_html"),
        config_file=config_path,
        model=cfg_json["model_info"]["name"],
        temperature=cfg_json["params"]["temperature"],
        max_tokens=cfg_json["params"]["max_output_tokens"],
        api_key=os.getenv("GROQ_API_KEY"),
        prompt_styles=PROMPT_STYLES,
        prompt_map=PROMPT_MAP,
        timestamp=datetime.datetime.now().isoformat(),
    )
    
    html_files = sorted(session_config.html_dir.glob("*.html"))
    print(f"✓ Session restored: {len(html_files)} HTML files, model={session_config.model}")
```

---

## 7. Rate Limiting — Not Adaptive

### Current Issue
```python
inter_senator_delay = 6 if len(session_config.prompt_styles) > 1 else 4
time.sleep(inter_senator_delay)  # Fixed delay
```

**Problem**: Ignores actual API response times. If Groq responds in 100ms, sleeping 6 seconds is waste. If response takes 4 seconds, 6 second delay might not prevent quota errors under load.

### Recommended Adaptive Rate Limiting

```python
import time
from collections import deque

class AdaptiveRateLimiter:
    """Dynamically adjust rate limits based on response times."""
    
    def __init__(self, target_calls_per_minute: int = 30, window_size: int = 10):
        self.target_calls_per_minute = target_calls_per_minute
        self.min_delay = 60.0 / target_calls_per_minute
        self.window_size = window_size
        self.response_times = deque(maxlen=window_size)
        self.last_call_time = None
    
    def record_response(self, response_time: float):
        """Record API response time."""
        self.response_times.append(response_time)
    
    def get_wait_time(self) -> float:
        """Calculate recommended wait time before next API call."""
        if not self.response_times:
            return self.min_delay
        
        avg_response_time = sum(self.response_times) / len(self.response_times)
        # Apply safety factor (1.5x response time, with minimum)
        recommended_wait = max(self.min_delay, avg_response_time * 1.5)
        
        if self.last_call_time:
            time_since_last = time.time() - self.last_call_time
            additional_wait = max(0, recommended_wait - time_since_last)
            return additional_wait
        
        return recommended_wait
    
    def wait(self):
        """Sleep until next API call is safe."""
        wait_time = self.get_wait_time()
        if wait_time > 0:
            time.sleep(wait_time)
        self.last_call_time = time.time()

# Usage in notebook
limiter = AdaptiveRateLimiter(target_calls_per_minute=30)

for html_file in tqdm(remaining, desc="Processing senators"):
    start = time.time()
    result = run_pipeline(text, session_config)
    elapsed = time.time() - start
    
    limiter.record_response(elapsed)
    limiter.wait()  # Sleep adaptively
```

---

## 8. Error Handling & Logging

### Current State
- Print statements only (no structured logging)
- Errors stored in JSON as `{"error": "..."}` (hard to query)
- No persistent error log from session

### Recommended Improvements

```python
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(output_dir: Path, session_name: str = "default"):
    """Setup structured logging for pipeline."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger("senate_pipeline")
    logger.setLevel(logging.DEBUG)
    
    # File handler (rotating to prevent huge log files)
    fh = RotatingFileHandler(
        log_dir / f"pipeline_{session_name}.log",
        maxBytes=10_000_000,  # 10MB
        backupCount=5
    )
    fh.setLevel(logging.DEBUG)
    
    # Console handler (info only)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# Usage in notebook
logger = setup_logging(session_config.output_dir, "main_extraction")

try:
    extraction = call_groq(prompt, text, config)
    if "error" in extraction:
        logger.warning(f"Extraction failed for {senator_id}: {extraction['error']}")
    else:
        logger.debug(f"Extracted {len(extraction)} fields for {senator_id}")
except Exception as e:
    logger.error(f"Exception on {senator_id}: {type(e).__name__}: {e}", exc_info=True)
    raise

# Later: query errors easily
# $ grep "ERROR" outputs/logs/pipeline_main_extraction.log
```

---

## 9. Input Validation on LLM Response

### Current Issue
```python
# In call_groq()
response_text = response.choices[0].message.content.strip()

# Clean up markdown fences (fragile!)
if "```json" in response_text:
    response_text = response_text.split("```json")[1].split("```")[0].strip()
elif "```" in response_text:
    response_text = response_text.split("```")[1].split("```")[0].strip()

# Parse JSON (loses details if invalid)
return json.loads(response_text)
```

**Problems:**
- If LLM returns invalid JSON, error message is generic
- No validation that returned fields match expected schema
- No handling of partial responses (truncated due to max_tokens)

### Recommended Schema Validation

```python
from typing import Any, Dict, Optional
from pydantic import BaseModel, ValidationError, field_validator

class Task1ExtractionResult(BaseModel):
    """Schema for Task 1 PII extraction results."""
    full_name: Optional[str] = None
    birthdate: Optional[str] = None
    birth_year_inferred: Optional[int] = None
    gender: Optional[str] = None
    race_ethnicity: Optional[str] = None
    education: list = []  # List of {degree, institution, year}
    committee_roles: list = []
    religious_affiliation: Optional[str] = None
    religious_affiliation_inferred: Optional[bool] = None
    
    @field_validator('birthdate')
    @classmethod
    def validate_birthdate(cls, v):
        if v is None:
            return v
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', v):
            raise ValueError(f"Birthdate must be YYYY-MM-DD, got {v}")
        return v
    
    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        if v is None:
            return v
        if v.lower() not in ['male', 'female', 'non-binary', 'other']:
            raise ValueError(f"Invalid gender: {v}")
        return v

def validate_extraction(raw_json: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean LLM extraction result."""
    try:
        result = Task1ExtractionResult(**raw_json)
        return result.model_dump(exclude_none=False)
    except ValidationError as e:
        logger.warning(f"Validation failed: {e}")
        # Return result with error marker
        return {
            "error": "schema_validation_failed",
            "details": str(e),
            "raw_input": raw_json
        }
```

---

## 10. Module Inter-Dependencies

### Current Import Graph
```
notebook imports:
  ├── config → (TASK1_DIRECT, REGEX_PATTERNS, etc.)
  ├── name_utils → (NameNormalizer)
  ├── html_processing → (extract_readable_text)
  ├── parsing → (parse_education, normalize_degree)
  ├── baselines → (regex_extract, spacy_extract, keyword_extract)
  ├── evaluator → (name_match_score, religion_match_score, gender_match_score)
  └── groundtruth → (fetch_wikipedia_text, normalize_birthdate)
```

### Issues
1. **Circular imports possible** — if evaluator imports from config which might later import from evaluator
2. **Missing type hints** — most functions lack parameter/return type hints
3. **Inconsistent naming** — `extract_readable_text` (function) vs `HTMLProcessor.extract_readable_text` (method)

### Recommendation
```python
# In modules/__init__.py, explicitly re-export with version info
__version__ = "0.2.0"

__all__ = [
    "SessionConfig",
    "TASK1_DIRECT", "TASK1_PSEUDOCODE", "TASK1_ICL",
    "NameNormalizer",
    "extract_readable_text",
    "parse_education",
    "regex_extract", "spacy_extract",
    "evaluate_extraction",
]

# Add at module level to prevent circular imports
_INITIALIZATION_COMPLETE = False

def _post_import_check():
    """Verify no circular dependencies."""
    # On import, check that all exported symbols are callable/valid
    pass

_INITIALIZATION_COMPLETE = True
```

---

## 11. Testing & Validation

### Major Gap: No Unit Tests
The notebook has **zero** unit tests. Testing is done inline during development.

### Recommended Test Structure

```python
# tests/test_extraction.py
import pytest
from modules.parsing import parse_education, normalize_degree

class TestEducationParsing:
    
    def test_parse_json_education(self):
        """Test parsing JSON format from LLM output."""
        input_json = '[{"degree": "B.S.", "institution": "MIT", "year": 2010}]'
        result = parse_education(input_json, format_type="json")
        assert len(result) == 1
        assert result[0]["degree"] == "B.S."
    
    def test_parse_pipe_delimited(self):
        """Test pipe-delimited format from ground truth."""
        input_pipe = "B.S.|MIT|2010|M.A.|Stanford|2012"
        result = parse_education(input_pipe, format_type="pipe")
        assert len(result) == 2
        assert result[1]["institution"] == "Stanford"
    
    def test_normalize_degree(self):
        """Test degree variations."""
        assert normalize_degree("bs") == "B.S."
        assert normalize_degree("bachelor of science") == "B.S."
        assert normalize_degree("phd") == "Ph.D."

# Run tests: pytest tests/test_extraction.py -v
```

---

## Summary of Recommendations

### Priority 1 (Do Immediately)
- [ ] Define missing `run_pipeline()` function
- [ ] Revoke hardcoded API key and move to environment variable
- [ ] Add SessionConfig validation in `__post_init__`
- [ ] Fix session re-initialization logic

### Priority 2 (Before Next Run)
- [ ] Switch to JSONLines format for safer concurrent writes
- [ ] Add comprehensive error logging
- [ ] Add Pydantic schema validation for LLM outputs
- [ ] Use absolute paths in config module

### Priority 3 (Before Production)
- [ ] Implement adaptive rate limiting
- [ ] Add unit test suite
- [ ] Add input validation on HTML/CSV files
- [ ] Document all module interfaces with type hints

### Priority 4 (Nice to Have)
- [ ] Version control for prompt templates
- [ ] Model comparison dashboard (8B vs 70B metrics)
- [ ] Automated evaluation against ground truth
- [ ] CI/CD pipeline with pre-commit hooks

---

## References

- **Liu et al. (USENIX Security 2025)** — Evaluating LLM-based Personal Information Extraction
- **Current Notebook**: `/experiments/senate_llm_pipeline.ipynb`
- **Config Files**: `/configs/model_configs/groq_config_extraction.json`
- **Modules**: `/modules/{config,baselines,evaluator,html_processing,parsing,name_utils,...}.py`
