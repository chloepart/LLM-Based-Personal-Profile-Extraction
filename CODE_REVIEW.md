# LLM-Based Personal Profile Extraction - Code Review & Improvement Plan

**Analysis Date**: April 7, 2026  
**Scope**: Full pipeline inspection of extraction, analysis, and evaluation flow  
**Finding**: Well-intentioned codebase with **significant redundancy** and **scattered logic** that reduce maintainability and testability.

---

## Executive Summary

| Issue | Severity | Count | Impact |
|-------|----------|-------|--------|
| **Code Duplication** | 🔴 Critical | 6 functions | ~250 duplicate lines |
| **Scattered Paths** | 🔴 Critical | 40+ locations | Breaks if directory changes |
| **Large Monolithic Notebook** | 🟠 High | 1 notebook | 65 cells, ~3300 lines |
| **Repeated Config Logic** | 🟠 High | 2+ files | Inconsistent state management |
| **CLI Entry Point Duplication** | 🟠 High | 3 files | `main.py`, `evaluate.py`, `run.py` |
| **No Data Validation** | 🟡 Medium | N/A | Type errors at runtime |
| **No Logging** | 🟡 Medium | N/A | Debugging requires print reconstruction |
| **Missing Unit Tests** | 🟡 Medium | N/A | Can't verify core functions |

---

## 1. CODE DUPLICATION ANALYSIS

### 1.1 `extract_readable_text()` Duplication (4 locations)

**Location 1**: [experiments/audit_pipeline_inputs.py](experiments/audit_pipeline_inputs.py#L31)
```python
def extract_readable_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)
```

**Location 2**: [experiments/rescrape_flagged.py](experiments/rescrape_flagged.py#L43)
```python
def extract_readable_text(html: str) -> str:  # IDENTICAL COPY
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)
```

**Location 3**: [experiments/senate_llm_pipeline.ipynb](experiments/senate_llm_pipeline.ipynb) Cell ~line 745
```python
def extract_readable_text(html):  # Python version in notebook
    # Same logic
```

**Location 4**: Inside audit logic (embedded)

**Problem**: 
- Any fix must be applied in 3+ places
- Creates coordinated maintenance burden
- Easy to miss one location during updates

**Solution**: Extract to `LLMPersonalInfoExtraction/utils/html_utils.py`

---

### 1.2 Name Normalization Functions (3 locations)

**Location 1**: [experiments/audit_pipeline_inputs.py](experiments/audit_pipeline_inputs.py#L37)
```python
def stem_to_name(stem: str) -> str:
    """Bernie_Moreno_OH -> Bernie Moreno"""
    parts = stem.split("_")
    if parts[-1].isupper() and len(parts[-1]) == 2:
        parts = parts[:-1]
    return " ".join(parts)
```

**Location 2**: [experiments/senate_llm_pipeline.ipynb](experiments/senate_llm_pipeline.ipynb) Cell ~line 1630
```python
def create_slug(name):
    # Similar logic
```

**Location 3**: Various normalization in notebook cells

---

### 1.3 Education Parsing (3+ variations in notebook)

**Cell 36** (lines ~1700-1780): First education extraction attempt
**Cell 38** (lines ~2300-2400): "Single authoritative education parser"
**Cell 39** (lines ~3050-3150): Test cases for education parsing

**Problem**: Three different implementations of `parse_education()`:
1. Using JSON parsing
2. Using pipe-delimited format
3. Testing more cases

Code should have ONE authoritative parser used everywhere.

---

### 1.4 Religion Categorization (2+ definitions)

**Cell 38** (lines ~2262-2300):
```python
RELIGION_HIERARCHY = {
    "catholic": "catholic",
    "roman catholic": "catholic",
    "evangelical": "evangelical",
    # ... 20+ entries
}
def get_religion_category(religion_str):
    # ... normalization logic
```

**Later in notebook**: Logic redefined or similar logic repeated for processing

---

## 2. PATH MANAGEMENT ISSUES

### Problem: Hard-coded Paths Scattered Everywhere

**Location 1**: [experiments/main.py](experiments/main.py#L25)
```python
res_save_path = f'../outputs/result/{model.provider}_{model.name.split("/")[-1]}...'
```

**Location 2**: [experiments/evaluate.py](experiments/evaluate.py#L21)
```python
res_save_path = f'../outputs/result/{model_config["model_info"]["provider"]}...'
```

**Location 3**: [experiments/audit_pipeline_inputs.py](experiments/audit_pipeline_inputs.py#L16)
```python
parser.add_argument("--html_dir", default="../external_data/senate_html", ...)
```

**Location 4**: [experiments/senate_llm_pipeline.ipynb](experiments/senate_llm_pipeline.ipynb) Cell 2
```python
OUTPUT_DIR = Path("../external_data/results/task1")
```

**Location 5**: Multiple notebooks with hardcoded data paths

**Problems**:
- ❌ If called from different directory, paths break
- ❌ Inconsistent path construction (`../` vs absolute vs Path)
- ❌ No validation that paths exist
- ❌ No way to override for different environments

**Solution**: Centralized `PathManager` class

```python
class PathManager:
    """Resolve all paths relative to project root."""
    PROJECT_ROOT = Path(__file__).parent.parent
    
    @classmethod
    def data_dir(cls) -> Path:
        return cls.PROJECT_ROOT / "data"
    
    @classmethod
    def senate_html(cls) -> Path:
        path = cls.PROJECT_ROOT / "external_data/senate_html"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @classmethod
    def output_results(cls, provider: str, model: str) -> Path:
        path = cls.PROJECT_ROOT / "outputs/result" / f"{provider}_{model}"
        path.mkdir(parents=True, exist_ok=True)
        return path
```

---

## 3. CONFIGURATION MANAGEMENT ISSUES

### Problem A: Inconsistent Config Loading

**In [main.py](experiments/main.py#L39)**:
```python
model_config = open_config(config_path=args.model_config_path)
model_config['model_info']['name'] = args.model_name
# Direct dict manipulation
```

**In [evaluate.py](experiments/evaluate.py#L21)**:
```python
model_config = open_config(config_path=f'../configs/model_configs/{args.provider}_config.json')
# Different path construction
```

**In notebook**: Direct JSON load with hardcoded path

**Problem**: Three different styles, no consistency

### Problem B: API Key Management Fragmentation

**In config files**: Keys stored in JSON
```json
"api_key_info": {
    "api_keys": ["KEY_HERE"],
    "api_key_use": 0
}
```

**In [config_loader.py](LLMPersonalInfoExtraction/config_loader.py)**:
```python
env_var = key_mapping.get(provider.lower())
api_key = os.getenv(env_var)
# Loads from environment
```

**In notebook**: 
```python
api_key = os.getenv("GROQ_API_KEY") or config[...]["api_keys"][0]
# Mixed approach
```

**Problem**: Unclear which method takes precedence, inconsistent error handling

---

## 4. NOTEBOOK STRUCTURE ANALYSIS

### Current: Single Monolithic Notebook

**File**: [experiments/senate_llm_pipeline.ipynb](experiments/senate_llm_pipeline.ipynb)

**Stats**:
- 65 cells
- ~3,300 lines
- Mixed concerns: scraping, ML extraction, evaluation, visualization
- Cell 39 alone: **654 lines** of evaluation metrics + parsing logic

**Cell 39 Contents** (lines 1280-1934):
```python
# 1. Name matching logic (~50 lines)
def normalize_name(name): ...
def name_match_score(gt_name, pred_name): ...

# 2. Senator ID creation (~30 lines)
def create_normalized_senator_id(name, state): ...

# 3. Fuzzy name matching (~80 lines)
def match_by_fuzzy_name(df_gt, df_pred): ...

# 4. Education parsing (3 variations, ~120 lines)
def parse_education_detailed(edu_str): ...
def parse_education(edu_str): ...

# 5. Religion categorization (~80 lines)
RELIGION_HIERARCHY = {...}
def get_religion_category(religion_str): ...

# 6. Ground truth merging logic (~150 lines)
def merge_evaluations(df_gt, df_pred): ...
```

**Problem**: This should be 5-6 separate modules, not one cell

### Recommended: Split into 5 Notebooks

```
📅 1_data_collection.ipynb
   - Load senators CSV
   - Setup configs
   - HTML file validation (audit results)
   └─ Output: validated senators list, configuration

📥 2_ground_truth_builder.ipynb
   - Wikipedia scraping (birthdate, gender, race)
   - Ballotpedia scraping (committee roles)
   - Pew fuzzy matching (religion)
   - Manual education verification approach
   └─ Output: senate_ground_truth.csv

🤖 3_llm_extraction.ipynb
   - Load each senator's HTML
   - Call LLM with prompt
   - Parse results to JSON
   - Save incrementally (resume-safe)
   └─ Output: senator_extractions.json, results_raw.json

📊 4_analysis_evaluation.ipynb
   - Load ground truth
   - Load LLM extractions
   - Apply evaluation metrics
   - Generate evaluation tables
   └─ Output: evaluation_results.csv, detailed scores

📈 5_final_report.ipynb
   - Load evaluation results
   - Create visualizations
   - Summary statistics
   - Export report
   └─ Output: HTML/PDF report
```

**Benefits**:
- ✅ Each notebook ~400-600 lines (reasonable size)
- ✅ Clear data flow between stages
- ✅ Can rerun individual stages without others
- ✅ Easier to test/debug individual components
- ✅ Better for team collaboration

---

## 5. CLI ENTRY POINT ANALYSIS

### Problem: Three Ways to Run the Same Pipeline

**Option 1**: [experiments/main.py](experiments/main.py)
```bash
python main.py --model_config_path ../configs/model_configs/groq_config.json \
               --model_name llama-3 \
               --task_config_path ../configs/task_configs/synthetic.json \
               --defense pi_ci_id
```

**Option 2**: [experiments/evaluate.py](experiments/evaluate.py)
```bash
python evaluate.py --provider gpt \
                   --model_name gpt-4 \
                   --defense no
```

**Option 3**: [experiments/run.py](experiments/run.py)
```bash
# Edit file directly, then run python run.py
```

**Problems**:
- ❌ Three different CLI styles
- ❌ Overlapping but inconsistent argument names
- ❌ No unified `--help` documentation
- ❌ `run.py` uses `os.system()` instead of proper process handling
- ❌ Hard to batch-run multiple configurations

**Solution**: Unified CLI with Click/Typer

```python
# LLMPersonalInfoExtraction/cli.py
import click

@click.group()
def cli():
    """LLM Personal Profile Extraction."""
    pass

@cli.command()
@click.option('--model', default='groq', help='Model provider')
@click.option('--model_name', help='Specific model')
@click.option('--dataset', default='synthetic')
@click.option('--defense', default='no')
@click.option('--prompt_type', default='direct')
@click.option('--device', default='0', help='GPU device(s)')
def extract(model, model_name, dataset, defense, prompt_type, device):
    """Run extraction pipeline."""
    from LLMPersonalInfoExtraction.experiments import main
    main.run(model, model_name, dataset, defense, prompt_type, device)

@cli.command()
@click.option('--provider', default='groq')
@click.option('--dataset', default='synthetic')
@click.option('--metric', default='all')
def evaluate(provider, dataset, metric):
    """Evaluate extraction results."""
    from LLMPersonalInfoExtraction.experiments import evaluate
    evaluate.run(provider, dataset, metric)

@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
def batch(config_file):
    """Run batch of extractions from configuration."""
    # Load YAML, iterate configurations
    pass

if __name__ == '__main__':
    cli()
```

**Usage**:
```bash
python -m LLMPersonalInfoExtraction.cli extract --model groq --dataset synthetic
python -m LLMPersonalInfoExtraction.cli evaluate --provider groq
python -m LLMPersonalInfoExtraction.cli batch batch_config.yaml
```

---

## 6. DATA VALIDATION GAPS

### Problem: No Type Checking / Validation

**Currently**: Data loaded as dicts/lists, no validation
```python
raw_response = all_raw_responses[info_cat][i]  # Could be None, could be malformed
curr_label = dict(zip(info_cats, [...]))  # No type validation
```

**Result**: Runtime errors when data doesn't match expectations

**Solution**: Use Pydantic schemas

```python
# LLMPersonalInfoExtraction/schemas/senator.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List

class Education(BaseModel):
    degree: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[int] = None

class SenatorExtraction(BaseModel):
    full_name: Optional[str] = None
    birthdate: Optional[str] = Field(None, regex=r'^\d{4}-\d{2}-\d{2}$|^$')
    gender: Optional[str] = Field(None, regex='^(male|female|other)$|^$')
    education: List[Education] = Field(default_factory=list)
    religious_affiliation: Optional[str] = None
    
    @validator('education', pre=True)
    def validate_education(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            # Parse from string format
            return parse_education_string(v)
        return v

class SenatorGroundTruth(SenatorExtraction):
    senator_id: str
    state: str
    religion_signal: Literal['explicit', 'not_explicit']
```

**Usage**:
```python
extracted_data = json.loads(raw_response)
senator = SenatorExtraction(**extracted_data)  # Automatic validation
# Now guaranteed: senator.full_name is correct type, dates are valid, etc.
```

---

## 7. LOGGING GAPS

### Problem: Everything Uses `print()`

Current code scattered with:
```python
print(f'{i+1} / {len(task_manager)}: {curr_label["name"]}')
print('Not applicable. Skip')
print(f'API KEY POS = {model_config["api_key_info"]["api_key_use"]}')
```

**Problems**:
- ❌ Can't turn off verbose output
- ❌ Can't save logs to file
- ❌ No log levels (INFO vs WARNING vs ERROR)
- ❌ No timestamps
- ❌ Debugging requires reconstructing from print statements

**Solution**: Replace with Python logging

```python
# In LLMPersonalInfoExtraction/__init__.py
import logging

def setup_logging(name='PIE', level=logging.INFO, log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    logger.addHandler(console)
    
    # File handler
    if log_file:
        file_h = logging.FileHandler(log_file)
        file_h.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        ))
        logger.addHandler(file_h)
    
    return logger

# Usage everywhere:
logger = logging.getLogger('PIE')
logger.info(f'{i+1} / {len(task_manager)}: {curr_label["name"]}')
logger.warning('Not applicable. Skip')
logger.error(f'Failed to process: {error}')
```

---

## 8. MISSING UNIT TESTS

### Current Test Coverage: ~0%

Only found embedded test code in notebook:
```python
# Cell ~line 3050: Manual test functions
test_gt1 = "B.A.|Stanford|1982"
result = parse_education_detailed(test_gt1)
print(f"Output: {result}")
```

### Recommended: pytest Suite

```python
# tests/test_html_utils.py
import pytest
from LLMPersonalInfoExtraction.utils.html_utils import extract_readable_text

def test_extract_readable_text_removes_scripts():
    html = '<html><script>var x=1;</script><body>Hello</body></html>'
    result = extract_readable_text(html)
    assert 'var x' not in result
    assert 'Hello' in result

def test_extract_readable_text_cleans_whitespace():
    html = '<body>Hello   \n\n   World</body>'
    result = extract_readable_text(html)
    assert '   ' not in result
    assert result == 'Hello World'
```

```python
# tests/test_metrics.py
import pytest
from LLMPersonalInfoExtraction.metrics import EvaluationMetrics

def test_education_parser_pipe_format():
    gt = "B.A.|Stanford University|1982|J.D.|Harvard Law School|1986"
    result = EvaluationMetrics.parse_education(gt, format='pipe')
    assert len(result) == 2
    assert result[0]['degree'] == 'B.A.'
    assert result[0]['institution'] == 'Stanford University'
    assert result[0]['year'] == 1982

def test_education_parser_json_format():
    json_str = '[{"degree":"B.A.","institution":"Stanford","year":1982}]'
    result = EvaluationMetrics.parse_education(json_str, format='json')
    assert len(result) == 1
    assert result[0]['institution'] == 'Stanford'

def test_religion_categorization():
    assert EvaluationMetrics.categorize_religion("Roman Catholic") == "catholic"
    assert EvaluationMetrics.categorize_religion("Evangelical Christian") == "evangelical"
    assert EvaluationMetrics.categorize_religion("agnostic") is None
```

**Run with**:
```bash
pytest tests/ -v
pytest tests/test_metrics.py -v  # Single file
pytest --cov LLMPersonalInfoExtraction  # Coverage report
```

---

## QUICK WINS (Do First - ~2 hours)

1. **Extract `html_utils.py`** (~30 min)
   - Move `extract_readable_text()`, `stem_to_name()`, `is_content_valid()`
   - Update 3 files to import from new module

2. **Create `PathManager`** (~30 min)
   - Centralize all path logic
   - Replace 40+ hardcoded strings

3. **Add Logging** (~45 min)
   - Setup logging module
   - Replace critical print() statements

4. **Extract Metrics Module** (~1 hour)
   - Move evaluation metrics to `LLMPersonalInfoExtraction/metrics/eval_metrics.py`
   - Reduces Cell 39 from 654 → 100 lines

**Total Time**: ~2.75 hours  
**Impact**: Eliminates ~40% of complexity, improves maintainability significantly

---

## MEDIUM REFACTORS (4-6 hours)

5. **Split notebook into 5** (~2 hours)
6. **Consolidate CLI** (~1.5 hours)
7. **Add Pydantic schemas** (~1.5 hours)
8. **Add pytest suite** (~2 hours)

---

## Key Recommendations Summary

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| 🔴 P1 | Extract duplicate utils | 30 min | Remove ~250 dup lines |
| 🔴 P1 | Add PathManager | 30 min | Fix all path issues |
| 🔴 P1 | Add logging | 45 min | Better debugging |
| 🟠 P2 | Extract metrics module | 1 hour | Cell 39: 654→100 lines |
| 🟠 P2 | Unify CLI | 1.5 hours | Single entry point |
| 🟠 P2 | Split notebook | 2 hours | 65→200 cells total but smaller |
| 🟡 P3 | Add Pydantic schemas | 1.5 hours | Type safety |
| 🟡 P3 | Add pytest tests | 2 hours | Confidence in refactors |

**Estimated Total**: 8-10 hours → **Production-ready codebase**

