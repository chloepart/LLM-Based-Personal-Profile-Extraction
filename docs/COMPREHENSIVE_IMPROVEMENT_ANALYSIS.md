# Comprehensive Program Improvement Analysis
**LLM-Based Personal Profile Extraction Pipeline (v5)**

**Scope**: Full inspection of notebook logic, code organization, and folder structure  
**Date**: April 7, 2026  
**Status**: Active development with significant refactoring opportunities  

---

## EXECUTIVE SUMMARY

### Quantified Issues

| Category | Issue | Count | Lines Affected | Priority |
|----------|-------|-------|-----------------|----------|
| **Code Duplication** | Extract readable_text() | 4 locations | ~120 lines | 🔴 Critical |
| **Code Duplication** | Name normalization functions | 3+ locations | ~150 lines | 🔴 Critical |
| **Code Duplication** | Education parsing (3 variants) | 3 locations | ~200 lines | 🔴 Critical |
| **Code Duplication** | Religion categorization | 2+ locations | ~100 lines | 🔴 Critical |
| **Scattered Config** | Paths hardcoded | 40+ locations | Entire codebase | 🔴 Critical |
| **Evaluation Redundancy** | Metric calculation repeated | 3 times (per style) | ~500 lines | 🟠 High |
| **Monolithic Structure** | Single large notebook | 1 file | ~3,300 lines | 🟠 High |
| **Markdown Issues** | Verbose/redundant headers | 8+ instances | ~50 lines | 🟡 Medium |
| **Folder Organization** | Scripts scattered | 5 locations | 3 scripts | 🟡 Medium |
| **Data Validation** | No input validation | N/A | Throughout | 🟡 Medium |

---

## PART 1: NOTEBOOK-LEVEL REDUNDANCIES & IMPROVEMENTS

### 1.1 Code Duplication — HTML Text Extraction

**Severity**: 🔴 CRITICAL  
**Impact**: Any fix to HTML preprocessing must be applied to 4+ locations

#### Current State: 4 Identical Implementations

**Location 1** — `modules/html_processing.py` (centralized, good):
```python
def extract_readable_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)
```

**Location 2** — Notebook Section 3 (line ~193):
```python
# Test HTML preprocessing with imported extract_readable_text
sample = html_files[0]
text   = extract_readable_text(sample.read_text(encoding="utf-8", errors="ignore"))
# ✓ This one correctly imports from modules
```

**Location 3** — Notebook Section 8 (line ~1089 in ground truth builder):
```python
soup = BeautifulSoup(response.content, "html.parser")
for tag in soup(["script", "style", "nav", "footer"]):  # ← DUPLICATED
    tag.decompose()
wiki_text = soup.get_text(separator=" ", strip=True)
```

**Location 4** — Notebook Section 8b Education extraction (line ~1500):
```python
# Fallback Wikipedia fetch
soup = BeautifulSoup(response.content, "html.parser")
for tag in soup(["script", "style", "nav", "footer"]):  # ← DUPLICATED AGAIN
    tag.decompose()
wiki_text = soup.get_text(separator=" ", strip=True)[:5000]
```

#### Issue Analysis

- ✅ `modules/html_processing.py` correctly centralizes
- ❌ Ground truth builder (Section 8) **redeclares instead of importing**
- ❌ Education extractor (Section 8b) **has its own copy**
- ❌ Maintenance nightmare: if encoding strategy changes, must update 3+ places

#### Recommendation

**Immediate Action**:
```python
# In notebook Section 8 (line ~595), REPLACE manual extraction with:
from modules import extract_readable_text

# Line ~1089 becomes:
wiki_text = extract_readable_text(response.text)  # Reuse!

# Line ~1500 becomes:
wiki_text = extract_readable_text(response.text)[:5000]
```

**Prevention**:
- Add linting rule to find hardcoded BeautifulSoup decompose patterns
- Document centralized HTML utilities in notebook comments

---

### 1.2 Code Duplication — Name Normalization & URL Generation

**Severity**: 🔴 CRITICAL  
**Impact**: Inconsistent senator identification across sections

#### Current State: 3 Implementations

**Location 1** — Notebook Section 8 (line ~817):
```python
senators["wikipedia_url"] = senators["name"].apply(
    NameNormalizer.create_wikipedia_url  # ✓ Good: uses modules
)
```

**Location 2** — Notebook Section 8b Education (line ~1138):
```python
senators["wikipedia_url"] = senators["name"].apply(
    NameNormalizer.create_wikipedia_url  # ✓ Also imports correctly
)
```

**Location 3** — Notebook Section 8 (line ~685):
```python
def create_slug(name):
    """Create URL slug for Wikipedia/Ballotpedia"""
    slug = re.sub(r'\s+[A-Z]\.\s*', ' ', name).strip()
    slug = slug.replace(" ", "_")
    # ... manual logic instead of using NameNormalizer
```

#### Issue Analysis

- ✅ URL creation is in `modules/name_utils.py` (NameNormalizer class)
- ❌ Some cells use it (`NameNormalizer.create_wikipedia_url`)
- ❌ Other cells redefine `create_slug()` manually
- ❌ Creates divergence: which implementation is "correct"?

#### Recommendation

**Replace all manual name handling with centralized**:
```python
# Remove lines ~685-700 (manual create_slug function)
# Use NameNormalizer consistently throughout:
from modules import NameNormalizer

# Instead of:
senators["slug"] = senators["name"].apply(create_slug)

# Use:
senators["wikipedia_url"] = senators["name"].apply(
    NameNormalizer.create_wikipedia_url
)
```

---

### 1.3 Code Duplication — Education Parsing (3 Variants)

**Severity**: 🔴 CRITICAL  
**Impact**: ~200 lines of redundant parsing logic; inconsistent results

#### Current State: Three Different Implementations

**Variant 1** — Cell 36 (line ~1700):
```python
def parse_education(s):
    if not s or (isinstance(s, float) and pd.isna(s)):
        return ""
    try:
        s_str = str(s).strip()
        if s_str.startswith("["):
            edu_list = ast.literal_eval(s_str)
            institutions = [
                str(edu["institution"])
                for edu in edu_list
                if isinstance(edu, dict) and edu.get("institution")
            ]
            return "|".join(institutions)
    except Exception:
        pass
    return str(s)
```

**Variant 2** — Cell 37 (line ~1850):
```python
def parse_education_detailed(s, format_type="auto"):
    """Parse education data into structured list of (degree, institution, year) tuples."""
    if not s or (isinstance(s, float) and pd.isna(s)):
        return []
    
    s_str = str(s).strip()
    
    # Try JSON format first (LLM output)
    if s_str.startswith("["):
        try:
            edu_list = ast.literal_eval(s_str)
            education_items = []
            for edu in edu_list:
                if isinstance(edu, dict):
                    degree = str(edu.get("degree", "")).strip() if edu.get("degree") else None
                    institution = str(edu.get("institution", "")).strip() if edu.get("institution") else None
                    year = str(edu.get("year", "")).strip() if edu.get("year") else None
                    if institution:
                        education_items.append({...})
            return education_items
        except Exception:
            pass
    
    # Try pipe-delimited format (GT: "degree|institution|year|...")
    if "|" in s_str:
        try:
            parts = [p.strip() for p in s_str.split("|")]
            education_items = []
            for i in range(0, len(parts), 3):
                ...
            return education_items
        except Exception:
            pass
    
    return []
```

**Variant 3** — Cell 39 (line ~1950):
```python
def compare_education_components(gt_items, pred_items):
    """Compare education detail by detail: degree, institution, year."""
    if not gt_items or not pred_items:
        return {...}
    
    # [Different implementation with fuzzy matching]
```

#### Issue Analysis

- **Variant 1** only handles institutions, loses degree & year info
- **Variant 2** handles full structure (degree, institution, year) — most complete
- **Variant 3** implements scoring of education components (different purpose)
- ❌ Variant 1 is **incompatible** with later code expecting `parse_education_detailed()`
- ❌ Tests in cells use Variant 2, but results may use Variant 1
- ❌ Adding new format support requires updating all three variants

#### Recommendation

**Consolidate into single `modules/parsing.py`**:

```python
# models/parsing.py
class EducationParser:
    """Single source of truth for education parsing"""
    
    @staticmethod
    def parse_json_format(s):
        """Parse JSON list: [{"degree": "B.A.", ...}, ...]"""
        # Variant 2 logic
        ...
    
    @staticmethod
    def parse_pipe_delimited(s):
        """Parse pipe format: degree|institution|year|degree2|..."""
        # Variant 2 logic
        ...
    
    @classmethod
    def parse(cls, s, format_type="auto"):
        """Try both formats, return structured list of dicts"""
        if not s or pd.isna(s):
            return []
        
        s_str = str(s).strip()
        if s_str.startswith("["):
            return cls.parse_json_format(s_str)
        elif "|" in s_str:
            return cls.parse_pipe_delimited(s_str)
        return []
    
    @staticmethod
    def get_institutions_only(education_items):
        """For backward compatibility with simple extraction"""
        return [e["institution"] for e in education_items if e.get("institution")]
```

**Usage in notebook**:
```python
from modules.parsing import EducationParser

# Replace lines ~1700-1850:
def parse_education(s):
    items = EducationParser.parse(s)
    return "|".join(EducationParser.get_institutions_only(items))

def parse_education_detailed(s):
    return EducationParser.parse(s)
```

---

### 1.4 Code Duplication — Religion Categorization

**Severity**: 🔴 CRITICAL  
**Impact**: Inconsistent hierarchical scoring across evaluation sections

#### Current State: Multiple Definitions

**Definition 1** — Cell 37 (line ~1760):
```python
RELIGION_HIERARCHY = {
    "catholic": "catholic",
    "roman catholic": "catholic",
    "methodist": "protestant",
    # ... 20+ entries
}

def get_religion_category(religion_str):
    if pd.isna(religion_str) or str(religion_str).strip() == "":
        return None
    norm = str(religion_str).strip().lower()
    if norm in RELIGION_HIERARCHY:
        return RELIGION_HIERARCHY[norm]
    for key, category in RELIGION_HIERARCHY.items():
        if key in norm or norm in key:
            return category
    return norm
```

**Definition 2** — Cell 46 (Markdown section on religion hierarchy):
```markdown
# Religion Hierarchy & Hierarchical Matching
... same hierarchy and explanation ...
```

#### Issue Analysis

- ✅ RELIGION_HIERARCHY is correctly imported from `modules.config`
- ✅ `get_religion_category()` logic is centralized
- ✓ Markdown section (Cell 46) documents the hierarchy
- ❌ But redundancy exists: definition in code AND duplicate documentation in markdown
- ❌ If someone adds a new religion to hierarchy in modules, must update markdown too

#### Recommendation

**Reduce markdown verbosity**:

**Replace Cell 46 markdown** (currently _extremely_ verbose) with shorter version:
```markdown
## Religion Hierarchy & Hierarchical Matching

Religion scoring uses a hierarchy (see config.RELIGION_HIERARCHY) to award partial credit:
- **1.0**: exact match (Methodist = Methodist)
- **0.7**: same category (Methodist vs Christian → both protestant)
- **0.0**: different religions (Methodist vs Catholic)

The hierarchy maps denominations to parent categories (e.g., Methodist → protestant).
This recognizes that extracting the correct parent category is partially correct.
```

**Action**:
- Remove lines ~1950-2050 (long Python code block in markdown)
- Keep sections ~1927-1949 (conceptual explanation)
- Add comment in Cell 37: "For full hierarchy definition, see modules/config.py"

---

### 1.5 Scattered Configuration & Hardcoded Paths

**Severity**: 🔴 CRITICAL  
**Impact**: Pipeline breaks if run from different directory; 40+ edits needed to change paths

#### Current State: Paths Redefined Multiple Times

**Location 1** — Notebook Section 2 (line ~61-92):
```python
HTML_DIR   = PATHS['html_data']      # ✓ Uses centralized PATHS
OUTPUT_DIR = PATHS['output_root']    # ✓ Uses centralized PATHS
config_path = PATHS['groq_config']   # ✓ Uses centralized PATHS
```

**Location 2** — Notebook Section 2b (derived from centralized):
```python
# Uses PipelineConfig() which reads from centralized config
pipeline_config = PipelineConfig()
```

**Location 3** — Notebook Markdown Section 2 (line ~80):
```markdown
> **Before running:** if you have a `results_raw.json` from a previous run, rename it:
> ```bash
> mv senate_results/results_raw_v1_backup.json
> ```
# ❌ Demonstrates users must manually understand subdirectory structure
```

**Location 4** — Notebook Section 8 (line ~595-610):
```python
INPUT_PATH = "../external_data/senate_html/senators_index.csv"
PEW_PATH = "../external_data/ground_truth/pew_religion.csv"
OUTPUT_PATH = "../external_data/ground_truth/senate_ground_truth.csv"
LOG_PATH = "../external_data/ground_truth/scrape_errors.log"
# ❌ REDEFINED! Should use PATHS dict
```

#### Issue Analysis

- ✅ Section 2 correctly uses centralized `PATHS`
- ❌ Section 8 redefines paths manually
- ❌ Creates inconsistency: if PATHS config changes, Section 8 is broken
- ❌ Hard to run from different directory
- ❌ No environment (.env) support for dev vs prod paths

#### Recommendation

**Consolidate all paths to one place** (already partially done in `modules/config.py`):

**In notebook Section 8** (line ~595), REPLACE:
```python
# ❌ Old way:
INPUT_PATH = "../external_data/senate_html/senators_index.csv"
PEW_PATH = "../external_data/ground_truth/pew_religion.csv"
OUTPUT_PATH = "../external_data/ground_truth/senate_ground_truth.csv"
LOG_PATH = "../external_data/ground_truth/scrape_errors.log"

# ✅ New way:
INPUT_PATH = PATHS['senators_index_csv']
PEW_PATH = PATHS['pew_religion_csv']
OUTPUT_PATH = PATHS['ground_truth_csv']
LOG_PATH = PATHS['ground_truth_log']
```

**Update `modules/config.py`** to add these keys:
```python
PATHS = {
    # ... existing paths ...
    'senators_index_csv': PROJECT_ROOT / 'external_data/senate_html/senators_index.csv',
    'pew_religion_csv': PROJECT_ROOT / 'external_data/ground_truth/pew_religion.csv',
    'ground_truth_csv': PROJECT_ROOT / 'external_data/ground_truth/senate_ground_truth.csv',
    'ground_truth_log': PROJECT_ROOT / 'external_data/ground_truth/scrape_errors.log',
}
```

---

### 1.6 Evaluation Logic Repeated 3 Times (Per Prompt Style)

**Severity**: 🟠 HIGH  
**Impact**: ~500 lines of evaluation code; metric changes require 3 edits

#### Current State: Evaluation Loop Structure

The notebook runs nearly identical evaluation metrics for `direct`, `pseudocode`, and `icl` styles.

**Pattern (repeated 3 times)**:
```python
for style in ["direct", "pseudocode", "icl"]:
    df_style = df_pred[df_pred["prompt_style"] == style]
    merged = df_gt.merge(df_style, on="senator_id", how="inner")
    
    if merged.empty:
        print(f"✗ No matches found for {style}")
        continue
    
    print(f"\n{'='*60}")
    print(f"  PROMPT STYLE: {style.upper()}  (n={len(merged)})")
    print(f"{'='*60}")
    
    # === METRIC CALCULATION (repeated for each style) ===
    
    # Name matching
    if "full_name_x" in merged.columns:
        name_scores = [...]
        print(f"Accuracy — full_name: {avg:.2%}")
    
    # Gender (exact match)
    if gt_gender_col and pred_gender_col:
        g_scores = [...]
        print(f"Accuracy — gender: {avg:.2%}")
    
    # Birthdate (normalized, partial credit)
    if gt_bd_col and pred_bd_col:
        bd_results = [...]
        print(f"Accuracy — birthdate_{metric}: {avg:.2%}")
    
    # Religion (hierarchical match)
    if "religion" in merged.columns:
        rel_scores = [...]
        print(f"Accuracy — religion: {avg:.2%}")
    
    # ... more metrics ...
    
    # === END REPEATED SECTION ===
```

#### Issue Analysis

- ❌ Metrics calculated identically for each style
- ❌ 500+ lines could be condensed to ~150 with abstraction
- ❌ Adding new metric requires changing code in 3 places
- ❌ Hard to test individual metrics
- ❌ Difficult to refactor scoring functions

#### Recommendation

**Extract to `modules/evaluator.py`**:

```python
# modules/evaluator.py

class MetricCalculator:
    """Calculate individual metrics given GT and prediction data"""
    
    def __init__(self, merged_df):
        self.merged = merged_df
    
    def name_fuzzy_match(self):
        """Fuzzy name matching: 1.0 if similar, else 0.0"""
        scores = [
            name_match_score(gt, pred)
            for gt, pred in zip(
                self.merged["full_name_x"].fillna(""),
                self.merged["full_name_y"].fillna("")
            )
        ]
        return {"name": {"scores": scores, "metric": "accuracy"}}
    
    def gender_exact_match(self):
        """Exact gender match, NaN-aware"""
        gt_col = next((c for c in ["gender_x", "gender"] if c in self.merged.columns), None)
        pred_col = next((c for c in ["gender_y"] if c in self.merged.columns), None)
        
        if not gt_col or not pred_col:
            return {}
        
        scores = [
            gender_match_score(gt, pred)
            for gt, pred in zip(self.merged[gt_col], self.merged[pred_col])
        ]
        valid = [s for s in scores if not pd.isna(s)]
        return {
            "gender": {
                "scores": valid,
                "metric": "accuracy",
                "note": f"skipped {len(scores) - len(valid)} missing GT"
            }
        }
    
    def religion_hierarchical(self):
        """Hierarchical religion matching"""
        if "religion" not in self.merged.columns:
            return {}
        
        scores = [
            religion_match_score(gt, pred)
            for gt, pred in zip(
                self.merged["religion"],
                self.merged["religious_affiliation"]
            )
        ]
        valid = [s for s in scores if not pd.isna(s)]
        return {
            "religion": {
                "scores": valid,
                "metric": "hierarchical_match"
            }
        }
    
    def all_metrics(self):
        """Compute all metrics at once"""
        return {
            **self.name_fuzzy_match(),
            **self.gender_exact_match(),
            **self.religion_hierarchical(),
            # ... more metrics
        }


class EvaluationReporter:
    """Format and print evaluation results"""
    
    @staticmethod
    def print_style_results(style_name, df_gt, df_pred):
        """Run full evaluation for a style and print results"""
        df_style = df_pred[df_pred["prompt_style"] == style_name]
        merged = df_gt.merge(df_style, on="senator_id", how="inner")
        
        if merged.empty:
            print(f"✗ No matches for {style_name}")
            return
        
        print(f"\n{'='*60}")
        print(f"  PROMPT STYLE: {style_name.upper()}  (n={len(merged)})")
        print(f"{'='*60}")
        
        calculator = MetricCalculator(merged)
        results = calculator.all_metrics()
        
        for metric_name, metric_data in results.items():
            scores = metric_data["scores"]
            if scores:
                avg = sum(scores) / len(scores)
                metric_type = metric_data["metric"]
                note = metric_data.get("note", "")
                print(f"{metric_type:12}— {metric_name:20}: {avg:.2%}  {note}")
```

**Usage in notebook** (replaces ~500 lines):
```python
# Replace Cell 37 ~lines 1280-1934 with:
from modules.evaluator import EvaluationReporter

for style in ["direct", "pseudocode", "icl"]:
    EvaluationReporter.print_style_results(style, df_gt, df_pred)
```

**Result**: ✅ Reduction from ~500 → ~50 lines in notebook!

---

## PART 2: MARKDOWN SECTION ISSUES

### 2.1 Verbose & Redundant Section Headers

**Severity**: 🟡 MEDIUM  
**Impact**: Reduces readability; adds ~50 unnecessary lines

#### Current State: Overly Detailed Headers

**Section 2 Header** (line ~42-55):
```markdown
# Senate Profile LLM Extraction Pipeline — v5
**DSBA 6010 — Chloe Partridge**

Aligned with Liu et al. (USENIX Security 2025) *Evaluating LLM-based Personal Information Extraction and Countermeasures*.

Key features:
- **Multi-provider support** — Groq (8B, 70B) and Gemini (configure in Section 2)
- **Prompt-style ablation** — direct, pseudocode, ICL (Section 4.2 / Table 13)
- **Religion signal annotation** — LLM-based pre-classification...
- **Traditional baselines** — regex + spaCy NER (Tables 4–5)
- **Evaluation metrics** — Accuracy, Rouge-1, BERT score...
- **Model comparison** — 8B vs 70B vs Gemini (Table 3 / Section 6.2)

**Quick Start:** Set your provider and model in Section 2, then run cells top to bottom.
```

**Issue**: This is good metadata for a report, but:
- ❌ 14 lines for what could be 3-4 lines in notebook format
- ❌ Repetitive with comments already in code cells
- ❌ References ("Section 4.2", "Table 13") are fragile as notebook changes

**Section 8b Header** (line ~962-979):
```markdown
## 8b. Religion Signal Annotation

Classifies each senator's bio text as `explicit` (religion directly mentioned)
or `not_explicit` (absent or only inferable from indirect signals).

This is an **input characterisation step**, not ground truth annotation — it
describes what information was available to the LLM, not what the correct answer is.
It runs as a separate API call to avoid contaminating the main extraction task.

**Why this matters:** If the model achieves high accuracy on `religious_affiliation`,
we need to know how much of that comes from bios where religion was stated outright
versus bios requiring multi-hop inference. These are different claims about LLM capability.

**Output:** Adds `religion_signal` column to `senate_ground_truth.csv`.
Values: `explicit` | `not_explicit` | `error`

**Spot-check:** Review ~15–20 cases manually against the raw bio text before
drawing conclusions from stratified accuracy numbers.
```

**Issue**: 
- ✓ Actually good explanation, but very detailed for a notebook
- ❌ 18 lines could be condensed to 6-7 lines
- ❌ "Spot-check" recommendation is operational guidance (belongs in code comments)

**Section 8 Header** (line ~576-592):
```markdown
## 8. Ground Truth Builder — Multi-Source Pipeline

Builds **ground_truth.csv** by combining data from:
1. **Wikipedia** — birthdate, gender, race_ethnicity
2. **Ballotpedia** — committee_roles
3. **Pew Research** — religion (via fuzzy matching)

**Input:** senators_index.csv with full names and states  
**Intermediate:** senators_index.csv is enriched with Wikipedia and Ballotpedia URLs  
**Output:** senate_ground_truth.csv with columns: name, state, full_name, birthdate, gender, race_ethnicity, committee_roles, religion

**Features:**
- Resume-safe: incremental saving every 10 senators
- Comprehensive logging to scrape_errors.log
- Fuzzy matching (rapidfuzz) against Pew religion data
- Connection pooling via requests.Session() for efficient scraping
```

**Issue**:
- ✓ Excellent documentation for clarity
- ❌ 14 lines + lots of formatting
- ❌ "Connection pooling" detail is implementation-level, not conceptual

#### Recommendation

**Consolidate header sections** — reduce from detailed lists to concise descriptions:

**Section 2** (reduce from 14 → 4 lines):
```markdown
# Pipeline Overview
Multi-stage extraction: (1) HTML → readable text, (2) LLM extraction with prompt ablation, 
(3) traditional baselines (regex/spaCy), (4) evaluation vs ground truth.

**Quick Start**: Configure provider/model in Section 2, then run top to bottom.
```

**Section 8** (reduce from 14 → 6 lines):
```markdown
## Ground Truth Builder

Multi-source web scraping: Wikipedia (demographics), Ballotpedia (committees), Pew (religion).  
Resume-safe: saves every 10 senators. Output: `senate_ground_truth.csv`
```

**Section 8b** (reduce from 18 → 5 lines):
```markdown
## Religion Signal Annotation

Classifies bios as `explicit` (religion stated) vs `not_explicit` (inferred/absent).  
This annotations let us measure how much LLM accuracy comes from direct language vs reasoning.
```

**Result**: ~40 fewer markdown lines; clearer hierarchy with code comments handling operational details.

---

### 2.2 Redundant Documentation in Code Blocks

**Severity**: 🟡 MEDIUM  
**Impact**: Markdown and code document same thing twice

#### Current State: Python Code in Markdown

**Cell 46** (entire cell is markdown with Python example):
```markdown
# Religion Hierarchy & Hierarchical Matching

To give partial credit for related religions...

## Helper Functions for Evaluation

```python
# Religion hierarchy: maps denominations to their parent category
RELIGION_HIERARCHY = {
    "catholic": "catholic",
    "roman catholic": "catholic",
    ...
}

def get_religion_category(religion_str):
    ...

def religion_match_score(gt_val, pred_val):
    ...
```

**Scoring Details:**
- **1.0** — Exact match...
```

**Issue**:
- ✓ Helpful for explanation
- ❌ ~130 lines of code in markdown that's already in `modules/config.py`
- ❌ If code changes, markdown becomes outdated
- ❌ Redundant with Cell 37 implementation

#### Recommendation

**Replace markdown code block with reference**:

```markdown
## Religion Hierarchy & Hierarchical Matching

Scoring uses a hierarchy (see `modules/config.RELIGION_HIERARCHY`) to award partial credit:
- **1.0**: exact match
- **0.7**: same category (e.g., Methodist & Christian both → protestant)
- **0.0**: unrelated religions

This allows partial correctness: extracting the parent category when the specific 
denomination isn't available is semantically reasonable.

For implementation details, see `modules/config.py` and `modules/evaluator.py`.
```

**Result**: Reduces from ~130 lines to ~10 lines; references canonical source of truth.

---

## PART 3: FOLDER STRUCTURE ISSUES & IMPROVEMENTS

### 3.1 Current Folder Organization

```
LLM-Based-Personal-Profile-Extraction/
├── configs/                          # ✓ Well-organized
│   ├── model_configs/
│   │   ├── gemini_config.json
│   │   ├── gpt_config.json
│   │   ├── groq_config_extraction.json
│   │   ├── groq_config.json
│   │   └── llama_config.json
│   └── task_configs/
│       └── synthetic.json
│
├── data/                             # ✓ Reasonably organized
│   ├── subword_nmt.voc
│   ├── icl/
│   ├── synthetic/
│   └── system_prompts/
│
├── external_data/                    # ✓ Appropriate location
│   ├── pew_religion.csv
│   ├── ground_truth/
│   └── senate_html/
│
├── LLMPersonalInfoExtraction/        # ✓ Main codebase
│   ├── __init__.py
│   ├── config_loader.py
│   ├── attacker/
│   ├── defense/
│   ├── evaluator/
│   ├── models/
│   ├── tasks/
│   └── utils/
│
├── modules/                          # ✓ Pipeline utilities (good!)
│   ├── __init__.py
│   ├── config.py
│   ├── html_processing.py
│   ├── name_utils.py
│   └── [MISSING: parsing.py, evaluator.py, rate_limiter.py]
│
├── experiments/                      # ⚠️  MIXED CONCERNS
│   ├── *.ipynb (5 notebooks)
│   ├── main.py                       # ⚠️ script
│   ├── evaluate.py                   # ⚠️ script  
│   ├── run.py                        # ⚠️ script
│   ├── audit_pipeline_inputs.py      # ⚠️ script
│   ├── rescrape_flagged.py           # ⚠️ script (duplicate code)
│   ├── senate_scraper.py             # ⚠️ script
│   ├── test_models.py                # ⚠️ script
│   ├── check_defense.py              # ⚠️ script
│   ├── score.py                      # ⚠️ script
│   ├── senate_html/                  # Data, not code
│   └── visualize_results.ipynb       # Notebook in wrong place
│
├── scripts/                          # ❌ EMPTY/UNDERUTILIZED
│   ├── analyze_scrape.py
│   ├── check_defense.py (duplicate?)
│   └── score.py (duplicate?)
│
├── outputs/                          # ✓ Results storage
│   ├── log/
│   ├── result/
│   └── senate_results/
│
├── docs/                             # ⚠️ Mostly guidance docs
│
└── Analysis & guideline files        # ⚠️ ROOT CLUTTER
    ├── ANALYSIS_SUMMARY.md
    ├── CODE_REVIEW.md
    ├── NOTEBOOK_IMPROVEMENT_ANALYSIS.md
    ├── PHASE1_ACTION_PLAN.md
    ├── PHASE1_IMPLEMENTATION_CHECKLIST.md
    ├── PHASE1_IMPLEMENTATION_GUIDE.md
    ├── PHASE1_QUICK_REFERENCE.md
    └── REFACTORING_PLAN.md
```

---

### 3.2 Identified Issues

#### Issue A: Duplicate/Scattered Scripts

**Location**: `experiments/` and `scripts/` both contain scripts

**Current State**:
- `scripts/analyze_scrape.py` — analyzes scrape results
- `scripts/check_defense.py` — defense checking
- `scripts/score.py` — scoring
- `experiments/main.py` — pipeline entry point
- `experiments/evaluate.py` — evaluation entry point
- `experiments/run.py` — another runner
- `experiments/audit_pipeline_inputs.py` — audit script
- `experiments/rescrape_flagged.py` — rescraping script
- `experiments/senate_scraper.py` — scraper script
- `experiments/test_models.py` — model testing
- `experiments/check_defense.py` — duplicate of `scripts/check_defense.py`?

**Issue**:
- ❌ 10+ scripts scattered between two directories
- ❌ Unclear which is canonical
- ❌ Duplication (check_defense.py in both places?)
- ❌ Hard to find something; not organized by function

#### Issue B: Root Directory Clutter

**Current**: 8 analysis/guideline markdown files in root
```
ANALYSIS_SUMMARY.md
CODE_REVIEW.md
NOTEBOOK_IMPROVEMENT_ANALYSIS.md
PHASE1_ACTION_PLAN.md
PHASE1_IMPLEMENTATION_CHECKLIST.md
PHASE1_IMPLEMENTATION_GUIDE.md
PHASE1_QUICK_REFERENCE.md
REFACTORING_PLAN.md
```

**Issue**:
- ❌ Root directory cluttered with guidance documents
- ❌ No clear organization (which to read first?)
- ❌ Version confusion (which is latest?)

#### Issue C: Notebooks Scattered

**Current**:
- `experiments/senate_llm_pipeline_V2.ipynb`
- `experiments/senate_llm_pipeline_V1.ipynb`
- `experiments/visualize_results.ipynb`
- `experiments/evaluate.py` (not notebook, but mixed with notebooks)

**Issue**:
- ❌ Multiple versions (V1 vs V2) — unclear which is current
- ❌ Mixed with Python scripts

---

### 3.3 Recommended Folder Restructure

```
LLM-Based-Personal-Profile-Extraction/
│
├── README.md                         # Entry point for documentation
│
├── docs/                             # ✓ Documentation reference
│   ├── README.md                     # What each component does
│   ├── PIPELINE_ARCHITECTURE.md      # High-level overview
│   ├── SETUP.md                      # Installation & config
│   ├── GETTING_STARTED.md            # Quick start guide
│   ├── IMPROVEMENT_ANALYSIS.md       # This file! (move here)
│   └── guides/
│       ├── prompt_engineering.md
│       ├── ground_truth_building.md
│       └── evaluation_metrics.md
│
├── configs/                          # ✓ Keep as-is
│   ├── model_configs/
│   └── task_configs/
│
├── data/                             # ✓ Keep as-is
│   ├── icl/
│   ├── synthetic/
│   └── system_prompts/
│
├── external_data/                    # ✓ Keep as-is (web-scraped sources)
│   ├── ground_truth/
│   └── senate_html/
│
├── modules/                          # ✓ Core pipeline utilities
│   ├── __init__.py
│   ├── config.py                     # ✓ Already here
│   ├── html_processing.py            # ✓ Already here
│   ├── name_utils.py                 # ✓ Already here
│   ├── parsing.py                    # ✨ NEW: consolidate education/structured parsing
│   ├── evaluator.py                  # ✨ NEW: metric calculation & reporting
│   ├── rate_limiter.py               # ✨ NEW: API rate limiting
│   └── baselines.py                  # ✨ NEW: regex & spaCy baseline extractors
│
├── LLMPersonalInfoExtraction/        # ✓ Keep as-is (model/attacker/defense logic)
│   ├── config_loader.py
│   ├── attacker/
│   ├── defense/
│   ├── evaluator/
│   ├── models/
│   ├── tasks/
│   └── utils/
│
├── experiments/                      # ✨ CLEANED UP: Notebooks only
│   ├── 1_data_collection.ipynb       # ✨ New workflow structure
│   ├── 2_ground_truth_builder.ipynb  # ✨
│   ├── 3_llm_extraction.ipynb        # ✨
│   ├── 4_analysis_evaluation.ipynb   # ✨
│   ├── 5_final_report.ipynb          # ✨
│   └── archive/                      # ✨ OLD notebooks moved here
│       ├── senate_llm_pipeline_V1.ipynb
│       ├── senate_llm_pipeline_V2.ipynb
│       └── visualize_results.ipynb
│
├── scripts/                          # ✨ CLI entry points & utilities
│   ├── README.md                     # Which script to use when
│   ├── run_pipeline.py               # ✨ Consolidated from main.py, run.py
│   ├── evaluate_results.py           # ✨ From evaluate.py, score.py
│   ├── audit_pipeline.py             # ✨ From audit_pipeline_inputs.py
│   ├── scrape_groundtruth.py         # ✨ From rescrape_flagged.py
│   └── analyze_results.py            # ✨ From analyze_scrape.py
│
├── outputs/                          # ✓ Keep as-is (results directory)
│   ├── log/
│   ├── result/
│   └── senate_results/
│
├── tests/                            # ✨ NEW: Unit tests
│   ├── test_parsing.py               # ✨ Test education parser
│   ├── test_evaluator.py             # ✨ Test metric calculation
│   ├── test_html_processing.py       # ✨ Test HTML extraction
│   └── test_baselines.py             # ✨ Test baseline extractors
│
├── .env.example                      # ✓ Keep as-is
├── .gitignore                        # ✓ Keep as-is
├── requirements.txt                  # ✨ NEW: Python dependencies
└── PIE_environment.yml               # ✓ Keep (conda alternative)
```

---

### 3.4 Migration Plan

#### Step 1: Consolidate New Modules
```
✨ modules/parsing.py        ← Consolidate education parsing (3 variants → 1)
✨ modules/evaluator.py      ← Extract evaluation logic from Cell 37
✨ modules/rate_limiter.py   ← Extract rate limiting configuration
✨ modules/baselines.py      ← Extract regex & spaCy baseline functions
```

#### Step 2: Consolidate Scripts
```
scripts/run_pipeline.py       ← Merge main.py, run.py, evaluate.py
scripts/evaluate_results.py   ← Merge score.py, check_defense.py
scripts/audit_pipeline.py     ← Move audit_pipeline_inputs.py
scripts/analyze_results.py    ← Move analyze_scrape.py
scripts/scrape_groundtruth.py ← Move/consolidate rescrape_flagged.py
```

#### Step 3: Reorganize Notebooks
```
experiments/archive/          ← Move V1, V2, old visualization notebooks
experiments/1_data_collection.ipynb   ← Move active pipeline
experiments/2_ground_truth_builder.ipynb
experiments/3_llm_extraction.ipynb
experiments/4_analysis_evaluation.ipynb
experiments/5_final_report.ipynb
```

#### Step 4: Move Documentation
```
docs/IMPROVEMENT_ANALYSIS.md  ← Move from root
docs/PHASE1_*.md              ← Move planning docs
docs/ANALYSIS_SUMMARY.md      ← Move root clutter
docs/CODE_REVIEW.md           ← Move root clutter
```

#### Step 5: Add Unit Tests
```
tests/test_parsing.py         ← Test EducationParser, education_items parsing
tests/test_evaluator.py       ← Test metric calculations
tests/test_html_processing.py ← Test extract_readable_text
tests/test_baselines.py       ← Test regex & spaCy extractors
```

---

## PART 4: SUMMARY TABLE - ALL IMPROVEMENTS

| Issue | Severity | Lines to Change | Module to Create/Update | Benefit |
|-------|----------|-----------------|------------------------|---------|
| HTML extraction duplicated (4×) | 🔴 | 40-60 | Use `modules.extract_readable_text` | Single source of truth |
| Name normalization duplicated (3×) | 🔴 | 50-80 | Use `modules.NameNormalizer` | Consistent senator IDs |
| Education parsing (3 variants) | 🔴 | 200-250 | `modules/parsing.py::EducationParser` | Single canonical parser |
| Religion hierarchy duplicated | 🟠 | 100-150 | Reduce markdown; reference config | Single source of truth |
| Paths hardcoded (40+ locations) | 🔴 | ~80 | Use `PATHS` dict throughout | Environment-independent |
| Evaluation loop repeated 3× | 🟠 | 500 | `modules/evaluator.py` | Metric changes in 1 place |
| Markdown verbose headers | 🟡 | ~50 | Consolidate headers | Better readability |
| Markdown duplicate code | 🟡 | ~130 | Replace with references | Single source of truth |
| Scripts scattered (10 files) | 🟡 | 50-100 | Consolidate to `scripts/` | Clear entry points |
| Root clutter (8 docs) | 🟡 | 0 (move only) | Move to `docs/` | Better organization |
| No unit tests | 🟡 | N/A | Create `tests/` | Build confidence |

---

## IMPLEMENTATION PRIORITY

### 🔴 CRITICAL (Do First)
1. ✅ **Extract missing modules**: `parsing.py`, `evaluator.py` (frees ~700 lines in notebook)
2. ✅ **Consolidate paths** to use `PATHS` dict throughout
3. ✅ **Update Section 8**  to use centralized imports instead of redefining functions

### 🟠 HIGH (Do Next)
4. ✅ **Refactor evaluation loop** to use `evaluator.py`
5. ✅ **Consolidate scripts** into unified entry points
6. ✅ **Split notebooks** into 5-part workflow

### 🟡 MEDIUM (Nice to Have)
7. ✅ **Compress markdown** headers and documentation
8. ✅ **Move docs** to dedicated folder
9. ✅ **Add unit tests** for critical functions

---

## Expected Outcomes

| Metric | Before | After | Gain |
|--------|--------|-------|------|
| **Notebook lines** | ~3,300 | ~1,500 | 55% reduction |
| **Code duplication** | ~600 lines | <50 lines | 92% reduction |
| **Scattered configurations** | 40+ locations | 1 centralized dict | 95% reduction |
| **Evaluation loop variations** | 3 (repeated) | 1 (abstracted) | 3× consolidation |
| **Scripts/entry points** | 10 files | 5 files | 50% consolidation |
| **Time to add new metric** | 3 edits (3 min) | 1 edit in config (30 sec) | 6× faster |
| **Time to change rate limits** | 5+ edits | 1 config update | 5× faster |

