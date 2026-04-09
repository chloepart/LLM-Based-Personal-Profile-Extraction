# Senate LLM Extraction Pipeline (v5) — Improvement Analysis

## Executive Summary

This notebook implements a comprehensive LLM-based personal information extraction and evaluation pipeline aligned with Liu et al. (USENIX Security 2025). While well-structured overall, the ~3300-line notebook contains **significant redundancies** and opportunities for modularity improvements that would:

- **Reduce code duplication** by ~25-30%
- **Improve maintainability** through centralized configuration
- **Enable easier testing** via extracted utility functions
- **Simplify evaluation** through metric consolidation

---

## 1. Critical Redundancies

### 1.1 — Duplicate URL Generation Functions

**Location:** Lines 817–870 (Section 8) and ~1138–1200 (Section 8b)

**Issue:** `create_slug()` function defined twice with **slightly different implementations** and name normalization logic.

**Current:**
```python
# Section 8 — First definition
def create_slug(name):
    slug = re.sub(r'\s+[A-Z]\.\s*', ' ', name).strip()
    slug = slug.replace(" ", "_")
    overrides = { "Bernard_Sanders": "Bernie_Sanders" }
    return overrides.get(slug, slug)

# Section 8b — Second definition (similar but more elaborate)
def create_slug(name):
    norm_name = str(name).lower().strip()
    # ... remove accents, expand abbreviations, etc.
    overrides = { "Bernard_Sanders": "Bernie_Sanders", "Dan_Sullivan": "Daniel_Sullivan", ... }
    return overrides.get(slug_lower, slug)
```

**Impact:** Second definition includes name normalization (Dan→Daniel expansions) that first lacks. Maintenance nightmare—if you fix one, the other breaks.

**Recommendation:** 
- Extract to `utils/url_builders.py` with single definitive implementation
- Include both basic and advanced normalization modes
- Use as single source of truth for all URL generation

---

### 1.2 — Repeated Markup Extraction

**Locations:** 
- Line 156 (`extract_readable_text()` definition, Section 3)
- Line 522 (uses same BeautifulSoup pattern manually)
- Line 1071+ (Wikipedia scraping with inline BeautifulSoup)

**Issue:** Multiple cells independently strip `<script>`, `<style>`, etc. without using the centralized function.

**Current Code Pattern (Repeated):**
```python
# Pattern 1 (centralized function - line 156)
def extract_readable_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    return soup.get_text(...)

# Pattern 2 (manual in wiki scraper - line 1089)
soup = BeautifulSoup(response.content, "html.parser")
for tag in soup(["script", "style", "nav", "footer"]):
    tag.decompose()  # ← DUPLICATED
wiki_text = soup.get_text(...)
```

**Impact:** Non-obvious that they should behave identically; changes to one might not propagate to others.

**Recommendation:** 
- Create centralized `html_processing.py` module with:
  - `extract_readable_text(html, separator=" ")`
  - `extract_infobox(soup, field_mappings)` for Wikipedia
  - `extract_text_length_limited(html, max_chars=10000)`

---

### 1.3 — Path Configuration Scattered Throughout

**Issue:** Paths redefined in multiple sections instead of single configuration:

**Locations:**
- Line 70–92: HTML_DIR, OUTPUT_DIR, config setup
- Line 590: LOG_PATH, INPUT_PATH, PEW_PATH
- Line 674: GT_PATH, EDUCATION_PROMPT prompt
- Line 2180+: Paths used but not centralized

**Current:**
```python
# Section 2
HTML_DIR   = Path("../external_data/senate_html")
OUTPUT_DIR = Path("../outputs/senate_results")

# Section 8
INPUT_PATH = "../external_data/senate_html/senators_index.csv"
PEW_PATH   = "../external_data/ground_truth/pew_religion.csv"
OUTPUT_PATH = "../external_data/ground_truth/senate_ground_truth.csv"
```

**Impact:** 
- Changes require editing multiple locations
- Hard to support different data paths (dev vs. prod)
- Inconsistent path separator handling

**Recommendation:**
- Create centralized `config.py` or `.env` file:
  ```python
  PATHS = {
      'html_data': Path("../external_data/senate_html"),
      'output_dir': Path("../outputs/senate_results"),
      'ground_truth': Path("../external_data/ground_truth"),
      'pew_religion': Path("../external_data/ground_truth/pew_religion.csv"),
  }
  ```
- Use throughout notebook via `from config import PATHS`

---

## 2. Evaluation Logic Redundancy

### 2.1 — Multi-Style Evaluation Loop Repetition

**Issue:** Nearly identical evaluation metric calculation repeated for each prompt style (direct, pseudocode, icl).

**Locations:** Lines 1280–1934

**Current Code Pattern (Repeated 3 times):**
```python
for style in ["direct", "pseudocode", "icl"]:
    df_style = df_pred[df_pred["prompt_style"] == style]
    merged = df_gt.merge(df_style, on="senator_id", how="inner")
    
    # Calculate metrics (REPEATED for each style):
    for col in ["full_name", "gender", "birthdate"]:
        scores = [calculate_score(gt, pred) for gt, pred in zip(merged[gt_col], merged[pred_col])]
        print(f"Metric — {col}: {avg(scores):.2%}")
```

**Impact:**
- 655 lines of code for what could be ~150 lines with abstraction
- Hard to add or modify metrics—must change in 3 places
- Testing/debugging 3× harder

**Recommendation:**
- Extract to `evaluator.py`:
  ```python
  def evaluate_extraction(df_gt, df_pred, style, metrics_config):
      """
      Args:
          df_gt: Ground truth DataFrame with columns: senator_id, full_name, gender, ...
          df_pred: Predictions DataFrame with columns: senator_id, prompt_style, full_name, ...
          style: "direct" | "pseudocode" | "icl"
          metrics_config: Dict specifying which metrics to compute
      
      Returns:
          Dict with results for each metric
      """
      df_style = df_pred[df_pred["prompt_style"] == style]
      merged = df_gt.merge(df_style, on="senator_id", how="inner", suffixes=("_gt", "_pred"))
      
      results = {}
      for metric_name, metric_fn in metrics_config.items():
          results[metric_name] = metric_fn(merged)
      
      return results
  ```
- Call once per style:
  ```python
  for style in ["direct", "pseudocode", "icl"]:
      results = evaluate_extraction(df_gt, df_pred, style, METRICS_CONFIG)
      print_results(results, style)
  ```

---

### 2.2 — Metric Calculation Functions Scattered

**Issue:** Helper functions for scoring (name_match_score, gender_match_score, religion_match_score, etc.) are useful but defined inline and hardcoded.

**Locations:** Lines 1280–1500 (within evaluation cell)

**Problem Example:**
```python
def name_match_score(gt_name, pred_name):
    # Defined at line 1320, but similar functions repeated
    ...

def gender_match_score(gt_val, pred_val):
    # Nearly identical structure to name_match_score
    if pd.isna(gt_val) or str(gt_val).strip() == "":
        return float("nan")
    ...
```

**Recommendation:**
- Extract to `metrics.py`:
  ```python
  class Metric:
      def score(self, gt, pred): raise NotImplementedError
      
  class ExactMatchMetric(Metric):
      def score(self, gt, pred):
          if pd.isna(gt) or pd.isna(pred):
              return float("nan")
          return float(str(gt).lower() == str(pred).lower())
  
  class FuzzyMatchMetric(Metric):
      def score(self, gt, pred):
          # Fuzzy matching logic here
          ...
  ```
- Use: `METRICS_CONFIG = {"gender": ExactMatchMetric(), "name": FuzzyMatchMetric()}`

---

## 3. Data Loading & Merging Redundancy

### 3.1 — Repeated CSV Loading

**Issue:** Same CSV files loaded multiple times across different sections.

**Locations:**
- Line 560: `pd.read_csv(OUTPUT_DIR / "results_raw.json")`
- Line 640: `pd.read_csv(OUTPUT_DIR / "task1_pii.csv")`
- Line 1280+: Repeated loading in evaluation loop
- Line 1800+: Another load for GT comparison
- Line 2010+: Loaded again for diagnostics

**Pattern:**
```python
# Multiple independent cells all do:
df_t1 = pd.read_csv(OUTPUT_DIR / "task1_pii.csv")
df_gt = pd.read_csv("../external_data/ground_truth/senate_ground_truth.csv")

# Then merge similarly:
merged = df_gt.merge(df_t1, on="senator_id", how="inner")
```

**Impact:**
- Disk I/O overhead if re-running analysis
- Inconsistent column naming (sometimes `full_name_x`/`full_name_y`, sometimes different)
- Hard to track which version of CSV is being used

**Recommendation:**
- Create `data_loader.py`:
  ```python
  class DataPipeline:
      def __init__(self, paths_config):
          self.paths = paths_config
          self._cache = {}
      
      def load_predictions(self, cache=True):
          key = "predictions"
          if cache and key in self._cache:
              return self._cache[key]
          df = pd.read_csv(self.paths['results_csv'])
          if cache:
              self._cache[key] = df
          return df
      
      def load_ground_truth(self, cache=True):
          key = "ground_truth"
          if cache and key in self._cache:
              return self._cache[key]
          df = pd.read_csv(self.paths['ground_truth_csv'])
          if cache:
              self._cache[key] = df
          return df
      
      def merge_datasets(self, **kwargs):
          """Merge GT and predictions with consistent column handling"""
          df_gt = self.load_ground_truth()
          df_pred = self.load_predictions()
          return df_gt.merge(df_pred, on="senator_id", how="inner", suffixes=("_gt", "_pred"))
  ```

---

## 4. API Call & Rate Limiting Management

### 4.1 — Inconsistent Rate Limiting Configuration

**Issue:** Rate limits hardcoded in multiple places and adjusted throughout.

**Locations:**
- Line 520: `INTER_SENATOR_DELAY = 6 if RUN_ALL_PROMPT_STYLES else 4`
- Line 900: `time.sleep(3)` between styles
- Line 1000: `time.sleep(2)` in education extraction
- Line 1700: Different defaults in ablation loop (`time.sleep(3)`)

**Current:**
```python
# Section 5
INTER_SENATOR_DELAY = 6 if RUN_ALL_PROMPT_STYLES else 4

# Section 8
time.sleep(1)    # Wikipedia
time.sleep(1.5)  # Ballotpedia
time.sleep(2)    # Education extraction

# Ablation
time.sleep(3)    # Between style calls
```

**Impact:**
- Hard to maintain consistent rate limiting
- Easy to accidentally exceed quota
- Difficult to tune for different providers

**Recommendation:**
- Create `rate_limiter.py`:
  ```python
  class RateLimiter:
      def __init__(self, config):
          self.delays = {
              'inter_senator': config.get('inter_senator_delay', 4),
              'between_styles': config.get('between_styles', 3),
              'web_scrape': config.get('web_scrape', 1.5),
              'baseline': config.get('baseline', 0.5),
          }
      
      def wait_inter_senator(self):
          time.sleep(self.delays['inter_senator'])
      
      def wait_between_styles(self):
          time.sleep(self.delays['between_styles'])
  ```

---

## 5. Baseline Comparison Logic

### 5.1 — Regex & spaCy Extraction Defined Too Late

**Issue:** Baseline extraction functions defined in Section 6 (line 595) but conceptually should be in Section 3 alongside HTML preprocessing.

**Current Structure:**
```
Section 1: Dependencies
Section 2: Configuration
Section 3: HTML Preprocessing ← extract_readable_text()
Section 4: Prompt Design
Section 5: Run Pipeline
Section 6: Flatten Results
Section XX: Baselines ← regex_extract(), spacy_extract() HERE (should be earlier)
```

**Impact:**
- Readers don't see baseline methodology early
- Baseline results can't be generated until after main pipeline completes
- Can't easily compute baselines on held-out subset

**Recommendation:**
- Move baseline extraction to Section 3.5 (after HTML preprocessing)
- Create `baselines.py` with:
  ```python
  class RegexBaseline:
      def extract(self, text):
          return {
              'full_name': self._extract_name(text),
              'email': self._extract_email(text),
              'phone': self._extract_phone(text),
          }
  
  class SpaCyBaseline:
      def extract(self, text):
          doc = self.nlp(text[:10000])
          return {
              'persons': [ent.text for ent in doc.ents if ent.label_ == "PERSON"],
              'orgs': [ent.text for ent in doc.ents if ent.label_ == "ORG"],
          }
  ```

---

## 6. Ground Truth Building Redundancy

### 6.2 — Repeated Name Normalization & URL Construction

**Issue:** Section 8 (ground truth builder) and Section 8b (religion signal annotation) both need to construct Wikipedia URLs; code is similar but not shared.

**Locations:**
- Lines 817–870: Name slug creation for Section 8
- Lines 1138–1210: Expanded version for Section 8b education extraction
- Both include similar abbreviation expansions and overrides

**Pattern:**
```python
# Section 8
def create_slug(name):
    slug = re.sub(r'\s+[A-Z]\.\s*', ' ', name).strip()
    # ... basic version

# Section 8b  
def create_wikipedia_url(senator_name):
    norm_name = str(senator_name).lower().strip()
    # ... expanded version with abbreviation handling
```

**Recommendation:**
- Extract to single `name_utils.py`:
  ```python
  class NameNormalizer:
      ABBREVIATIONS = {
          r'\bdan\b': 'daniel',
          r'\btom\b': 'thomas',
          # ... all expansions
      }
      OVERRIDES = {
          'Bernard_Sanders': 'Bernie_Sanders',
          # ... all overrides
      }
      
      def normalize(self, name):
          norm = str(name).lower().strip()
          # Apply abbreviations
          # Remove accents
          # Apply overrides
          return norm
      
      def create_slug(self, name, format='wikipedia'):
          norm = self.normalize(name)
          # Format differently for wikipedia vs ballotpedia if needed
          return formatted_slug
  ```

---

## 7. Evaluation Diagnostics & Testing

### 7.1 — Exploratory Analysis Cells Should Be Functions

**Issue:** Cells 2010–2100+ contain exploratory/diagnostic code that could be refactored into reusable functions.

**Examples:**
- Line 2020+: Education extraction diagnostics (comparing GT vs LLM side-by-side)
- Line 2080+: Root cause analysis of degree/year matching
- Line 2170+: Degree matching deep dive
- Line 2260+: Detailed trace of component matching

**Current (Cell 2020):**
```python
print("\n" + "="*100)
print("🔍 EDUCATION EXTRACTION DIAGNOSTICS")
print("="*100)

for style in ["direct"]:
    df_style = df_pred[df_pred["prompt_style"] == style]
    merged_diag = df_gt.merge(df_style, ...)
    
    has_edu = merged_diag[
        (merged_diag["education_text"].notna()) & 
        (merged_diag["education"].notna())
    ].copy()
    
    for idx, row in has_edu.sample(5).iterrows():
        print(f"  GT:  {row['education_text'][:150]}")
        print(f"  LLM: {row['education'][:150]}")
```

**Recommendation:**
- Create `diagnostics.py`:
  ```python
  class ExtractionDiagnostics:
      def __init__(self, df_gt, df_pred):
          self.df_gt = df_gt
          self.df_pred = df_pred
      
      def compare_field(self, field, style, sample_size=5):
          """Compare GT vs LLM for a specific field"""
          df_style = self.df_pred[self.df_pred["prompt_style"] == style]
          merged = self.df_gt.merge(df_style, on="senator_id", how="inner")
          
          for _, row in merged.sample(sample_size).iterrows():
              gt_val = row.get(f"{field}_gt", "")
              pred_val = row.get(f"{field}_pred", "")
              print(f"GT:  {str(gt_val)[:150]}")
              print(f"LLM: {str(pred_val)[:150]}")
      
      def show_component_mismatch(self, style, component_type="education", threshold=0.8):
          """Show cases where component matching score is below threshold"""
          ...
  ```

---

## 8. Markdown Documentation Issues

### 8.1 — Inconsistent Section Headers & Documentation

**Issue:** Markdown headers are verbose and sometimes duplicated or redundantly explanatory.

**Examples:**
- Line 2 (Section 1): Title already says "Senate Profile LLM Extraction Pipeline — v5"
  ```markdown
  # Senate Profile LLM Extraction Pipeline — v5
  ## 1. Dependencies
  ```
  
- Line 8b header (Line 1051): Very long explanation that could be condensed:
  ```markdown
  ## 8b. Religion Signal Annotation
  
  Classifies each senator's bio text as `explicit` (religion directly mentioned)
  or `not_explicit` (absent or only inferable from indirect signals).
  
  This is an **input characterisation step**, not ground truth annotation...
  [continues for many lines]
  ```

- Line 9 header (Line 1264): Vague name:
  ```markdown
  ## 9. Evaluation Metrics (Liu et al. Section 6.1.4)
  ```
  Could be clearer: `## 9. Structured PII Evaluation Metrics (Accuracy, Rouge-1, BERT)`

**Recommendation:**
- Consolidate header descriptions into consistent format:
  ```markdown
  ## 2. Configuration
  Load model config, set API parameters, initialize client.
  
  ### 2.1 Configuration Loading
  ...
  
  ### 2.2 Session Metadata & Prompt Selection
  ...
  ```

- Create Table of Contents at top:
  ```markdown
  # Table of Contents
  1. Dependencies
  2. Configuration
  3. HTML Preprocessing & Baselines
  4. Prompt Design (Direct / Pseudocode / ICL)
  5. Run Pipeline
  6. Results Flattening & CSV Export
  7. Ground Truth Collection
  8. Religion Signal Annotation
  9. Evaluation Metrics
  10. Baseline Comparison
  A. Prompt Ablation Study
  ```

---

## 9. Configuration Management for Prompt Selection

### 9.1 — Prompt Styles Hardcoded in Multiple Places

**Issue:** The prompt style selection (direct, pseudocode, icl) is defined multiple times without centralized configuration.

**Locations:**
- Line 125: `STYLES_TO_RUN` configuration
- Line 300: `PROMPT_STYLE_MAP` definition
- Line 910: `ABLATION_STYLES` redefined
- Line 1180: Referenced again for evaluation loop

**Current:**
```python
# Line 125
RUN_ALL_PROMPT_STYLES = True
ACTIVE_PROMPT_STYLE = "direct"
STYLES_TO_RUN = ["direct", "pseudocode", "icl"]

# Line 300
PROMPT_STYLE_MAP = {
    "direct": TASK1_DIRECT,
    "pseudocode": TASK1_PSEUDOCODE,
    "icl": TASK1_ICL
}

# Line 910 (repeated)
ABLATION_STYLES = {
    "direct": TASK1_DIRECT,
    "pseudocode": TASK1_PSEUDOCODE,
    "icl": TASK1_ICL
}
```

**Recommendation:**
- Single prompt registry:
  ```python
  PROMPTS = {
      "direct": {
          "template": TASK1_DIRECT,
          "description": "Simple direct extraction (Liu et al. Table 13)",
          "use_icl": False,
          "use_pseudo_reasoning": False,
      },
      "pseudocode": {
          "template": TASK1_PSEUDOCODE,
          "description": "Step-by-step reasoning (Liu et al. Table 13)",
          "use_icl": False,
          "use_pseudo_reasoning": True,
      },
      "icl": {
          "template": TASK1_ICL,
          "description": "In-context learning example (Liu et al. Figure 2)",
          "use_icl": True,
          "use_pseudo_reasoning": False,
      },
  }
  
  # Configuration
  RUN_CONFIG = {
      'run_all_styles': True,
      'active_style': 'direct',  # Used only if run_all_styles=False
  }
  ```

---

## 10. Error Handling & Logging

### 10.1 — Inconsistent Error Handling & Logging

**Issue:** Error handling is scattered and logging is informal (print statements vs. structured logging).

**Locations:**
- Line 390: Exception handling in `call_groq()` with print
- Line 590: File errors with logging.warning()
- Line 700+: Missing error handling in many scrapers
- Line 1000+: Hard to track which operations fail

**Recommendation:**
- Create `logging_config.py`:
  ```python
  import logging
  
  class PipelineLogger:
      def __init__(self, log_file):
          self.logger = logging.getLogger('pipeline')
          handler = logging.FileHandler(log_file)
          formatter = logging.Formatter(
              '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
          )
          handler.setFormatter(formatter)
          self.logger.addHandler(handler)
      
      def log_extraction_start(self, senator_id, style):
          self.logger.info(f"Starting extraction: {senator_id} with {style}")
      
      def log_api_error(self, senator_id, error, attempt, max_retries):
          self.logger.warning(f"API error for {senator_id}: {error} (attempt {attempt}/{max_retries})")
  ```

---

## 11. Testing & Validation

### 11.1 — Critical Validation Missing

**Issue:** Many operations lack validation:
- No check that HTML files exist before processing
- No validation of extractions (expected fields present)
- No unit tests for parsing functions
- No integration tests for full pipeline

**Recommendation:**
- Create `validation.py`:
  ```python
  class ValidationRules:
      @staticmethod
      def validate_extraction(result, required_fields):
          """Ensure extraction has expected structure"""
          if not isinstance(result, dict):
              raise ValueError(f"Expected dict, got {type(result)}")
          
          missing = [f for f in required_fields if f not in result]
          if missing:
              raise ValueError(f"Missing fields: {missing}")
          
          return result
      
      @staticmethod
      def validate_file_paths(paths_dict):
          """Ensure all required paths exist"""
          for name, path in paths_dict.items():
              if not Path(path).exists():
                  raise FileNotFoundError(f"{name}: {path} does not exist")
  ```

---

## 12. Suggested Modular Refactoring

### High-Priority Changes (20% effort, 70% improvement)

```
notebooks/
├── senate_llm_pipeline.ipynb (reduced to ~1000 lines)
└── modules/
    ├── config.py (paths, prompts, API settings)
    ├── html_processing.py (BeautifulSoup utilities)
    ├── api_client.py (Groq/LLM calls with retry logic)
    ├── baseline_extractors.py (regex, spaCy)
    ├── metrics.py (score functions in classes)
    ├── evaluator.py (evaluation pipeline)
    ├── data_loader.py (CSV loading/caching)
    ├── scrapers.py (Wikipedia, Ballotpedia, Pew)
    ├── name_utils.py (normalization, slug generation)
    ├── validation.py (input/output validation)
    ├── logging_config.py (structured logging)
    ├── diagnostics.py (exploration tools)
    └── utils.py (general utilities)

tests/
├── test_html_processing.py
├── test_metrics.py
├── test_name_utils.py
├── test_api_client.py
└── test_end_to_end.py (mini pipeline on sample data)
```

### Step-by-Step Refactoring Plan

1. **Phase 1 (Immediate):** Extract configuration
   - Move all path/prompt definitions to `config.py`
   - Reduces notebook by ~80 lines

2. **Phase 2 (Week 1):** Extract utility functions
   - `html_processing.py`: `extract_readable_text()`, `extract_infobox()`, etc.
   - `name_utils.py`: `create_slug()`, `normalize_name()`
   - `baseline_extractors.py`: regex/spaCy classes
   - Reduces notebook by ~200 lines

3. **Phase 3 (Week 2):** Extract metrics & evaluation
   - Create `Metric` base class and subclasses
   - Move `evaluate_extraction()` to `evaluator.py`
   - Reduces notebook by ~300 lines

4. **Phase 4 (Week 3):** Extract scrapers & API calls
   - `api_client.py`: `call_groq()` with retry
   - `scrapers.py`: Wikipedia, Ballotpedia, Pew functions
   - Reduces notebook by ~250 lines

5. **Phase 5 (Week 4):** Add tests & documentation
   - Unit tests for each module
   - Integration test on sample (5) senators
   - Updated README with module descriptions

---

## 13. Summary of Redundant Code Patterns

| Redundancy Type | Locations | Lines | Reduction Potential |
|---|---|---|---|
| Duplicate helper functions | 817, 1138 | 300 | Extract to utils (80% reduction) |
| Repeated HTML processing | 156, 522, 1089 | 150 | Centralize in 1 function (90%) |
| Path definitions | 70, 590, 674, etc | 80 | Move to config.py (95%) |
| Evaluation loops (3×) | 1280–1934 | 655 | Refactor to `evaluate_extraction()` (80%) |
| Metric calculations | Throughout evaluation | 300 | Create `Metric` classes (85%) |
| CSV loading | 560, 640, 1280+, etc | 100 | Centralize in `DataPipeline` (90%) |
| Rate limiting | 520, 900, 1000, 1700 | 50 | `RateLimiter` class (100%) |
| Baseline extraction | Section XX | 150 | `Baseline` classes (75%) |
| Error handling | Throughout | 200 | Structured logging (60%) |
| **TOTAL** | | **1985** | **~600 lines (30% reduction)** |

---

## 14. Recommended Priority Implementation

### Quick Wins (Can implement now):

1. ✅ **Extract duplicate `create_slug()` to single function** (1 hour)
   - Create `name_utils.py` with unified implementation
   - Update both Section 8 and 8b to import from it

2. ✅ **Centralize path configuration** (30 minutes)
   - Create `config.py` with all paths
   - Replace hardcoded paths throughout with `PATHS['html_data']` etc.

3. ✅ **Create README documenting pipeline sections** (1 hour)
   - Helps new readers understand flow
   - Lists inputs/outputs per section
   - Notes which sections can run independently

### Medium Term (Improves maintainability):

4. 📊 **Refactor evaluation loops into `evaluate_extraction(style, metrics_config)` function** (4 hours)
   - Reduces 655 lines to ~150
   - Single source of truth for metrics

5. 🔧 **Extract metric calculations into `Metric` classes** (3 hours)
   - Makes adding new metrics trivial
   - Easier to test individually

6. 📂 **Create `data_loader.py` with `DataPipeline` class** (2 hours)
   - Central place for all CSV loading
   - Consistent column naming
   - Caching support for faster re-runs

### Long Term (Enables testing & scaling):

7. 🧪 **Add unit tests** (8–10 hours)
   - Test parsers: `parse_education_detailed()`, `normalize_degree()`, etc.
   - Test baseline extractors
   - Test metric calculations on known examples

8. 📚 **Extract scrapers to standalone module** (4 hours)
   - `WikipediaScraper`, `BallotpeediaScraper`, `PewMatcher` classes
   - Enables faster development (can test scraper independently)

---

## Conclusion

This notebook implements a sophisticated research pipeline with real value, but suffers from:
- **30% code duplication** through repeated helper functions and evaluation logic
- **Scattered configuration** making parameter changes brittle
- **Limited modularity** preventing code reuse outside notebook
- **Informal error handling** making bugs hard to track

**Implementing the refactoring plan would:**
✓ Reduce notebook size by 30% (consolidate logic)  
✓ Enable unit testing (extract functions to modules)  
✓ Improve maintainability (single source of truth for each concept)  
✓ Support future research (research team can import `modules` in other projects)  
✓ Enable parallelizable runs (scrapers/baselines can run independently)  

**Estimated effort:** 25–30 hours of refactoring work  
**Expected payoff:** 10× faster iteration on future experiments
