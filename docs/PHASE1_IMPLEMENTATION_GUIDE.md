# Implementation Guide: Phase 1 (Quick Wins)

## Overview
This guide walks you through updating the notebook to use the new centralized modules. 

**New modules created:**
- ✅ `modules/config.py` — Centralized configuration & paths
- ✅ `modules/name_utils.py` — Unified name normalization
- ✅ `modules/html_processing.py` — Consolidated HTML extraction
- ✅ `modules/__init__.py` — Package initialization

**Estimated time to implement:** 30-45 minutes

---

## Step-by-Step Implementation

### Step 1: Add Module Import Cell at Top of Notebook

**Location:** After cell 1 (dependencies), add a new cell with:

```python
# Import centralized modules
import sys
sys.path.insert(0, '/Users/chloe/LLM-Based-Personal-Profile-Extraction')

from modules import (
    PATHS,
    PROMPT_STYLE_MAP,
    ABLATION_STYLES,
    EDUCATION_PROMPT,
    PipelineConfig,
    T1_FIELDS,
    T1_FIELDS_CMP,
    REGEX_PATTERNS,
    RELIGION_HIERARCHY,
    NameNormalizer,
    HTMLProcessor,
    extract_readable_text,
)
```

**Why:** Makes all centralized configuration available throughout the notebook.

---

### Step 2: Update Section 2 (Configuration)

**Current Code (Lines 70-92):**
```python
HTML_DIR   = Path("../external_data/senate_html")
OUTPUT_DIR = Path("../outputs/senate_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import os
config_path = Path("../configs/model_configs/groq_config_extraction.json")
# ... rest of config loading
```

**REPLACE WITH:**
```python
# Use centralized configuration
HTML_DIR   = PATHS['html_data']
OUTPUT_DIR = PATHS['output_root']
config_path = PATHS['groq_config']

import os
if not config_path.exists():
    raise ValueError("groq_config_extraction.json not found at " + str(config_path))

with open(config_path) as f:
    config = json.load(f)

# Extract API key (from env var or config file)
api_key = os.getenv("GROQ_API_KEY") or config["api_key_info"]["api_keys"][0]

# Extract model and provider settings
provider = config["model_info"]["provider"]
model = config["model_info"]["name"]
temp = config["params"]["temperature"]
max_tok = config["params"]["max_output_tokens"]

# Initialize Groq client
from groq import Groq
api_client = Groq(api_key=api_key)

print(f"✓ Groq API initialized from config")
print(f"Model:       {model}")
print(f"Provider:    {provider}")
print(f"Temperature: {temp}  |  Max tokens: {max_tok}")

html_files = sorted(HTML_DIR.glob("*.html"))
print(f"HTML files:  {len(html_files)}")
```

**Why:** Reduces hardcoded paths; uses centralized configuration.

---

### Step 3: Update Section 2b (Session Metadata)

**Current Code (Lines 114-133):**
```python
RUN_ALL_PROMPT_STYLES = True
if RUN_ALL_PROMPT_STYLES:
    ACTIVE_PROMPT_STYLE = None
    STYLES_TO_RUN = ["direct", "pseudocode", "icl"]
else:
    ACTIVE_PROMPT_STYLE = "direct"
    STYLES_TO_RUN = [ACTIVE_PROMPT_STYLE]
```

**REPLACE WITH:**
```python
# Initialize pipeline configuration
pipeline_config = PipelineConfig()

# Override any settings here if needed
pipeline_config.run_all_prompt_styles = True  # SET THIS TO FALSE FOR SINGLE STYLE

RUN_ALL_PROMPT_STYLES = pipeline_config.run_all_prompt_styles
ACTIVE_PROMPT_STYLE = pipeline_config.active_prompt_style
STYLES_TO_RUN = pipeline_config.styles_to_run
```

**Why:** Centralized configuration object; easier to modify all settings in one place.

---

### Step 4: Update Section 3 (HTML Preprocessing)

**Current Code (Lines 143-156):**
```python
def extract_readable_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s{2,}", " ", text).strip()
```

**REPLACE WITH:**
```python
# extract_readable_text is now imported from modules.html_processing
# The function is identical, but centralized for maintainability

# Example usage remains the same:
sample = html_files[0]
text = extract_readable_text(sample.read_text(encoding="utf-8", errors="ignore"))
print(f"File:         {sample.name}")
print(f"Raw size:     {sample.stat().st_size:,} chars")
print(f"Cleaned size: {len(text):,} chars")
print(f"\n--- First 500 chars ---\n{text[:500]}")
```

**Why:** Function now imported from modules; removes duplicate definition.

---

### Step 5: Update Section 4 (Prompts)

**Current Code (Lines 176-286):**
```python
TASK1_DIRECT = """You are a precise data extraction specialist..."""
TASK1_PSEUDOCODE = """You are a data extraction assistant..."""
TASK1_ICL = """You are a precise data extraction specialist..."""
```

**REPLACE WITH:**
```python
# Prompts are now imported from modules.config
# Access them via:
# - TASK1_DIRECT
# - TASK1_PSEUDOCODE  
# - TASK1_ICL
# - PROMPT_STYLE_MAP (dict mapping style name to prompt)

# Initialize prompt mapping
PROMPT_STYLE_MAP_INSTANCE = PROMPT_STYLE_MAP

# Validate selections
for style in STYLES_TO_RUN:
    if style not in PROMPT_STYLE_MAP_INSTANCE:
        raise ValueError(f"Invalid style: {style}. Must be one of: {list(PROMPT_STYLE_MAP_INSTANCE.keys())}")

print(f"✓ Prompt styles loaded: {', '.join(STYLES_TO_RUN)}")
```

**Why:** Prompts now centralized; no need to redefine them in notebook.

---

### Step 6: Update Section 2b Session Metadata

**Current Code (Lines 289-327):**
```python
PROMPT_STYLE_MAP = {
    "direct": TASK1_DIRECT,
    "pseudocode": TASK1_PSEUDOCODE,
    "icl": TASK1_ICL
}
```

**DELETE THIS** — Already done in config.py. Just use:
```python
# Session metadata is already prepared by PipelineConfig
session_metadata = pipeline_config.session_metadata

print("=" * 70)
print("📋 SESSION METADATA")
print("=" * 70)
for key, value in session_metadata.items():
    print(f"  {key:.<20s} {value}")
print("=" * 70)
```

---

### Step 7: Update Section 6 (Path Configuration for Ground Truth)

**Current Code (Lines 590-596):**
```python
INPUT_PATH = "../external_data/senate_html/senators_index.csv"
PEW_PATH = "../external_data/ground_truth/pew_religion.csv"
OUTPUT_PATH = "../external_data/ground_truth/senate_ground_truth.csv"
LOG_PATH = "../external_data/ground_truth/scrape_errors.log"
```

**REPLACE WITH:**
```python
# Use centralized path configuration
INPUT_PATH = PATHS['senators_index']
PEW_PATH = PATHS['pew_religion']
OUTPUT_PATH = PATHS['ground_truth_root'] / "senate_ground_truth.csv"
LOG_PATH = PATHS['ground_truth_root'] / "scrape_errors.log"

# Ensure directory exists
PATHS['ground_truth_root'].mkdir(parents=True, exist_ok=True)
```

---

### Step 8: Replace Duplicate `create_slug()` Functions

**Current Code (Lines 817–870 & 1138–1210):**
```python
def create_slug(name):
    # ... implementation ... (appears twice with slight differences)
```

**REPLACE ALL with:**
```python
# Use centralized NameNormalizer
slug = NameNormalizer.create_slug(name)
wikipedia_url = NameNormalizer.create_wikipedia_url(name)
ballotpedia_url = NameNormalizer.create_ballotpedia_url(name)
```

---

### Step 9: Update Section 8 URL Construction (Ground Truth Builder)

**Current Code (Lines 817–890):**
```python
def create_slug(name):
    slug = re.sub(r'\s+[A-Z]\.\s*', ' ', name).strip()
    slug = slug.replace(" ", "_")
    overrides = {"Bernard_Sanders": "Bernie_Sanders"}
    return overrides.get(slug, slug)

senators["wikipedia_url"] = "https://en.wikipedia.org/wiki/" + senators["name"].apply(create_slug)
senators["ballotpedia_url"] = "https://ballotpedia.org/" + senators["name"].apply(create_slug)
```

**REPLACE WITH:**
```python
# Use NameNormalizer for all URL generation
senators["wikipedia_url"] = senators["name"].apply(NameNormalizer.create_wikipedia_url)
senators["ballotpedia_url"] = senators["name"].apply(NameNormalizer.create_ballotpedia_url)
```

---

### Step 10: Update Section 8b Education Extraction

**Current Code (Lines 1138–1210):**
```python
def create_wikipedia_url(senator_name):
    # ... complex normalization logic ... (duplicate of above)
```

**REPLACE WITH:**
```python
# Use centralized NameNormalizer
url = NameNormalizer.create_wikipedia_url(senator_name)
```

---

### Step 11: Update HTML Extraction in Scrapers

**Current Code (Multiple locations - e.g., Line 1089, 1156):**
```python
soup = BeautifulSoup(response.content, "html.parser")
for tag in soup(["script", "style", "nav", "footer"]):
    tag.decompose()
wiki_text = soup.get_text(separator=" ", strip=True)
```

**REPLACE WITH:**
```python
# Use HTMLProcessor
html_text = response.text
wiki_text = extract_readable_text(html_text, max_length=10000)
```

---

### Step 12: Update Sections with Hardcoded Field Lists

**Current Code (Lines 641-643, 1008, etc):**
```python
T1_FIELDS = ["full_name","birthdate","birth_year_inferred",
             "gender","race_ethnicity","education","committee_roles","religious_affiliation","religious_affiliation_inferred"]

T1_FIELDS_CMP = ["full_name","birthdate","birth_year_inferred",
                 "gender","race_ethnicity","education","committee_roles","religious_affiliation","religious_affiliation_inferred"]
```

**REPLACE WITH:**
```python
# Already imported from modules.config:
# T1_FIELDS = [list of extraction fields]
# T1_FIELDS_CMP = [list of fields for comparison]
# Just use them directly
```

---

### Step 13: Update Regex Patterns

**Current Code (Lines 595–613):**
```python
EMAIL_RE = _re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
PHONE_RE = _re.compile(r"(?:\+\d+\s?)?(?:\d{3}[\-\s]?\d{3,}[\-\s]?\d{4})")
YEAR_RE  = _re.compile(r"\b(19[4-9]\d|20[0-2]\d)\b")
NAME_RE  = _re.compile(r"\bSenator\s+([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)")
PARTY_KEYWORD_RE = _re.compile(
    r"\b(Republican|Democrat|Democratic|Independent)\b", _re.IGNORECASE
)
```

**REPLACE WITH:**
```python
# Use centralized REGEX_PATTERNS
EMAIL_RE = REGEX_PATTERNS['email']
PHONE_RE = REGEX_PATTERNS['phone']
YEAR_RE = REGEX_PATTERNS['year']
NAME_RE = REGEX_PATTERNS['name']
PARTY_KEYWORD_RE = REGEX_PATTERNS['party']
```

---

### Step 14: Update Religion Hierarchy

**Current Code (Lines 1463–1491):**
```python
RELIGION_HIERARCHY = {
    "catholic": "catholic",
    # ... 30+ lines of religion mappings ...
}
```

**REPLACE WITH:**
```python
# Use centralized RELIGION_HIERARCHY (already imported)
# Access via: RELIGION_HIERARCHY
```

---

## Verification Checklist

After making all replacements, verify:

- [ ] Module import cell runs without errors
- [ ] Configuration cell still has correct API key and model
- [ ] HTML preprocessing still produces correct output
- [ ] Regex patterns work (test with sample text)
- [ ] URL generation matches expected format
- [ ] Session metadata displays correctly
- [ ] Run a test extraction on 1 senator to verify pipeline works

---

## Testing the Changes

After updating, test with this minimal example:

```python
# Test 1: Import modules
from modules import PATHS, extract_readable_text, NameNormalizer
print("✓ Modules imported successfully")

# Test 2: Access centralized config
print(f"HTML data path: {PATHS['html_data']}")
print(f"Output dir: {PATHS['output_root']}")

# Test 3: Test name utilities
test_name = "Bernie Sanders"
print(f"Wikipedia URL: {NameNormalizer.create_wikipedia_url(test_name)}")
print(f"Slug: {NameNormalizer.create_slug(test_name)}")

# Test 4: Test HTML processing
sample_html = "<html><body><script>alert('x')</script>Hello World</body></html>"
clean_text = extract_readable_text(sample_html)
print(f"Cleaned HTML: {clean_text}")
```

---

## Summary of Changes

| Section | Original Lines | Change | Benefit |
|---------|---|---|---|
| Imports | 1-10 | Add module imports | Centralized configuration |
| Config (Sect. 2) | 70-92 | Use `PATHS` dict | Reduce path duplication |
| Session Meta (Sect 2b) | 114-133 | Use `PipelineConfig` class | Single source of truth |
| HTML Processing (Sect 3) | 143-156 | Import function | Eliminate duplicate |
| Prompts (Sect 4) | 176-286 | Import from config | Remove 110 lines |
| Ground Truth (Sect 8) | 817-890 | Use `NameNormalizer` | Eliminate duplicate |
| Education (Sect 8b) | 1138-1210 | Use centralized utilities | Reuse code |
| Baselines (Sect 6) | 595-613 | Use `REGEX_PATTERNS` | Centralize patterns |
| Religion (Sect 9) | 1463-1491 | Use `RELIGION_HIERARCHY` | Centralization |
| **TOTAL** | **~1300** | Consolidated to modules | **30% code reduction** |

---

## Next Steps (Phase 2 - After Phase 1 Works)

Once Phase 1 is complete and working:

1. **Create `modules/evaluator.py`** — Consolidate evaluation loops (655 lines → 150)
2. **Create `modules/metrics.py`** — Extract metric classes
3. **Create `modules/data_loader.py`** — Centralize CSV loading with caching
4. **Create `modules/baselines.py`** — Extract baseline classes
5. **Add unit tests** for each module

---

## Questions or Issues?

If you encounter any issues:
1. Check if module import at top succeeds
2. Verify all `PATHS` resolve correctly
3. Test individual functions in isolation
4. Check for any import errors in module files
