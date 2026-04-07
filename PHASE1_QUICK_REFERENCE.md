# Quick Reference: Before & After Code Snippets

## Module Import (Add at top of notebook after dependencies)

### BEFORE: Paths Scattered Throughout
```python
HTML_DIR   = Path("../external_data/senate_html")
OUTPUT_DIR = Path("../outputs/senate_results")
INPUT_PATH = "../external_data/senate_html/senators_index.csv"
PEW_PATH = "../external_data/ground_truth/pew_religion.csv"
```

### AFTER: Centralized Import
```python
from modules import PATHS, extract_readable_text, NameNormalizer, HTMLProcessor
from modules import PROMPT_STYLE_MAP, RELIGION_HIERARCHY, T1_FIELDS

HTML_DIR   = PATHS['html_data']
OUTPUT_DIR = PATHS['output_root']
INPUT_PATH = PATHS['senators_index']
PEW_PATH = PATHS['pew_religion']
```

---

## HTML Extraction Function

### BEFORE: Function Definition in Notebook
```python
def extract_readable_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s{2,}", " ", text).strip()
```

### AFTER: Direct Usage (Function Imported)
```python
# Already imported at top: from modules import extract_readable_text

text = extract_readable_text(html)
# All same functionality, but no duplicate definition
```

---

## Name Slug Generation

### BEFORE: Duplicate Functions (2 definitions!)
```python
# Section 8
def create_slug(name):
    slug = re.sub(r'\s+[A-Z]\.\s*', ' ', name).strip()
    slug = slug.replace(" ", "_")
    overrides = {"Bernard_Sanders": "Bernie_Sanders"}
    return overrides.get(slug, slug)

# Section 8b (similar but with more logic)
def create_wikipedia_url(senator_name):
    norm_name = str(senator_name).lower().strip()
    # ... more complex logic ...
    return "https://en.wikipedia.org/wiki/" + slug
```

### AFTER: Single Centralized Version
```python
# Import at top: from modules import NameNormalizer

url = NameNormalizer.create_wikipedia_url(name)
ballotpedia_url = NameNormalizer.create_ballotpedia_url(name)
slug = NameNormalizer.create_slug(name)
```

---

## Configuration & Session Setup

### BEFORE: Scattered Configuration
```python
RUN_ALL_PROMPT_STYLES = True
if RUN_ALL_PROMPT_STYLES:
    ACTIVE_PROMPT_STYLE = None
    STYLES_TO_RUN = ["direct", "pseudocode", "icl"]
else:
    ACTIVE_PROMPT_STYLE = "direct"
    STYLES_TO_RUN = [ACTIVE_PROMPT_STYLE]

INTER_SENATOR_DELAY = 6 if RUN_ALL_PROMPT_STYLES else 4

# Then later...
PROMPT_STYLE_MAP = {
    "direct": TASK1_DIRECT,
    "pseudocode": TASK1_PSEUDOCODE,
    "icl": TASK1_ICL
}
```

### AFTER: Centralized Configuration Object
```python
from modules import PipelineConfig, PROMPT_STYLE_MAP

# Create configuration
pipeline_config = PipelineConfig()

# Override defaults if needed
pipeline_config.run_all_prompt_styles = True
pipeline_config.inter_senator_delay = 6

# Access via object
RUN_ALL_PROMPT_STYLES = pipeline_config.run_all_prompt_styles
INTER_SENATOR_DELAY = pipeline_config.inter_senator_delay
STYLES_TO_RUN = pipeline_config.styles_to_run

# PROMPT_STYLE_MAP already imported
```

---

## Prompt Definitions

### BEFORE: 110 Lines in Notebook
```python
TASK1_DIRECT = """You are a precise data extraction specialist..."""
TASK1_PSEUDOCODE = """You are a data extraction assistant..."""
TASK1_ICL = """You are a precise data extraction specialist..."""

PROMPT_STYLE_MAP = {
    "direct": TASK1_DIRECT,
    "pseudocode": TASK1_PSEUDOCODE,
    "icl": TASK1_ICL
}
```

### AFTER: Imported from Config
```python
# At top: from modules import TASK1_DIRECT, TASK1_PSEUDOCODE, TASK1_ICL, PROMPT_STYLE_MAP

# Access directly, no definition needed
```

---

## Regex Patterns

### BEFORE: 5 Pattern Definitions
```python
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(?:\+\d+\s?)?(?:\d{3}[\-\s]?\d{3,}[\-\s]?\d{4})")
YEAR_RE  = re.compile(r"\b(19[4-9]\d|20[0-2]\d)\b")
NAME_RE  = re.compile(r"\bSenator\s+([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)")
PARTY_KEYWORD_RE = re.compile(r"\b(Republican|Democrat|Democratic|Independent)\b", re.IGNORECASE)
```

### AFTER: Use Centralized Dict
```python
# At top: from modules import REGEX_PATTERNS

EMAIL_RE = REGEX_PATTERNS['email']
PHONE_RE = REGEX_PATTERNS['phone']
YEAR_RE = REGEX_PATTERNS['year']
NAME_RE = REGEX_PATTERNS['name']
PARTY_KEYWORD_RE = REGEX_PATTERNS['party']
```

---

## Field Definitions

### BEFORE: Repeated Definitions
```python
# Line 641-643
T1_FIELDS = ["full_name","birthdate","birth_year_inferred",
             "gender","race_ethnicity","education","committee_roles","religious_affiliation","religious_affiliation_inferred"]

# Line 1008  
T1_FIELDS_CMP = ["full_name","birthdate","birth_year_inferred",
                 "gender","race_ethnicity","education","committee_roles","religious_affiliation","religious_affiliation_inferred"]
```

### AFTER: Single Import
```python
# At top: from modules import T1_FIELDS, T1_FIELDS_CMP

# Use directly throughout notebook
for field in T1_FIELDS:
    # process field
```

---

## Religion Hierarchy

### BEFORE: ~30 Lines in Notebook
```python
RELIGION_HIERARCHY = {
    "catholic": "catholic",
    "roman catholic": "catholic",
    "methodist": "christian",
    "united methodist": "christian",
    # ... many more entries ...
    "atheist": "none",
    "agnostic": "none",
}
```

### AFTER: Imported Dict
```python
# At top: from modules import RELIGION_HIERARCHY

# Use directly:
category = RELIGION_HIERARCHY.get(religion_str, religion_str)
```

---

## BeautifulSoup HTML Processing

### BEFORE: Manual Processing (Repeated)
```python
soup = BeautifulSoup(response.content, "html.parser")
for tag in soup(["script", "style", "nav", "footer"]):
    tag.decompose()
text = soup.get_text(separator=" ", strip=True)
```

### AFTER: Use HTMLProcessor
```python
from modules import extract_readable_text, HTMLProcessor

# Simple usage:
text = extract_readable_text(html_content, max_length=10000)

# Advanced usage:
processor = HTMLProcessor()
title = processor.extract_page_title(html)
infobox = processor.extract_infobox(html)
links = processor.extract_links(html, filter_text="committee")
```

---

## URL Generation for Scrapers

### BEFORE: Manual URL Construction
```python
def create_slug(name):
    slug = re.sub(r'\s+[A-Z]\.\s*', ' ', name).strip()
    slug = slug.replace(" ", "_")
    # Handle abbreviations, overrides...
    return slug

senators["wikipedia_url"] = "https://en.wikipedia.org/wiki/" + senators["name"].apply(create_slug)
senators["ballotpedia_url"] = "https://ballotpedia.org/" + senators["name"].apply(create_slug)
```

### AFTER: Use NameNormalizer
```python
from modules import NameNormalizer

senators["wikipedia_url"] = senators["name"].apply(NameNormalizer.create_wikipedia_url)
senators["ballotpedia_url"] = senators["name"].apply(NameNormalizer.create_ballotpedia_url)
```

---

## Path Usage Throughout Notebook

### BEFORE: Hardcoded Paths
```python
# Different places in notebook:
output_path = OUTPUT_DIR / "results.csv"
log_path = "../outputs/log/extraction.log"
ground_truth_path = "../external_data/ground_truth/senate_ground_truth.csv"
```

### AFTER: Centralized Paths
```python
from modules import PATHS

output_path = PATHS['output_root'] / "results.csv"
log_path = PATHS['log_dir'] / "extraction.log"
ground_truth_path = PATHS['ground_truth_root'] / "senate_ground_truth.csv"

# Benefits:
# - Single place to modify if directory structure changes
# - Consistent path handling across project
# - Easy to support different environments (dev/prod)
```

---

## Import All at Once

The complete set of imports needed at the top of the notebook:

```python
# ============================================================================
# IMPORT CENTRALIZED MODULES
# ============================================================================
import sys
sys.path.insert(0, '/Users/chloe/LLM-Based-Personal-Profile-Extraction')

from modules import (
    # Configuration
    PATHS,
    TASK1_DIRECT,
    TASK1_PSEUDOCODE,
    TASK1_ICL,
    PROMPT_STYLE_MAP,
    ABLATION_STYLES,
    EDUCATION_PROMPT,
    PipelineConfig,
    
    # Field definitions
    T1_FIELDS,
    T1_FIELDS_CMP,
    GT_FIELDS,
    
    # Patterns & hierarchies
    REGEX_PATTERNS,
    RELIGION_HIERARCHY,
    
    # Utilities
    NameNormalizer,
    HTMLProcessor,
    WikipediaExtractor,
    extract_readable_text,
    extract_infobox,
)

print("✓ All modules imported successfully")
```

---

## Summary of Reductions

| Component | Original Lines | After Import | Reduction |
|-----------|---|---|---|
| Path definitions | 40+ | 5 | 87% ↓ |
| Prompt definitions | 110 | 0 (imported) | 100% ↓ |
| HTML extraction | 8 × N | 1 (func) | 87% ↓ |
| Name slug generation | 50 × N | 3 (calls) | 94% ↓ |
| Config variables | 30+ | 5 | 83% ↓ |
| Regex patterns | 20+ | 5 | 75% ↓ |
| Field definitions | 30+ | 0 (imported) | 100% ↓ |
| Religion hierarchy | 30 | 0 (imported) | 100% ↓ |
| **TOTAL** | **~1300** | **~600** | **~52%** |

Each location these patterns appeared is consolidated into a single definition!
