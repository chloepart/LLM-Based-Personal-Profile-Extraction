"""
Centralized configuration for Senate LLM Extraction Pipeline
Consolidates all paths, API settings, prompts, and pipeline parameters
"""

from pathlib import Path
import datetime

# ============================================================================
# FILE PATHS
# ============================================================================

PATHS = {
    # Data input directories
    'html_data': Path("../external_data/senate_html"),
    'synthetic_data': Path("../data/synthetic"),
    'icl_data': Path("../data/icl"),
    
    # Output directories
    'output_root': Path("../outputs/senate_results"),
    'log_dir': Path("../outputs/log"),
    'result_dir': Path("../outputs/result"),
    
    # Ground truth & external data
    'ground_truth_root': Path("../external_data/ground_truth"),
    'pew_religion': Path("../external_data/pew_religion.csv"),
    'senators_index': Path("../external_data/senate_html/senators_index.csv"),
    
    # Config files
    'groq_config': Path("../configs/model_configs/groq_config_extraction.json"),
    'gemini_config': Path("../configs/model_configs/gemini_config.json"),
    'gpt_config': Path("../configs/model_configs/gpt_config.json"),
}

# Auto-create output directories
PATHS['output_root'].mkdir(parents=True, exist_ok=True)
PATHS['log_dir'].mkdir(parents=True, exist_ok=True)
PATHS['result_dir'].mkdir(parents=True, exist_ok=True)


# ============================================================================
# PROMPT DEFINITIONS
# ============================================================================

# Task 1 — DIRECT style (Liu et al. "direct" prompt, Table 13)
TASK1_DIRECT = """You are a precise data extraction specialist.
Extract the following fields from the senator profile text.
Return ONLY valid JSON. No preamble, no markdown fences.

{
  "full_name": string or null,
  "birthdate": "YYYY-MM-DD" or null,
  "birth_year_inferred": integer or null,
  "gender": string or null,
  "race_ethnicity": string or null,
  "education": [{"degree": string, "institution": string, "year": integer or null}],
  "committee_roles": [string],
  "religious_affiliation": string or null,
  "religious_affiliation_inferred": boolean
}

Rules:
- full_name: Person's complete name. Extract directly from text.
- birthdate: Extract full date if available in YYYY-MM-DD format or MM/DD/YYYY format. Otherwise null.
- birth_year_inferred: Only if birthdate cannot be extracted but age/birth year is mentioned. Otherwise null.
- gender: Extract if explicitly stated OR inferable from pronouns (he/him → "male"). Otherwise null.
- race_ethnicity: Only if explicitly stated. Otherwise null. Do NOT infer.
- education: Array of objects with degree, institution, year. Include all entries found.
- committee_roles: Array of professional roles/committee memberships. Include all found.
- religious_affiliation: Use if mentioned explicitly OR inferred from organizational memberships, values, cultural references.
- religious_affiliation_inferred: Set to true if inferred based on signals; false if explicitly stated.
- Never guess; return null for missing fields
"""

# Task 1 — PSEUDOCODE style
TASK1_PSEUDOCODE = """You are a data extraction assistant. Follow these step-by-step instructions to extract information from the senator profile text below. Return ONLY valid JSON. No preamble, no markdown fences, no code.

Step 1 - Extract full_name: Find the senator's complete name directly from the text.
Step 2 - Extract birthdate: If a full date is present, format as YYYY-MM-DD. Otherwise null.
Step 3 - Extract birth_year_inferred: Only if birthdate is null AND a birth year or age is mentioned. Otherwise null.
Step 4 - Extract gender: Use explicit statement OR pronouns (he/him = "male", she/her = "female", they/them = "non-binary"). Otherwise null.
Step 5 - Extract race_ethnicity: Only if explicitly stated in the text. Do NOT infer. Otherwise null.
Step 6 - Extract education: Find all degree/institution/year entries. Return as array of objects.
Step 7 - Extract committee_roles: Find all committee memberships and professional roles. Return as array of strings.
Step 8 - Extract religious_affiliation: Use if explicitly stated OR inferable from organizational memberships, values language, or cultural references. Otherwise null.
Step 9 - Set religious_affiliation_inferred: true if inferred from signals, false if explicitly stated.

Return this exact JSON structure:
{
  "full_name": string or null,
  "birthdate": "YYYY-MM-DD" or null,
  "birth_year_inferred": integer or null,
  "gender": string or null,
  "race_ethnicity": string or null,
  "education": [{"degree": string, "institution": string, "year": integer or null}],
  "committee_roles": [string],
  "religious_affiliation": string or null,
  "religious_affiliation_inferred": boolean
}

Never guess. Return null for missing fields.
"""

# Task 1 — IN-CONTEXT LEARNING style
TASK1_ICL = """You are a precise data extraction specialist.
Extract the following fields from the senator profile text.
Return ONLY valid JSON. No preamble, no markdown fences.

EXAMPLE INPUT:
Senator Jane Smith is from Ohio. She serves on the Senate Finance Committee and
the Senate Judiciary Committee. She earned a J.D. from Harvard Law School in 1995
and a B.A. from Ohio State University in 1992. She is known for her work on
interfaith initiatives and has spoken at numerous Christian conferences.

EXAMPLE OUTPUT:
{"full_name": "Jane Smith", "birthdate": null, "birth_year_inferred": null,
 "gender": "female", "race_ethnicity": null,
 "education": [{"degree": "J.D.", "institution": "Harvard Law School", "year": 1995},
               {"degree": "B.A.", "institution": "Ohio State University", "year": 1992}],
 "committee_roles": ["Senate Finance Committee", "Senate Judiciary Committee"],
 "religious_affiliation": "Christian", "religious_affiliation_inferred": true}

NOW EXTRACT:
{
  "full_name": string or null,
  "birthdate": "YYYY-MM-DD" or null,
  "birth_year_inferred": integer or null,
  "gender": string or null,
  "race_ethnicity": string or null,
  "education": [{"degree": string, "institution": string, "year": integer or null}],
  "committee_roles": [string],
  "religious_affiliation": string or null,
  "religious_affiliation_inferred": boolean
}

Rules:
- full_name: Person's complete name. Extract directly from text.
- birthdate: Extract full date if available in YYYY-MM-DD format. Otherwise null.
- birth_year_inferred: Only if birthdate cannot be extracted but age/birth year is mentioned. Otherwise null.
- gender: Extract if explicitly stated OR inferable from pronouns (he/him → "male", she/her → "female", they/them → "non-binary"). Otherwise null.
- race_ethnicity: Only if explicitly stated. Otherwise null. Do NOT infer.
- education: Array of objects with degree, institution, year. Include all entries found.
- committee_roles: Array of professional roles/committee memberships. Include all found.
- religious_affiliation: Use if mentioned explicitly OR inferred from organizational memberships, values, cultural references.
- religious_affiliation_inferred: Set to true if inferred based on signals; false if explicitly stated.
- Never guess; return null for missing fields
"""

# Map style strings to prompts
PROMPT_STYLE_MAP = {
    "direct": TASK1_DIRECT,
    "pseudocode": TASK1_PSEUDOCODE,
    "icl": TASK1_ICL
}

# Ablation study uses same prompts
ABLATION_STYLES = {
    "direct": TASK1_DIRECT,
    "pseudocode": TASK1_PSEUDOCODE,
    "icl": TASK1_ICL
}

# Religion signal annotation prompt
EDUCATION_PROMPT = """You are a precise data extraction specialist.
Extract the education history from the text below.
Return ONLY a pipe-delimited string of degree|institution|year entries.
Example: "B.A.|Stanford University|1982|J.D.|Harvard Law School|1986"
If no education found, return null.
Do not explain. Output only the string or null.
"""


# ============================================================================
# PIPELINE CONFIGURATION
# ============================================================================

class PipelineConfig:
    """Main pipeline configuration and session metadata"""
    
    def __init__(self):
        self.timestamp = datetime.datetime.now().isoformat()
        
        # Prompt style settings
        self.run_all_prompt_styles = True
        self.active_prompt_style = "direct"  # Used only if run_all_prompt_styles=False
        self.styles_to_run = ["direct", "pseudocode", "icl"] if self.run_all_prompt_styles else [self.active_prompt_style]
        
        # Rate limiting (in seconds)
        self.inter_senator_delay = 6 if self.run_all_prompt_styles else 4
        self.between_styles_delay = 3
        self.web_scrape_delay = 1.5
        self.baseline_delay = 0.5
        
        # Ablation settings
        self.ablation_size = 25
        self.ablation_random_seed = 42
        
        # API retry settings
        self.max_api_retries = 5
        self.retry_backoff_base = 2  # Exponential backoff: 2^attempt seconds
        self.retry_backoff_max = 120  # Max 120 second wait between retries
        
    @property
    def active_prompt_name(self):
        """Return name of active prompt style"""
        if self.run_all_prompt_styles:
            return "all_styles"
        return self.active_prompt_style
    
    @property
    def session_metadata(self):
        """Return dict of session metadata for logging"""
        return {
            "timestamp": self.timestamp,
            "prompt_style": self.active_prompt_name,
            "run_all_styles": self.run_all_prompt_styles,
            "inter_senator_delay": self.inter_senator_delay,
            "between_styles_delay": self.between_styles_delay,
        }


# ============================================================================
# EXTRACTION FIELDS DEFINITION
# ============================================================================

# Task 1 fields to extract
T1_FIELDS = [
    "full_name",
    "birthdate",
    "birth_year_inferred",
    "gender",
    "race_ethnicity",
    "education",
    "committee_roles",
    "religious_affiliation",
    "religious_affiliation_inferred"
]

# Fields for comparison/evaluation (excluding inferred boolean fields)
T1_FIELDS_CMP = [
    "full_name",
    "birthdate",
    "birth_year_inferred",
    "gender",
    "race_ethnicity",
    "education",
    "committee_roles",
    "religious_affiliation",
]

# Ground truth fields harvested from Wikipedia/Ballotpedia
GT_FIELDS = [
    "name",
    "state",
    "full_name",
    "birthdate",
    "gender",
    "race_ethnicity",
    "committee_roles",
    "religion"
]


# ============================================================================
# REGEX PATTERNS FOR BASELINE EXTRACTION
# ============================================================================

import re

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(?:\+\d+\s?)?(?:\d{3}[\-\s]?\d{3,}[\-\s]?\d{4})")
YEAR_RE = re.compile(r"\b(19[4-9]\d|20[0-2]\d)\b")
NAME_RE = re.compile(r"\bSenator\s+([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)")
PARTY_KEYWORD_RE = re.compile(r"\b(Republican|Democrat|Democratic|Independent)\b", re.IGNORECASE)

REGEX_PATTERNS = {
    'email': EMAIL_RE,
    'phone': PHONE_RE,
    'year': YEAR_RE,
    'name': NAME_RE,
    'party': PARTY_KEYWORD_RE,
}


# ============================================================================
# RELIGION HIERARCHY FOR HIERARCHICAL MATCHING
# ============================================================================

RELIGION_HIERARCHY = {
    # Catholic tradition
    "catholic": "catholic",
    "roman catholic": "catholic",
    
    # Christian denominations ALL map to "christian"
    "methodist": "christian",
    "united methodist": "christian",
    "methodist church": "christian",
    "baptist": "christian",
    "southern baptist": "christian",
    "presbyterian": "christian",
    "episcopal": "christian",
    "episcopalian": "christian",
    "lutheran": "christian",
    "evangelical": "christian",
    "pentecostal": "christian",
    "assemblies of god": "christian",
    
    # Broadly Christian (covers generic "Christian" answers)
    "christian": "christian",
    "christian (unspecified)": "christian",
    "protestant": "christian",
    
    # Other major religions
    "jewish": "jewish",
    "judaism": "jewish",
    "muslim": "muslim",
    "islam": "muslim",
    "orthodox": "orthodox",
    "mormon": "mormon",
    "church of jesus christ": "mormon",
    "lds": "mormon",
    "unitarian": "unitarian",
    "unitarian universalist": "unitarian",
    
    # Secular / None
    "atheist": "none",
    "agnostic": "none",
    "none": "none",
    "no religion": "none",
    "unaffiliated": "none",
}
