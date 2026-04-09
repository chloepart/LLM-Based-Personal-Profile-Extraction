"""
Unified configuration for Senate LLM Extraction Pipeline
Consolidates all paths, API settings, prompts, and pipeline parameters with improved structure
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any


# ============================================================================
# PROJECT ROOT RESOLUTION
# ============================================================================

def get_project_root() -> Path:
    """Resolve project root using the location of this config file"""
    return Path(__file__).parent.parent


# ============================================================================
# FILE PATHS
# ============================================================================

PATHS = {
    # Base project root
    'project_root': get_project_root(),
    
    # Data input directories
    'html_data': get_project_root() / 'external_data' / 'senate_html',
    'synthetic_data': get_project_root() / 'data' / 'synthetic',
    'icl_data': get_project_root() / 'data' / 'icl',
    
    # Output directories
    'output_root': get_project_root() / 'outputs' / 'senate_results',
    'log_dir': get_project_root() / 'outputs' / 'log',
    'result_dir': get_project_root() / 'outputs' / 'result',
    
    # Ground truth & external data
    'ground_truth_root': get_project_root() / 'external_data' / 'ground_truth',
    'pew_religion': get_project_root() / 'external_data' / 'pew_religion.csv',
    
    # Config files
    'groq_config': get_project_root() / 'configs' / 'model_configs' / 'groq_config_extraction.json',
    'gemini_config': get_project_root() / 'configs' / 'model_configs' / 'gemini_config.json',
    'gpt_config': get_project_root() / 'configs' / 'model_configs' / 'gpt_config.json',
}

# Auto-create output directories
for path_key in ['output_root', 'log_dir', 'result_dir']:
    PATHS[path_key].mkdir(parents=True, exist_ok=True)


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
Step 5 - Extract race_ethnicity: Only if explicitly stated in the text. Otherwise null. Do NOT infer.
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

EXAMPLE1 INPUT:
Senator Jane Smith is from Ohio. She serves on the Senate Finance Committee and
the Senate Judiciary Committee. She earned a J.D. from Harvard Law School in 1995
and a B.A. from Ohio State University in 1992. She is known for her work on
Church initiatives and has spoken at numerous Christian conferences.

EXAMPLE1 OUTPUT:
{"full_name": "Jane Smith", "birthdate": null, "birth_year_inferred": null,
 "gender": "female", "race_ethnicity": null,
 "education": [{"degree": "J.D.", "institution": "Harvard Law School", "year": 1995},
               {"degree": "B.A.", "institution": "Ohio State University", "year": 1992}],
 "committee_roles": ["Senate Finance Committee", "Senate Judiciary Committee"],
 "religious_affiliation": "Christian", "religious_affiliation_inferred": true}

EXAMPLE2 INPUT:
Senator Joe Schmoe is a Republican from Missouri. He graduated from high school in 1968 and went to 
D'Youville College but did not graduate. He enjoys going to his local Mosque. He has served on the Agriculture Committee
because he enjoyed cow-tipping when he was young.

EXAMPLE2 OUTPUT:
{"full_name": "Joe Schmoe", "birthdate": null, "birth_year_inferred": 1950,
 "gender": "male", "race_ethnicity": null,
 "education": [{"degree": null, "institution": "D'Youville College", "year": null}],
 "committee_roles": ["Agriculture Committee"], "religious_affiliation": "Muslim", "religious_affiliation_inferred": true}

EXAMPLE3 INPUT:
Senator John Doe is a Democrat from Florida. He was born on July 4, 1979 and graduated from the 
University of Wisconsin, earning a B.S. in Political Science in 2001. He has served on the Armed Services Committee and the Veterans' Affairs Committee.
He enjoys celebrating Christmas and Easter with his family, but has not spoken publicly about his religious beliefs.

EXAMPLE3 OUTPUT:
{"full_name": "John Doe", "birthdate": "1979-07-04", "birth_year_inferred": null,
 "gender": "male", "race_ethnicity": null,
 "education": [{"degree": "B.S.", "institution": "University of Wisconsin", "year": 2001}],
 "committee_roles": ["Armed Services Committee", "Veterans' Affairs Committee"], "religious_affiliation": null, "religious_affiliation_inferred": false}

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

Never guess. Return null for missing fields other than religious_affiliation_inferred..
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

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(?:\+\d+\s?)?(?:\d{3}[\-\s]?\d{3,}[\-\s]?\d{4})")
YEAR_RE = re.compile(r"\b(19[4-9]\d|20[0-2]\d)\b")
NAME_RE = re.compile(r"\bSenator\s+([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)")
PARTY_KEYWORD_RE = re.compile(r"\b(Republican|Democrat|Democratic|Independent)\b", re.IGNORECASE)

REGEX_PATTERNS = {
    'EMAIL_RE': EMAIL_RE,
    'PHONE_RE': PHONE_RE,
    'YEAR_RE': YEAR_RE,
    'NAME_RE': NAME_RE,
    'PARTY_KEYWORD_RE': PARTY_KEYWORD_RE,
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


# ============================================================================
# DATACLASSES - CONFIGURATION COMPONENTS
# ============================================================================

@dataclass
class RateLimitConfig:
    """Rate limiting and delay settings for API and web scraping"""
    inter_senator_delay: float = 6.0
    inter_style_delay: float = 3.0
    web_scrape_delay: float = 1.5
    backoff_base: int = 2
    backoff_max: int = 120
    
    def get_inter_senator_delay(self, num_prompt_styles: int) -> float:
        """
        Adjust inter-senator delay based on number of prompt styles.
        If running multiple styles, add proportional delay between styles.
        
        Args:
            num_prompt_styles: Number of prompt styles being evaluated
            
        Returns:
            Adjusted delay in seconds
        """
        total_style_delay = (num_prompt_styles - 1) * self.inter_style_delay
        return self.inter_senator_delay + total_style_delay


@dataclass
class AblationConfig:
    """Ablation study settings"""
    enabled: bool = False
    subset_size: int = 25
    random_seed: int = 42
    
    def validate(self, total_senators: int) -> bool:
        """
        Validate ablation configuration against total senator count.
        
        Args:
            total_senators: Total number of senators available
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        if self.enabled and self.subset_size > total_senators:
            raise ValueError(
                f"Ablation subset_size ({self.subset_size}) cannot exceed "
                f"total senators ({total_senators})"
            )
        if self.subset_size <= 0:
            raise ValueError("subset_size must be positive")
        if self.random_seed < 0:
            raise ValueError("random_seed cannot be negative")
        return True


@dataclass
class APIConfig:
    """API and authentication settings"""
    provider: str = "groq"
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    max_retries: int = 5
    max_tokens: int = 2048
    temperature: float = 0.0
    
    def __post_init__(self):
        """Validate API configuration after initialization"""
        valid_providers = ["groq", "openai", "gemini", "local"]
        if self.provider not in valid_providers:
            raise ValueError(f"provider must be one of {valid_providers}, got {self.provider}")
        
        if self.provider != "local" and not self.api_key:
            env_key_name = f"{self.provider.upper()}_API_KEY"
            self.api_key = os.getenv(env_key_name)
            if not self.api_key:
                raise ValueError(
                    f"API key required for {self.provider}. "
                    f"Set {env_key_name} environment variable or pass api_key parameter."
                )
        
        if not self.model_name:
            raise ValueError("model_name is required")
        
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")


@dataclass
class PipelineConfig:
    """Main pipeline configuration composing API, rate limit, and ablation settings"""
    api_config: APIConfig = field(default_factory=lambda: APIConfig(provider="groq"))
    rate_limit_config: RateLimitConfig = field(default_factory=RateLimitConfig)
    ablation_config: AblationConfig = field(default_factory=AblationConfig)
    
    prompt_styles: List[str] = field(default_factory=lambda: ["direct", "pseudocode", "icl"])
    prompt_map: Dict[str, str] = field(default_factory=dict)  # Add this
    run_all_styles: bool = True
    
    _api_client: Optional[Any] = field(default=None, init=False, repr=False)
    _output_dir: Optional[Path] = field(default=None, init=False, repr=False)
    _html_dir: Optional[Path] = field(default=None, init=False, repr=False)
    
    @property
    def model(self) -> str:
        """Convenience accessor for model name"""
        return self.api_config.model_name
    
    @property
    def temperature(self) -> float:
        """Convenience accessor for temperature setting"""
        return self.api_config.temperature
    
    @property
    def max_tokens(self) -> int:
        """Convenience accessor for max_tokens setting"""
        return self.api_config.max_tokens
    
    @property
    def api_client(self):
        """Lazy-initialized API client for Groq"""
        if self._api_client is None:
            from groq import Groq
            self._api_client = Groq(api_key=self.api_config.api_key)
        return self._api_client
    
    @property
    def output_dir(self) -> Path:
        """Output directory for results"""
        if self._output_dir is None:
            self._output_dir = PATHS['output_root']
        return self._output_dir
    
    @property
    def html_dir(self) -> Path:
        """HTML data directory"""
        if self._html_dir is None:
            self._html_dir = PATHS['html_data']
        return self._html_dir
    
    def to_dict(self) -> Dict[str, Any]:
        """Return serializable config dict"""
        return {
            "model": self.model,
            "temperature": self.api_config.temperature,
            "max_tokens": self.api_config.max_tokens,
            "prompt_styles": self.prompt_styles,
            "output_dir": str(self.output_dir),
            "html_dir": str(self.html_dir),
        }
    
    @classmethod
    def from_groq_config_json(cls, config_path: Optional[Path] = None) -> "PipelineConfig":
        """
        Load pipeline configuration from Groq config JSON file.
        
        Args:
            config_path: Path to groq_config.json. If None, uses default from PATHS.
            
        Returns:
            PipelineConfig instance
        """
        if config_path is None:
            config_path = PATHS['groq_config']

        config_path = Path(config_path) 
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Extract API configuration
        # Handle both nested (model_info.name) and flat (model_name) config structures
        model_name = config_data.get('model_name')
        if not model_name and 'model_info' in config_data:
            model_name = config_data['model_info'].get('name', 'mixtral-8x7b-32768')
        if not model_name:
            model_name = 'mixtral-8x7b-32768'
        
        api_config = APIConfig(
            provider="groq",
            api_key=config_data.get('api_key') or os.getenv('GROQ_API_KEY'),
            model_name=model_name,
            max_retries=config_data.get('max_retries', 5),
            max_tokens=config_data.get('max_tokens', config_data.get('params', {}).get('max_output_tokens', 2048)),
            temperature=config_data.get('temperature', config_data.get('params', {}).get('temperature', 0.0)),
        )
        
        rate_limit_config = RateLimitConfig(
            inter_senator_delay=config_data.get('inter_senator_delay', 6.0),
            inter_style_delay=config_data.get('inter_style_delay', 3.0),
            web_scrape_delay=config_data.get('web_scrape_delay', 1.5),
            backoff_base=config_data.get('backoff_base', 2),
            backoff_max=config_data.get('backoff_max', 120),
        )
        
        ablation_config = AblationConfig(
            enabled=config_data.get('ablation_enabled', False),
            subset_size=config_data.get('ablation_subset_size', 25),
            random_seed=config_data.get('ablation_random_seed', 42),
        )
        
        return cls(
            api_config=api_config,
            rate_limit_config=rate_limit_config,
            ablation_config=ablation_config,
            prompt_styles=config_data.get('prompt_styles', ["direct", "pseudocode", "icl"]),
            run_all_styles=config_data.get('run_all_styles', True),
        )
    
    def get_active_prompt_styles(self) -> List[str]:
        """Return list of prompt styles to run"""
        return self.prompt_styles if self.run_all_styles else self.prompt_styles[:1]
