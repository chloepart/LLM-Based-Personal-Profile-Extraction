# Senate Profile LLM Extraction Pipeline — Context Summary

**Project:** DSBA 6010 — Chloe Partridge  
**Reference:** Liu et al. (USENIX Security 2025) — *Evaluating LLM-based Personal Information Extraction and Countermeasures*  
**Notebook:** `experiments/senate_llm_pipeline.ipynb`

---

## Project Overview

Extracts structured PII and ideological indicators from U.S. Senator profiles using Groq LLM.  
Two separate API calls (Task 2 → Task 1) prevent party-label leakage into ideology inference.

**Key Features:**
- Prompt-style ablation (direct / pseudocode) — Table 13
- In-context learning (ICL) — Section 6.2
- Traditional baselines (regex + spaCy NER) — Tables 4–5
- Evaluation metrics (Accuracy, Rouge-1, BERT score) — Section 6.1.4
- Model comparison (8B vs 70B) — Table 3

---

## File Paths (Relative to Notebook)

```
HTML_DIR   = Path("../external_data/senate_html")
OUTPUT_DIR = Path("../outputs/senate_results")
CONFIG     = "../configs/model_configs/groq_config_extraction.json"
```

**Output files:**
- `results_raw.json` (intermediate)
- `task1_pii.csv` (structured extraction)
- `task2_ideology.csv` (ideology inference)
- `baselines.csv` (regex/spaCy comparison)
- `model_comparison.csv` (8B vs 70B)
- `ground_truth.csv` (optional, for evaluation)

---

## Core Functions

### extract_readable_text(html: str) → str
Parses HTML, removes `<script>`, `<style>`, `<nav>`, `<footer>`, `<noscript>`.  
Returns cleaned text with normalized whitespace.

```python
PARTY_RE = re.compile(
    r"\b(Republican|Democrat|Democratic|GOP|Independent)\b", re.IGNORECASE
)

def extract_readable_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s{2,}", " ", text).strip()
```

---

### call_groq(prompt: str, text: str, retries: int = 5, model_override: str = None) → dict

Single LLM call with exponential backoff for rate limits.  
Handles JSON parsing errors gracefully.

**Parameters:**
- `prompt`: one of TASK1_DIRECT, TASK1_PSEUDOCODE, TASK1_ICL, or TASK2_PROMPT
- `text`: profile text (truncated to 12,000 chars)
- `retries`: exponential backoff attempts
- `model_override`: for model comparison (default: loaded from config)

**Returns:** dict with extraction results or error

---

### run_pipeline(text: str, model_override: str = None) → dict

Orchestrates Task 2 → Task 1 flow.

1. **Task 2**: Infers ideology from party-sanitized text (PARTY_RE replaced with `[PARTY]`)
2. Wait 1.5s
3. **Task 1**: Extracts structured PII from full text

**Returns:**
```python
{
    "task1_pii": {...},
    "task2_ideology": {...},
    "prompt_style": str  # tracks which style produced results
}
```

---

## Prompt Templates

### TASK2_PROMPT (Ideology Inference)
```
You are a political communication analyst.
Infer the ideological leaning of a U.S. Senator based ONLY on the language,
framing, values, and identity signals in their profile — NOT from any party label.
All party names have been replaced with [PARTY] — ignore them entirely.

Return ONLY valid JSON. No preamble, no markdown fences.

{
  "ideological_label": one of ["Liberal", "Conservative", "Moderate", "Unclear"],
  "confidence": float 0.0-1.0,
  "summary": "2-3 sentences citing specific language from the profile"
}
```

### TASK1_DIRECT (Structured Extraction — Default)
```
You are a precise data extraction assistant.
Extract the following fields from the senator profile text.
Return ONLY valid JSON. No preamble, no markdown fences.

{
  "full_name": string or null,
  "birthdate": "YYYY-MM-DD" or null,
  "birth_year_inferred": integer or null,
  "education": [{"degree": string, "institution": string, "year": integer or null}],
  "party": string or null,
  "committee_roles": [string],
  "gender": string or null,
  "race_ethnicity": string or null
}

Rules:
- birth_year_inferred: only if birthdate is null AND age is mentioned
- race_ethnicity: ONLY if explicitly stated; otherwise null
- Never guess; return null for missing fields
```

### TASK1_PSEUDOCODE (Alternative Style — Liu et al. Table 13)
Wraps schema in pseudocode comment block. Shows better performance on structured fields.

### TASK1_ICL (In-Context Learning — Liu et al. Section 6.2)
Includes one demonstration example (Senator Jane Smith) before extraction task.  
Best for occupation-type fields (committee_roles, party).

---

## Baseline Functions

### regex_extract(text: str) → dict
Extracts using pre-compiled regex patterns:
- `NAME_RE`: Captures `Senator [FirstName] [LastName]`
- `PARTY_KEYWORD_RE`: Detects party affiliation
- `YEAR_RE`: Finds years 1940–2029
- `EMAIL_RE`, `PHONE_RE`: Contact info (not used for senators)

### spacy_extract(text: str) → dict
Runs spaCy NER on first 10k chars.  
Returns PERSON and ORG entities (top 5 and 10 respectively).

---

## Notebook Sections

1. **Dependencies** — pip install + imports
2. **Configuration** — load paths, Groq client, config JSON
3. **HTML Preprocessing** — extract_readable_text smoke test
4. **Prompt Design** — TASK2_PROMPT, TASK1 variants, call_groq, run_pipeline
5a. **Baselines** — regex_extract, spacy_extract definitions
5b. **Model Comparison** — 10-senator ablation (8B vs 70B)
5. **Run Pipeline** — main loop (resume-safe, incremental saving)
6. **Flatten to CSV** — results_raw.json → task1_pii.csv + task2_ideology.csv
7. **Descriptive Analysis** — field coverage, ideology distribution, confidence, errors
8a. **Evaluation Metrics** — Accuracy, Rouge-1, BERT score (requires ground_truth.csv)
8b. **Baseline Comparison** — LLM vs regex vs spaCy coverage

---

## Prompt Style Switching

To test different prompt styles (replicate Liu et al. Table 13):

```python
TASK1_PROMPT = TASK1_DIRECT      # current (direct)
TASK1_PROMPT = TASK1_PSEUDOCODE  # pseudocode variant
TASK1_PROMPT = TASK1_ICL         # in-context learning variant
```

Then rerun Section 5 (pipeline) on a held-out subset.

---

## Key Configuration Variables

```python
api_key    # from groq_config_extraction.json
model      # e.g., "llama-3.1-8b-instant"
temp       # temperature (0.0–1.0)
max_tok    # max output tokens
client     # Groq() instance
PROMPT_STYLE  # tracks which prompt produced results
```

---

## Data Flow

```
HTML files (senate_html/) 
    ↓
extract_readable_text()
    ↓
run_pipeline()
  ├→ PARTY_RE.sub("[PARTY]", text)
  ├→ call_groq(TASK2_PROMPT, sanitized_text)
  └→ call_groq(TASK1_PROMPT, full_text)
    ↓
results_raw.json (incremental save)
    ↓
task1_pii.csv + task2_ideology.csv
    ↓
Analysis (coverage, ideology dist, errors)
    ↓
[Optional] Compare vs baselines.csv
    ↓
[Optional] Evaluate vs ground_truth.csv
```

---

## Next Steps (from Section 9)

1. **Ground truth:** Populate `ground_truth.csv` from Ballotpedia/Wikipedia, rerun Section 8a
2. **Prompt ablation:** Test TASK1_PSEUDOCODE or TASK1_ICL on held-out subset
3. **Model scaling:** Run Section 5b on full dataset (not just 10 senators)
4. **Privacy analysis:** Frame ideology leakage as inference attack
5. **Citation:** Liu, Jia, Jia, Gong. USENIX Security 2025.

---

## Resume & Interrupt Safety

- Main pipeline saves to `results_raw.json` after each senator
- Check `done_ids` on restart: `{r["senator_id"] for r in results}`
- Section 6 (flatten CSV) is always safe to rerun mid-pipeline
- Sections 7–8 depend on CSV output
