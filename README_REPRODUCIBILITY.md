# Senate LLM Extraction Pipeline — Reproducibility Guide

**Document:** Setup, execution, and result caching for `senate_llm_pipeline_V3.ipynb`  
**Last Updated:** Phase 5 Streamlining Complete  
**Target Audience:** Students, researchers replicating Liu et al. (USENIX Security 2025)

## Quick Start

### 1. Environment Setup

```bash
# Create Python 3.9+ environment
python3 -m venv pie_env
source pie_env/bin/activate

# Install dependencies
pip install -r requirements.txt
# OR manually:
pip install groq pandas tqdm rouge-score bert-score spacy beautifulsoup4

# Download spaCy English model
python -m spacy download en_core_web_sm
```

### 2. Configure API Keys

The pipeline uses **Groq API** for LLM extraction.

**Option A: Environment Variable (Recommended)**
```bash
export GROQ_API_KEY="gsk_..."  # Your Groq API key
jupyter notebook experiments/senate_llm_pipeline_V3.ipynb
```

**Option B: Config File**
Edit `configs/model_configs/groq_config_extraction.json`:
```json
{
  "api_key_info": {
    "api_keys": ["gsk_..."]
  },
  "model_info": {
    "name": "mixtral-8b-7b-instruct-v0.1"
  }
}
```

### 3. Run the Notebook

```bash
cd experiments/
jupyter notebook senate_llm_pipeline_V3.ipynb
```

Execute cells in order. Key sections:
- **Cell 5:** Session initialization (loads config, API client, HTML files)
- **Cell 7:** LLM extraction (main pipeline with 3 prompt styles)
- **Cell 12:** Baseline methods (regex, spaCy NER, keyword search)
- **Cell 15:** Evaluation with result caching

---

## Runtime Expectations

### First Run (Full Computation)

| Phase | Runtime | Notes |
|-------|---------|-------|
| **Session Init** | <1s | Loads config, spaCy model, HTML files |
| **LLM Extraction** | 3-8 min | API calls + rate limiting (4-6s per senator × 3 styles) |
| **Baselines** | 30-90s | Local ML (regex, spaCy, keyword) — no API calls |
| **Ground Truth Load** | <1s | Loads pre-scraped Wikipedia/Ballotpedia data |
| **Evaluation** | 60-90s | Computes metrics + caches to JSON |
| **TOTAL** | **5-10 min** | Depends on HTML file count + API latency |

### Subsequent Runs (With Caching)

| Phase | Runtime | Notes |
|-------|---------|-------|
| **Session Init** | <1s | Same |
| **LLM Extraction** | 0s | Skipped if `results_raw.json` exists (resume=True) |
| **Baselines** | 30-90s | Always rerun (can be skipped manually) |
| **Ground Truth** | <1s | Same |
| **Evaluation** | <1s | **Loads cached JSON** from `evaluation_results_cache.json` |
| **TOTAL** | **~30-40s** | Dramatic speedup for iteration! |

---

## Result Caching & File Structure

### What Gets Cached

The pipeline automatically caches these files to `outputs/senate_results/`:

```
outputs/senate_results/
├── results_raw.json                   # Raw LLM extraction output (all senators, all styles)
├── task1_pii.csv                      # Flattened CSV: one row per (senator, prompt_style)
├── baselines.csv                      # Baseline method results (regex, spaCy, keyword)
├── evaluation_results_cache.json      # ✅ CACHED metrics (no recomputation!)
└── ground_truth_errors.log            # Errors from GT scraping phase (optional)
```

### How Caching Works

**LLM Extraction** (`run_main_pipeline`):
- Checks `results_raw.json` for senators already processed
- Skips completed senators, resumes from checkpoint
- Use `resume=True` (default) to speed up interrupted runs

**Evaluation** (`evaluate_all_styles`):
- Checks `evaluation_results_cache.json` for cached metrics
- If found + `overwrite=False`, loads cache instantly (<1s)
- If `overwrite=True`, recomputes metrics and updates cache
- Supports explicit cache clearing by deleting the JSON file

### Clearing Caches

To force recomputation, delete cache files:

```bash
# Recompute all extractions (but keep previous results)
rm outputs/senate_results/results_raw.json

# Recompute evaluation metrics only
rm outputs/senate_results/evaluation_results_cache.json

# Start completely fresh
rm -rf outputs/senate_results/
```

---

## Resuming Interrupted Runs

The pipeline is **resume-safe** — if execution stops midway, you can restart without losing progress.

### Resume LLM Extraction

```python
# In Cell 7, this is the default:
results = run_main_pipeline(
    html_files=session['html_files'],
    session_config=session['session_config'],
    resume=True  # ← Skip already-processed senators
)
```

**How it works:**
- Checks `results_raw.json` for completed `senator_id` values
- Only processes senators NOT in the file
- Appends new results incrementally to JSON

**To restart from scratch:**
```python
results = run_main_pipeline(
    ...,
    resume=False  # Force recomputation of all senators
)
```

### Resume Baseline Extraction

Baselines are fast (30-90s) so resumption is rarely needed. To skip:

```python
# Manually skip baseline cell, or comment it out:
# baseline_metrics = run_baselines(...)
```

---

## Model Configuration

### Editing the Model

Edit `configs/model_configs/groq_config_extraction.json`:

```json
{
  "model_info": {
    "name": "mixtral-8b-7b-instruct-v0.1",
    "max_output_tokens": 2048,
    "temperature": 0.7
  },
  "params": {
    "temperature": 0.7,
    "max_output_tokens": 2048
  }
}
```

**Supported Groq models:**
- `mixtral-8b-7b-instruct-v0.1` (slower, cheaper, default)
- `mixtral-8b-instruct-v0.9` (alternative 8B model)
- Other Groq-supported models via API

**Note:** Changes require restarting the Jupyter kernel.

---

## Reproducibility & Fixed Seeds

### Fixed Random Seed (Ablation Subset)

The ablation study selects a fixed subset of **25 senators** with `seed=42`:

```python
# In session initialization (Cell 5)
session = initialize_pipeline_session(
    ...,
    ablation_subset_size=25  # Fixed seed ensures same senators each run
)
```

This ensures **reproducible subset selection** — same 25 senators tested across all runs.

**To change ablation size:**
```python
session = initialize_pipeline_session(
    ...,
    ablation_subset_size=50  # Now 50 senators instead of 25
)
```

---

## Output Interpretation

### Prediction CSV (`task1_pii.csv`)

Columns:
```
senator_id          |  full_name  | birthdate | gender | education | prompt_style
================================================================================
John_Smith_CA       |  John Smith | 1965-05-12| Male   | BA, Cornell| direct
John_Smith_CA       |  John Smith | 1965-05-12| M      | Cornell    | pseudocode
John_Smith_CA       |  John Smith | 1965-05-12| Male   | B.A. 1986  | icl
```

**Interpretation:**
- Same senator may have slightly different extractions across prompt styles
- `gender` column shows variations (Male / M / male → normalize before evaluation)
- `education` format varies (prompts have different output conventions)
- `extraction_error` column shows API errors (if not null → extraction failed)

### Evaluation Results (`evaluation_results_cache.json`)

```json
{
  "timestamp": "2024-01-15 10:23:45",
  "by_style": {
    "direct": {
      "style": "direct",
      "record_count": 95,
      "metrics": {
        "birthdate_exact": 0.847,
        "gender_exact": 0.921,
        "religion_hierarchical": 0.654,
        "education_rouge1": 0.412
      }
    },
    "pseudocode": { ... },
    "icl": { ... }
  }
}
```

**Key Metrics:**
- `birthdate_exact`: Exact match on full date (YYYY-MM-DD)
- `gender_exact`: Case-insensitive match (M/Male/male all match)
- `religion_hierarchical`: Partial credit for same parent category (Methodist → Protestant)
- `education_rouge1`: ROUGE-1 overlap with ground truth

---

## Data Validation Checklist

Before running evaluation, verify:

- [ ] `external_data/senate_html/` contains ≥50 .html files
- [ ] `external_data/ground_truth/senate_ground_truth.csv` exists (pre-scraped)
- [ ] `configs/model_configs/groq_config_extraction.json` has valid model name + token limits
- [ ] `GROQ_API_KEY` environment variable is set or config file has API key
- [ ] `pie_env/` virtualenv is activated (check `which python`)
- [ ] spaCy model installed: `python -c "import spacy; spacy.load('en_core_web_sm')"`

If any validation fails, the notebook will error early with a clear message.

---

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'modules'`

**Solution:** Make sure you're running the notebook from the `experiments/` directory:
```bash
cd experiments/
jupyter notebook senate_llm_pipeline_V3.ipynb
```

The notebook uses `sys.path.insert(0, '..')` to load modules from parent directory.

### Error: `ValueError: API key required for groq`

**Solution:** Set environment variable before running:
```bash
export GROQ_API_KEY="gsk_..."
jupyter notebook ...
```

Or add API key to `configs/model_configs/groq_config_extraction.json`.

### Error: `FileNotFoundError: ../external_data/senate_html/`

**Solution:** HTML files must be in correct location. Verify:
```bash
ls ../external_data/senate_html/ | head -5  # Should show .html files
```

### Extraction timing out / Rate limit errors

The pipeline automatically retries with exponential backoff. If you still see errors:
1. Increase `inter_senator_delay` in Cell 7 (default is 4-6s)
2. Reduce `max_tokens` in groq_config_extraction.json
3. Check Groq API quota: https://console.groq.com/

### Evaluation returns NaN values

This usually means:
- Ground truth CSV is missing required  columns
- Predictions CSV is empty or malformed
- Merge (senator_id match) found no overlaps

**Debug:** Check cell outputs:
```python
print(f"GT shape: {df_gt.shape}, Pred shape: {df_pred.shape}")
print(f"Merged count: {merged.shape[0]}")  # Should be > 0
```

---

## Extending the Pipeline

### Add New Prompt Style

1. Edit `configs/model_configs/groq_config_extraction.json` or `modules/config_unified.py`
2. Add new prompt template to `ABLATION_STYLES` dictionary
3. Update Cell 5 `prompt_styles` list:
   ```python
   session = initialize_pipeline_session(
       ...,
       prompt_styles=["direct", "pseudocode", "icl", "your_new_style"]
   )
   ```
4. Re-run extraction (will process new style on all senators)

### Add New Evaluation Metric

1. Implement metric function with signature: `metric(gt_value, pred_value) → float`
2. Add to `modules/evaluation_suite.py` → `_evaluate_style()` function
3. Re-run evaluation with `overwrite=True` to compute new metric

### Change Default Configuration

Edit `modules/config_unified.py` to change default paths, models, or parameters. Changes apply globally without modifying notebook cells.

---

## Citation & Attribution

**Paper Referenced:**
> Liu et al. (2025). *Evaluating LLM-based Personal Information Extraction and Countermeasures*. USENIX Security Symposium.

**Replication Details:**
- Model: Groq Mixtral-8B instruct
- Ablation subset: 25 senators (seed=42) for reproducibility
- Evaluation metrics: Accuracy, ROUGE-1, BERT score, hierarchical matching
- Ground truth sources: Wikipedia, Ballotpedia, Pew religion data

**Usage:** This notebook is provided for **educational purposes** to understand LLM-based PII extraction. Users should be aware of privacy and ethical implications when working with personal information.

---

## Contact & Support

For issues or questions:
1. Check this document's troubleshooting section
2. Review notebook inline comments (cells have detailed docstrings)
3. Check module docstrings: `python -c "from modules import initialize_pipeline_session; help(initialize_pipeline_session)"`
4. Consult `docs/` folder for architecture and design decisions

---

**Document Version:** 1.0  
**Notebook Version:** V3 (Streamlined)  
**Last Updated:** 2024  
✅ All phases complete
