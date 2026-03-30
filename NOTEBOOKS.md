# Notebooks Guide

Documentation for Jupyter notebooks in this repository.

## Notebooks Overview

| Notebook | Purpose | Status | Audience |
|----------|---------|--------|----------|
| `senate_llm_pipeline_v4.ipynb` | Active data exploration pipeline | ✅ Active | Researchers |
| `senate_llm_pipeline.ipynb` | Legacy version (archived) | 🗂️ Archived | Reference only |
| `visualize_results.ipynb` | Result visualization | ✅ Available | Analysis |

## senate_llm_pipeline_v4.ipynb (Active)

**Purpose**: Comprehensive pipeline for scraping, parsing, and analyzing Senate member profiles.

### What It Does

1. **Data Preparation**
   - Loads Senate HTML files from `senate_html/`
   - Parses biographical information using regex and NLP
   - Extracts structured data (name, contact, party, etc.)

2. **Analysis**
   - Compares extraction methods:
     - Pure regex matching
     - spaCy NLP
     - LLM-based (Groq API)
   - Evaluates performance (coverage, accuracy)

3. **Visualization**
   - Creates comparison tables and figures
   - Generates HTML export for interactive inspection
   - Plots coverage metrics by extraction method

### Key Variables

- `HTML_DIR`: Path to Senate HTML files
- `OUTPUT_DIR`: Results and exports directory
- `SAMPLE_N`: Number of profiles to process
- `COMPARISON_MODEL`: LLM to use for extraction
- `PROMPT_STYLE`: Attack prompt type (direct, ICL, etc.)

### Running the Notebook

```python
# Standard notebook execution
jupyter notebook senate_llm_pipeline_v4.ipynb

# Or in VS Code
# File > Open senate_llm_pipeline_v4.ipynb
```

### Cell Structure

The notebook is organized into logical sections:

1. **Imports & Setup** (cells 1-3)
   - Import dependencies
   - Set configuration variables

2. **Regex Extraction** (cells 4-6)
   - Define regex patterns
   - Extract data using regex

3. **spaCy NLP** (cells 7-9)
   - Load spaCy model
   - Extract entities
   - Parse relationships

4. **LLM-Based Extraction** (cells 10-12)
   - Initialize Groq API client
   - Generate prompts
   - Query model
   - Parse responses

5. **Comparison & Evaluation** (cells 13-15)
   - Compare methods by coverage
   - Calculate accuracy metrics
   - Generate comparison tables

6. **Visualization** (cells 16-18)
   - Create comparison plots
   - Export HTML tables
   - Generate summary statistics

7. **Export Results** (cells 19-20)
   - Save results to CSV/JSON
   - Create summary report
   - Export detailed analysis

### Key Outputs

- **CSV**: `senate_results/comparison_results.csv`
- **JSON**: `senate_results/detailed_extraction.json`
- **HTML**: `senate_results/comparison.html` (interactive table)
- **Plots**: Various PNG/PDF visualizations in `senate_results/`

### Example Analysis

The notebook includes examples showing:

```python
# Extract email with different methods
regex_email = regex_pattern.search(html_content).group()
spacy_email = extract_with_spacy(html_content)['email']
llm_email = groq_client.extract_email(html_content)

# Compare on sample
comparison_df = pd.DataFrame({
    'Method': ['Regex', 'spaCy', 'LLM'],
    'Email Coverage': [0.95, 0.92, 0.98],
    'Email Accuracy': [0.87, 0.91, 0.96]
})
```

### Customization

Change these variables in early cells to customize:

```python
# Which model to use
COMPARISON_MODEL = 'llama-3.1-8b-instant'

# Number of profiles
SAMPLE_N = 50  # Default is all

# Attack prompt style
PROMPT_STYLE = 'icl'  # or 'direct', 'pseudocode'

# Which fields to extract
T1_FIELDS = ['name', 'email', 'phone']
T2_FIELDS = ['party', 'birth_year']
```

### Performance Tips

- Start with `SAMPLE_N=10` to test
- Use local regex/spaCy for quick iteration  
- Cache LLM results before visualization
- Comment out visualization cells if not needed

### Troubleshooting

**Cell fails with "Senate HTML not found"**
- Ensure `senate_html/` directory exists
- Check file permissions

**spaCy model not found**
```python
import spacy
spacy.cli.download("en_core_web_sm")
```

**Groq API errors**
- Verify API key in notebook
- Check internet connection
- Try `test_models.py` script first

**Out of memory**
- Reduce `SAMPLE_N`
- Process in batches
- Restart kernel

## visualize_results.ipynb (Optional)

**Purpose**: Create publication-ready figures from experimental results.

### Usage

```python
# Load results from multiple experiments
results = {
    'no_defense': load_results('result/groq_llama/synthetic_no_...'),
    'pi_ci': load_results('result/groq_llama/synthetic_pi_ci_...'),
}

# Generate comparison plots
plot_defense_effectiveness(results)
plot_extraction_by_field(results)
```

## senate_llm_pipeline.ipynb (Legacy - Archived)

**Status**: ⚠️ Legacy - Use v4 instead

This was the original version. Kept for reference only. v4 includes:
- Better organization
- More efficient processing
- Updated APIs
- Enhanced visualization

## Creating Your Own Notebook

### Template Structure

```python
# Cell 1: Imports & Setup
import numpy as np
import pandas as pd
from LLMPersonalInfoExtraction.utils import open_config
import LLMPersonalInfoExtraction as PIE

# Cell 2: Load Data
config = open_config('./configs/task_configs/synthetic.json')
task_manager, icl_manager = PIE.create_task(config)

# Cell 3: Initialize Models & Tools
model_config = open_config('./configs/model_configs/groq_config.json')
model = PIE.create_model(model_config)
evaluator = PIE.create_evaluator('groq', ['email', 'phone'])

# Cell 4: Run Analysis
results = []
for i in range(len(task_manager)):
    raw_html, label = task_manager[i]
    # Your analysis here
    results.append(...)

# Cell 5: Visualize & Export
df = pd.DataFrame(results)
print(df.describe())
df.to_csv('output.csv')
```

## Best Practices for Notebooks

1. **Clear Cell Organization**: One logical step per cell
2. **Comments**: Explain what each section does
3. **Reproducibility**: Set random seeds, use configs
4. **Variable Naming**: Use descriptive names
5. **Error Handling**: Add try/except for API calls
6. **Documentation**: Add markdown cells explaining logic
7. **Clean Output**: Clear figures and tables
8. **Version Control**: Keep notebooks in sync with code

## Running Notebooks Programmatically

```python
# Execute notebook from Python
import subprocess
subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', 
                '--execute', 'senate_llm_pipeline_v4.ipynb'])

# Or using papermill
import papermill as pm
pm.execute_notebook(
    'senate_llm_pipeline_v4.ipynb',
    'output.ipynb',
    parameters={'SAMPLE_N': 50}
)
```

## Exporting Results from Notebooks

Common patterns:

```python
# Save as CSV
results_df.to_csv('results.csv', index=False)

# Save as JSON
import json
with open('results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

# Save with NumPy
np.savez('results.npz', data=results_array, labels=labels)

# Export plot
import matplotlib.pyplot as plt
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
```

## Integration with Main Pipeline

Notebooks can call main pipeline components:

```python
# Import modules
import LLMPersonalInfoExtraction as PIE

# Use core components
model = PIE.create_model(model_config)
response = model.query("Extract email from profile...")

# Use evaluators
evaluator = PIE.create_evaluator('groq', info_cats)
metric = evaluator.update(response, label, 'email', defense)
```

This allows notebooks to leverage the structured codebase while enabling interactive analysis.
