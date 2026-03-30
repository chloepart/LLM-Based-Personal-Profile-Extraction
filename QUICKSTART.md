# Quick Start Guide

Get up and running in 5 minutes with a simple experiment.

## 1. Install (2 minutes)

```bash
# Clone repo
git clone <repo-url>
cd LLM-Based-Personal-Profile-Extraction

# Create environment using conda
conda env create -f PIE_environment.yml
conda activate PIE
```

**Note**: If you get dependency errors, update problematic packages:
```bash
pip install --upgrade openai google-generativeai anthropic
```

## 2. Set Up API Key (1 minute)

### For Groq (Recommended - Free)

```bash
# Open config
vim configs/model_configs/groq_config.json

# Add your API key from https://console.groq.com
```

Edit the `"api_keys"` array:
```json
{
  "api_key_info": {
    "api_keys": ["your-api-key-here"]
  }
}
```

### For OpenAI

```bash
vim configs/model_configs/gpt_config.json
```

```json
{
  "api_key_info": {
    "api_keys": ["sk-...your-key..."]
  }
}
```

### For Google Gemini

```bash
vim configs/model_configs/gemini_config.json
```

```json
{
  "api_key_info": {
    "api_keys": ["your-gemini-key"]
  }
}
```

## 3. Run First Experiment (2 minutes)

### Option A: Using Groq (Fastest, Free)

```bash
python main.py \
    --model_config_path configs/model_configs/groq_config.json \
    --model_name llama-3.1-8b-instant \
    --task_config_path configs/task_configs/synthetic.json \
    --defense no \
    --prompt_type direct
```

### Option B: Using OpenAI

```bash
python main.py \
    --model_config_path configs/model_configs/gpt_config.json \
    --model_name gpt-3.5-turbo \
    --task_config_path configs/task_configs/synthetic.json \
    --defense no \
    --prompt_type direct
```

### What This Does

- Loads 100 synthetic HTML profiles
- Queries model to extract **email addresses**
- Compares extracted vs ground truth
- Prints metrics (Accuracy, ROUGE-1)
- Saves detailed results to `result/`

### Expected Output

```text
Testing synthetic with no defense
Email Accuracy: 0.85
Phone Accuracy: 0.62
Name ROUGE-1: 0.78
...
```

## 4. View Results

Results are saved to:
```
result/groq_llama-3.1-8b-instant/synthetic_no_direct_0_adaptive_no/
├── all_raw_responses.npz
└── all_raw_responses_with_parsed.npz
```

Quick summary:
```bash
python scripts/score.py
```

## Common Next Steps

### Run with Defense

```bash
python main.py \
    --model_config_path configs/model_configs/groq_config.json \
    --model_name llama-3.1-8b-instant \
    --task_config_path configs/task_configs/synthetic.json \
    --defense pi_ci \
    --prompt_type direct
```

### Run Multiple Experiments

Edit `run.py`:
```python
model_info = ['groq', 'llama-3.1-8b-instant']
datasets = ['synthetic']
defenses = ['no', 'replace_at', 'pi_ci']
prompt_types = ['direct']
```

Run:
```bash
python run.py
```

### Evaluate with BERT-Score

```bash
python evaluate.py \
    --provider groq \
    --model_name llama-3.1-8b-instant \
    --dataset synthetic \
    --defense pi_ci \
    --m2 bert-score
```

## Troubleshooting

### Error: "API key not found"

```
Error: Please enter a valid API key to use
```

**Fix**: Check your config file has correct API key at index specified

### Error: "Module not found"

```
ModuleNotFoundError: No module named 'groq'
```

**Fix**: Verify you activated the conda environment:
```bash
conda activate PIE
```

### Error: "Rate limit exceeded"

```
Error: Rate limit exceeded
```

**Fix**: Wait a moment and try again. Use `--api_key_pos 1` to switch API keys if you have multiple.

### Model returns empty response

```
Warning: Empty response from model
```

**Fix**: 
1. Verify model name is correct for provider
2. Check internet connection
3. Try a different model

### "No such file or directory" for data/

```
FileNotFoundError: data/synthetic/profile1.html
```

**Fix**: Download datasets from [Zenodo](https://zenodo.org/records/14737200) and extract:
```bash
unzip zenodo_data.zip -d data/
```

## Parameter Reference

### Main Arguments

| Argument | Default | Options |
|----------|---------|---------|
| `--model_config_path` | palm2_config.json | Path to model config |
| `--model_name` | (required) | Model identifier |
| `--task_config_path` | synthetic.json | Path to task config |
| `--defense` | no | See defenses below |
| `--prompt_type` | direct | direct, icl, pseudocode |
| `--icl_num` | 0 | 0-10 (number of examples) |
| `--adaptive_attack` | no | See adaptive attacks below |

### Defense Types

```
no                  # No defense
replace_at          # Replace @ with AT
replace_dot         # Replace . with DOT
replace_at_dot      # Both replacements
hyperlink           # Convert to HTML links
mask                # Replace with [REDACTED]
pi_ci               # Context-intelligent defense
pi_id               # Identity-aware defense
pi_ci_id            # Combined defense
image               # Convert to image (Gemini/GPT-4V only)
```

### Adaptive Attacks

```
no                  # No adaptive attack
sandwich            # Wrap instructions with benign text
xml                 # Use XML tags for structure
delimiters          # Custom delimiters
random_seq          # Random character injection
instructional       # Contradictory instructions
paraphrasing        # Rephrase the question
retokenization      # Break tokens strategically
```

## Next: Advanced Usage

See [USAGE.md](USAGE.md) for:
- Custom datasets
- Batch experiment configuration
- Interpreting results in detail
- Adding new models/defenses

## Getting Help

1. Check [ARCHITECTURE.md](ARCHITECTURE.md) for design details
2. Review notebooks in `senate_llm_pipeline_v4.ipynb` for examples
3. Run utility scripts for diagnostics:
   ```bash
   python scripts/test_models.py    # Check API connectivity
   python scripts/check_defense.py  # Debug defense mechanisms
   ```
4. Open an issue with error output

## Key Files to Know

- **main.py** - Core experiment runner
- **run.py** - Batch orchestration (edit to configure)
- **evaluate.py** - Additional metrics (BERT-Score)
- **configs/model_configs/** - Model settings
- **configs/task_configs/** - Dataset settings
- **LLMPersonalInfoExtraction/** - Core package
