# Comprehensive Usage Guide

Detailed examples and configurations for running experiments.

## Table of Contents

1. [Configuration Overview](#configuration-overview)
2. [Running Experiments](#running-experiments)
3. [Batch Operations](#batch-operations)
4. [Interpreting Results](#interpreting-results)
5. [Advanced Features](#advanced-features)
6. [Customization](#customization)
7. [Troubleshooting](#troubleshooting)

## Configuration Overview

Two types of configs control behavior:

### 1. Model Configuration

Located in `configs/model_configs/{provider}_config.json`

**Example: Groq**

```json
{
  "model_info": {
    "provider": "groq",
    "name": "llama-3.1-8b-instant",
    "type": "api"
  },
  "api_key_info": {
    "api_keys": ["gsk_...", "gsk_..."],
    "api_key_use": 0
  },
  "params": {
    "temperature": 0.7,
    "max_output_tokens": 500,
    "seed": 42,
    "gpus": []
  }
}
```

**Example: Local LLaMA**

```json
{
  "model_info": {
    "provider": "llama",
    "name": "meta-llama/Llama-2-7b-hf",
    "type": "local"
  },
  "params": {
    "temperature": 0.7,
    "max_output_tokens": 500,
    "device": "cuda",
    "gpus": ["0"]
  }
}
```

### 2. Task Configuration

Located in `configs/task_configs/{dataset}.json`

**Example: Synthetic**

```json
{
  "task_info": {
    "task": "personal_information_extraction",
    "type": "single_turn"
  },
  "dataset_info": {
    "dataset": "synthetic",
    "path": "./data/synthetic",
    "label_path": "./data/synthetic/labels.json",
    "icl_path": "./data/icl",
    "icl_label_path": "./data/icl/labels.json"
  }
}
```

## Running Experiments

### Single Experiment with All Options

```bash
python main.py \
    --model_config_path configs/model_configs/groq_config.json \
    --model_name llama-3.1-8b-instant \
    --task_config_path configs/task_configs/synthetic.json \
    --api_key_pos 0 \
    --defense pi_ci \
    --prompt_type icl \
    --icl_num 5 \
    --adaptive_attack sandwich \
    --verbose 1 \
    --redundant_info_filtering True
```

### Parameter Definitions

| Parameter | Values | Meaning |
|-----------|--------|---------|
| `--model_config_path` | Path | Location of model config JSON |
| `--model_name` | string | Model identifier (overrides config) |
| `--task_config_path` | Path | Location of task config JSON |
| `--api_key_pos` | 0,1,2,... | Which API key to use (if multiple) |
| `--defense` | string | Defense mechanism (see QUICKSTART.md) |
| `--prompt_type` | direct/icl/pseudocode | Attack prompt style |
| `--icl_num` | 0-10 | Number of in-context examples |
| `--adaptive_attack` | string | Adaptive attack technique |
| `--gpus` | 0,1,2 | GPU IDs for local models (comma-separated) |
| `--verbose` | 0/1 | Print detailed progress |
| `--redundant_info_filtering` | True/False | Filter out irrelevant HTML |

## Batch Operations

### Scenario 1: Test All Defenses on One Model

Edit `run.py`:

```python
model_info = ['groq', 'llama-3.1-8b-instant']
datasets = ['synthetic']
prompt_types = ['direct']
icl_nums = [0]
defenses = [
    'no', 
    'replace_at', 
    'replace_dot', 
    'hyperlink', 
    'mask',
    'pi_ci', 
    'pi_id', 
    'pi_ci_id'
]
adaptive_attacks_on_pi = ['no']
```

Run:
```bash
python run.py
```

This will run 8 experiments (one per defense), taking ~20+ minutes depending on model and dataset size.

### Scenario 2: Compare Multiple Models

```python
# Would require modifying run.py to add model_info as list
# For now, run separately:
```

```bash
# GPT-4 on synthetic
python main.py \
    --model_config_path configs/model_configs/gpt_config.json \
    --model_name gpt-4 \
    --task_config_path configs/task_configs/synthetic.json \
    --defense no

# Gemini on synthetic
python main.py \
    --model_config_path configs/model_configs/gemini_config.json \
    --model_name gemini-pro \
    --task_config_path configs/task_configs/synthetic.json \
    --defense no

# LLaMA 2 local on synthetic
python main.py \
    --model_config_path configs/model_configs/llama_config.json \
    --model_name meta-llama/Llama-2-7b-hf \
    --gpus 0 \
    --task_config_path configs/task_configs/synthetic.json \
    --defense no
```

### Scenario 3: Test Adaptive Attacks

```python
# Modify run.py
adaptive_attacks_on_pi = [
    'no',
    'sandwich',
    'xml',
    'delimiters',
    'random_seq',
    'instructional',
    'paraphrasing',
    'retokenization'
]

# Run one combination (full combinations will take long time!)
```

### Scenario 4: In-Context Learning Comparison

```bash
# No few-shot
python main.py --icl_num 0 --prompt_type direct ...

# 3-shot
python main.py --icl_num 3 --prompt_type icl ...

# 5-shot  
python main.py --icl_num 5 --prompt_type icl ...
```

## Interpreting Results

### Result Directory Structure

After running experiments:

```
result/
├── groq_llama-3.1-8b-instant/
│   ├── synthetic_no_direct_0_adaptive_no/
│   │   ├── all_raw_responses.npz
│   │   ├── all_raw_responses_with_parsed.npz
│   │   └── accuracy_rouge.txt
│   ├── synthetic_replace_at_direct_0_adaptive_no/
│   │   └── ...
│   └── ...
├── gpt_gpt-4/
│   └── ...
```

### Log Files

In `log/{provider}_{model}/`:

```
synthetic_no_direct_0_adaptive_no_filter_True.txt
synthetic_pi_ci_direct_0_adaptive_no_filter_True.txt
```

**Log Format:**

```
Testing synthetic with no defense
Processing profile 1/100: profile_name
  Email Accuracy: 1.0 (PREV: 1.0)
  Phone Accuracy: 0.5 (PREV: 0.5)
  Name ROUGE-1: 0.92 (PREV: 0.92)
  ...

Final Results:
  Email: 0.85
  Phone: 0.62
  Name: 0.78
  Birthday: 0.45
  ...
```

### Loading and Analyzing Results

```python
import numpy as np
from LLMPersonalInfoExtraction.utils import open_json

# Load raw responses
data = np.load('result/groq_llama-3.1-8b-instant/synthetic_no_direct_0_adaptive_no/all_raw_responses.npz', 
               allow_pickle=True)

res = data['res'].item()        # Model responses by info_type
label = data['label'].item()    # Ground truth labels by info_type

# Analyze email extraction
emails_pred = res['email']
emails_true = label['email']

correct = sum(1 for p, t in zip(emails_pred, emails_true) 
              if p.strip().lower() == t.strip().lower())
accuracy = correct / len(emails_pred)
print(f"Email accuracy: {accuracy:.2%}")

# Analyze impact of defense
defense_no = np.load('result/.../synthetic_no_direct_0.../all_raw_responses.npz', allow_pickle=True)
defense_yes = np.load('result/.../synthetic_pi_ci_direct_0.../all_raw_responses.npz', allow_pickle=True)

drop = (defense_no['res'].item()['email'].shape[0] - 
        sum(1 for p in defense_yes['res'].item()['email'] if p))
print(f"Defense prevented {drop} correct extractions")
```

## Advanced Features

### In-Context Learning (Few-Shot)

ICL examples are auto-loaded from `data/icl/`.

```bash
# Use 3 examples
python main.py --prompt_type icl --icl_num 3 ...

# Use 5 examples (better performance, longer prompts)
python main.py --prompt_type icl --icl_num 5 ...
```

The attacker automatically:
1. Samples `icl_num` examples from `data/icl/`
2. Parses example profiles
3. Prepends examples to attack prompt

### Adaptive Attacks

Test adversarial prompt techniques:

```bash
# Sandwich attack (wrap instructions with benign text)
python main.py --adaptive_attack sandwich ...

# XML injection
python main.py --adaptive_attack xml ...

# Delimiter manipulation
python main.py --adaptive_attack delimiters ...
```

These modify how the prompt is constructed without changing the underlying task.

### Multiple API Keys

If you have multiple API keys (for rate limit distribution):

```json
{
  "api_key_info": {
    "api_keys": [
      "key1", 
      "key2",
      "key3"
    ],
    "api_key_use": 0
  }
}
```

Switch between them:

```bash
python main.py --api_key_pos 0 ...  # Use key1
python main.py --api_key_pos 1 ...  # Use key2
python main.py --api_key_pos 2 ...  # Use key3
```

### Image-Based Defense

For models supporting vision:

```bash
python main.py --defense image ...
```

This:
1. Renders HTML to image
2. Sends image instead of text
3. Only works with GPT-4V or Gemini Vision

Note: Requires pre-generated images in `data/synthetic_images/`.

## Customization

### Add Your Own Dataset

1. **Prepare data directory:**

```bash
mkdir -p data/mydataset
```

2. **Create HTML profiles:**

```html
<!-- data/mydataset/profile1.html -->
<html>
<body>
  <h1>John Smith</h1>
  <p>Email: john@example.com</p>
  <p>Phone: (555) 123-4567</p>
  <p>Born: 1990</p>
</body>
</html>
```

3. **Create labels.json:**

```json
{
  "profile1": {
    "name": "John Smith",
    "email": "john@example.com",
    "phone": "(555) 123-4567",
    "birth_year": "1990"
  }
}
```

4. **Create config:**

```bash
cp configs/task_configs/synthetic.json configs/task_configs/mydataset.json
```

Edit `configs/task_configs/mydataset.json`:

```json
{
  "dataset_info": {
    "dataset": "mydataset",
    "path": "./data/mydataset",
    "label_path": "./data/mydataset/labels.json",
    ...
  }
}
```

5. **Run:**

```bash
python main.py \
    --task_config_path configs/task_configs/mydataset.json \
    --model_name ... \
    ...
```

### Modify Defense Behavior

Defenses apply during preprocessing. To customize:

1. Edit `LLMPersonalInfoExtraction/defense/{Defense}.py`
2. Modify the `apply()` method
3. Test with `scripts/check_defense.py`

Example - Custom `pi_ci` variant:

```python
class PICustom(Defense):
    def apply(self, profile: str, label: dict) -> str:
        # Your custom logic here
        modified = profile.replace(label['email'], 'EMAIL_HIDDEN')
        modified = modified.replace(label['phone'], 'PHONE_HIDDEN')
        return modified
```

### Add New Prompt Type

1. Add template to `data/system_prompts/`
2. Modify `load_instruction()` in `LLMPersonalInfoExtraction/utils/`
3. Update attacker to use new prompt type

## Troubleshooting

### "Remote end closed connection"

```
ConnectionError: Remote end closed connection without response
```

**Causes**:
- API key expired or revoked
- Server timeout
- Network issue

**Solutions**:
1. Check API key is still valid
2. Try `--api_key_pos` with backup key
3. Reduce `--icl_num` to shorten prompt
4. Wait a moment and retry

### Very Low/Zero Accuracy

- Verify `labels.json` ground truth is correct
- Check model actually processes the prompt (increase `--verbose 1`)
- Try without defense to see baseline performance
- Verify model is correct for dataset format

### Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions**:
- Reduce `batch` size if implemented
- Use smaller model
- Use API-based model instead of local

### Results Not Saved

Check:
1. `result/` directory exists
2. Disk space available
3. File permissions: `chmod 755 result/`

## Performance Tips

### Fastest Setup
- Use **Groq** (free, fast)
- No ICL (`--icl_num 0`)
- Simple defense (`--defense no`)
- Direct prompt type

### Most Accurate Setup
- Higher-end model (GPT-4)
- ICL with 5 examples (`--icl_num 5`)
- Smart defense (`--defense pi_ci_id`)

### Cost-Conscious Setup
- Use Groq or free tier models
- Run on small subset first
- Estimate costs before large experiments

## Reporting Issues

Include:
1. Python version: `python --version`
2. Environment: `conda env export`
3. Command run
4. Full error message (last 20 lines minimum)
5. System info: `uname -a`
