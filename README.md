# LLM-Based Personal Information Extraction and Countermeasures

**Evaluating Large Language Model based Personal Information Extraction and Countermeasures**

This is the official repository for research on evaluating personal profile extraction techniques using Large Language Models (LLMs) and their corresponding defensive countermeasures. The work is published at USENIX Security 2025.

## 📖 Overview

This repository provides a comprehensive framework for:

- **Attacking**: Extracting personal information (email, phone, names, etc.) from HTML profiles using LLMs
- **Defending**: Implementing and evaluating defense mechanisms against such attacks
- **Evaluating**: Measuring both attack success and defense robustness across multiple attack vectors

The codebase supports multiple LLM providers (OpenAI, Google, Groq, Anthropic, open-source models) and multiple datasets (synthetic, celebrity, physician profiles).

## 📚 Documentation

Before running experiments, refer to the comprehensive guides:

- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute setup and first experiment
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Detailed module organization and design
- **[NOTEBOOKS.md](NOTEBOOKS.md)** - Data pipeline and notebook documentation
- **[USAGE.md](USAGE.md)** - Comprehensive examples and configuration

## 🔧 Setup

### Prerequisites

- Python 3.8+
- API keys for desired LLM providers (OpenAI, Google, Groq, etc.)
- ~10GB disk space for datasets and results

### Installation

```bash
# Create conda environment from file
conda env create -f PIE_environment.yml
conda activate PIE
```

### Configure API Keys

Add your API keys to `configs/model_configs/`:

```bash
vim configs/model_configs/gpt_config.json
vim configs/model_configs/gemini_config.json
vim configs/model_configs/groq_config.json
```

See [QUICKSTART.md](QUICKSTART.md) for detailed setup.

## 📁 Project Structure

```
LLMPersonalInfoExtraction/              # Core package
├── attacker/                           # Attack implementation
├── defense/                            # Defense mechanisms
├── evaluator/                          # Evaluation metrics
├── models/                             # LLM provider integrations
├── tasks/                              # Dataset management
└── utils/                              # Parsing, config, helpers

configs/                                # Configuration templates
├── model_configs/                      # LLM provider configs
└── task_configs/                       # Dataset/task configs

data/                                   # Datasets & prompts
├── synthetic/                          # Synthetic HTML profiles
├── system_prompts/                     # Task instructions
└── icl/                                # In-context learning examples

scripts/                                # Utility scripts
├── test_models.py                      # Validate connectivity
├── score.py                            # Quick evaluation
├── analyze_scrape.py                   # Data quality checks
└── check_defense.py                    # Defense testing

main.py                                 # Core experiment runner
run.py                                  # Batch orchestration
evaluate.py                             # Post-hoc evaluation
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for full documentation.

## 🚀 Quick Start

### 1. Run a Single Experiment

```bash
python main.py \
    --model_config_path configs/model_configs/groq_config.json \
    --model_name llama-3.1-8b-instant \
    --task_config_path configs/task_configs/synthetic.json \
    --defense no \
    --prompt_type direct
```

### 2. Configure Batch Experiments

Edit `run.py`:

```python
model_info = ['groq', 'llama-3.1-8b-instant']
datasets = ['synthetic']
defenses = ['no', 'pi_ci', 'pi_id']
prompt_types = ['direct']

python run.py
```

### 3. Evaluate Results

```bash
python evaluate.py \
    --provider groq \
    --model_name llama-3.1-8b-instant \
    --dataset synthetic \
    --defense pi_ci
```

For more details, see [QUICKSTART.md](QUICKSTART.md).

## 📊 Supported Features

### Attack Prompt Types
- **Direct**: Simple question answering
- **In-Context Learning (ICL)**: Few-shot examples  
- **Pseudocode**: Structured reasoning
- **Adaptive Attacks**: Adversarial techniques (8+ variants)

### Defense Mechanisms

| Type | Examples |
|------|----------|
| **Trivial** | Replace text, masking, hyperlinks |
| **Smart** | `pi_ci` (context), `pi_id` (identity), `pi_ci_id` (combined) |
| **Encoding** | Image-based profiles |

### Models

- **API-based**: GPT-4, Gemini, PaLM 2, Claude, Groq
- **Open-source**: LLaMA, Flan-T5, Vicuna, InternLM
- Easy to add more via `LLMPersonalInfoExtraction/models/`

### Datasets

| Dataset | Profiles | Source | Availability |
|---------|----------|--------|--------------|
| Synthetic | 100 | GPT-4 generated | ✅ Included |
| Celebrity | 50+ | Public web | 📧 On request |
| Physician | 50+ | Public web | 📧 On request |
| Court | 100+ | Public records | Auto-download |

## 📈 Results Format

Results saved to `result/{provider}_{model}/{dataset}_{defense}_{prompt}_{icl}_{adaptive}/`:

```
all_raw_responses.npz          # Raw LLM outputs
all_raw_responses_with_parsed.npz  # Parsed outputs
...
```

Analyze with `evaluate.py` or `scripts/score.py`.

## 🛠️ Development

### Adding a New Defense

See [ARCHITECTURE.md - Extending](ARCHITECTURE.md#extending-the-framework) for detailed steps.

### Adding a New Model Provider

Implement `Model` interface in `LLMPersonalInfoExtraction/models/`. See examples in `GPT.py`, `Gemini.py`, etc.

## 📋 Utility Scripts

Quick tools in `scripts/`:

```bash
python scripts/test_models.py     # Validate API connectivity
python scripts/analyze_scrape.py  # Analyze HTML data quality
python scripts/score.py           # Quick performance summary
python scripts/check_defense.py   # Debug a defense mechanism
```

See [scripts/README.md](scripts/README.md).

## ⚠️ Important Notes

### Library Compatibility

`PIE_environment.yml` tested on 2024-01-27. Some libraries may need updates:

```bash
# If you encounter API errors, upgrade:
pip install --upgrade openai google-generativeai anthropic
```

### Cost Considerations

Running API-based models incurs costs:

| Provider | Cost |
|----------|------|
| **Groq** | Free tier available |
| **OpenAI** | ~$0.01-0.10/exp |
| **Google** | Credit-based |
| **Anthropic** | Credit-based |

## 📥 Data

Download datasets from [Zenodo](https://zenodo.org/records/14737200):

```bash
# Extract to data/
unzip zenodo_data.zip -d data/
```

**Dataset Notes:**
- `data/synthetic/` - Included in repo
- `data/{celebrity,physician}/` - Request access
- `labels.json` - Ground truth labels per dataset

## 📖 Citation & Credits

### Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{liu2025llminfoextraction,
    title={Evaluating Large Language Model based Personal Information Extraction and Countermeasures},
    author={Yupei Liu and Yuqi Jia and Jinyuan Jia and Neil Zhenqiang Gong},
    booktitle={USENIX Security Symposium},
    year={2025}
}
```

**Paper**: [https://arxiv.org/abs/2408.07291](https://arxiv.org/abs/2408.07291)  
**Artifact**: [Zenodo](https://zenodo.org/records/14737200)

### Acknowledgments

This project builds on the contributions of:

- **Research**: USENIX Security Symposium for publishing
- **Infrastructure**: LLM providers (OpenAI, Google, Groq, Anthropic) for API access and support
- **Open Source**: PyTorch, spaCy, Hugging Face Transformers, NLTK, and scikit-learn
- **Community**: Dataset contributors and research benchmarks

See `PIE_environment.yml` for complete library versions and dependencies.

## 📞 Support & Contributing

- **Setup Issues**: See [QUICKSTART.md](QUICKSTART.md)
- **Technical Questions**: Check [ARCHITECTURE.md](ARCHITECTURE.md)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)
- **Bug Reports**: Open an issue with error details

## 📄 License

Apache 2.0 License - see [LICENSE](LICENSE) for details.
