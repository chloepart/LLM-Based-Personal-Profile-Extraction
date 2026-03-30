# LLM-Based Personal Profile Extraction

A comprehensive framework for extracting personal information from documents using Large Language Models, with built-in defense mechanisms and evaluation metrics.

## 🚀 Quick Start

1. **Copy the environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Add your API keys to `.env`:**
   Edit `.env` and add your API keys for the providers you want to use:
   - Google Gemini
   - Groq
   - OpenAI (optional)
   - Anthropic (optional)

3. **Run a simple experiment:**
   ```bash
   cd experiments
   python main.py --model_config_path ../configs/model_configs/groq_config.json \
                  --task_config_path ../configs/task_configs/synthetic.json
   ```

## 📁 Project Structure

```
├── docs/                    # Documentation
│   ├── QUICKSTART.md
│   ├── ARCHITECTURE.md
│   ├── USAGE.md
│   └── ...
│
├── src/LLMPersonalInfoExtraction/  # Main package
│   ├── models/              # LLM implementations
│   ├── attacker/            # Attack strategies
│   ├── defense/             # Defense mechanisms
│   ├── evaluator/           # Evaluation metrics
│   ├── tasks/               # Task managers
│   ├── utils/               # Utilities
│   └── config_loader.py     # Environment variable loader
│
├── experiments/             # Entry points & notebooks
│   ├── main.py              # Single experiment runner
│   ├── run.py               # Batch orchestrator
│   ├── evaluate.py          # Evaluation script
│   ├── senate_scraper.py    # Data scraper
│   └── *.ipynb              # Jupyter notebooks
│
├── scripts/                 # Utility scripts
│   ├── test_models.py
│   ├── score.py
│   ├── analyze_scrape.py
│   └── check_defense.py
│
├── configs/                 # Configuration files
│   ├── model_configs/       # Model configs (API keys via .env)
│   └── task_configs/        # Task configs
│
├── data/                    # Data & prompts
│   ├── synthetic/           # Synthetic datasets
│   ├── system_prompts/      # Task instructions
│   └── icl/                 # In-context learning examples
│
├── outputs/                 # Generated results (git-ignored)
│   ├── result/              # Extraction results
│   ├── log/                 # Execution logs
│   └── senate_results/      # Senate-specific results
│
└── external_data/           # Downloaded data (git-ignored)
    └── senate_html/         # Senator profiles
```

## 🔐 Security

**API Keys Management:**
- All API keys are loaded from `.env` file via environment variables
- `.env` is in `.gitignore` and never committed to git
- `.env.example` shows the required keys (with placeholder values)

**Setup:**
```bash
# Copy template
cp .env.example .env

# Edit with your actual keys (local only)
nano .env
```

## 📚 Documentation

- **[QUICKSTART.md](docs/QUICKSTART.md)** - 5-minute intro
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design
- **[USAGE.md](docs/USAGE.md)** - Detailed usage guide
- **[CONTRIBUTING.md](docs/CONTRIBUTING.md)** - Contribution guidelines

## 🏃 Running Experiments

### Single Experiment
```bash
cd experiments
python main.py --model_config_path ../configs/model_configs/groq_config.json \
               --task_config_path ../configs/task_configs/synthetic.json \
               --defense pi_ci
```

### Batch Experiments
```bash
cd experiments
python run.py
```

### Evaluation
```bash
cd experiments
python evaluate.py --provider groq --dataset synthetic
```

## 📊 Supported Models

- **OpenAI**: GPT-3.5, GPT-4
- **Google**: Gemini, PaLM2
- **Groq**: LLaMA 3.1 (8B), LLaMA 3.3 (70B)
- **Anthropic**: Claude
- **Local**: LLaMA, Vicuna, Flan, InternLM

## 🛡️ Defense Mechanisms

- **PI-CI**: Contextual input injection
- **PI-ID**: Instruction deployment defense
- **PI-CI-ID**: Combined approach
- **Image-based**: Convert text to image

## 📈 Evaluation Metrics

- Accuracy (exact match)
- ROUGE-1 (token overlap)
- BERT-Score (semantic similarity)

## ⚖️ License

MIT License - See LICENSE file for details.

## 📝 Citation

```bibtex
@inproceedings{liu2025evaluating,
  title={Evaluating LLM-based Personal Information Extraction and Countermeasures},
  author={Liu, ..., Gong, ...},
  booktitle={USENIX Security},
  year={2025}
}
```

## 💬 Questions?

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) or check existing issues on GitHub.
