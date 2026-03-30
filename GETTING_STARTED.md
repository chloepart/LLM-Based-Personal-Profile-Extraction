# Getting Started - Quick Navigation

**Start here** if you're new to this project.

## 🎯 I want to...

### Understand what this project does
→ Read **[README.md](README.md)** (5 min)

### Run my first experiment (fastest way)
→ Follow **[QUICKSTART.md](QUICKSTART.md)** (10 min)

### Understand the code architecture
→ Read **[ARCHITECTURE.md](ARCHITECTURE.md)** (20 min)

### Learn all available options and configurations
→ Read **[USAGE.md](USAGE.md)** (30 min)

### Add a new model/defense/dataset
→ Follow **[CONTRIBUTING.md](CONTRIBUTING.md)** (guides provided)

### Find where specific files are located
→ Check **[MANIFEST.md](MANIFEST.md)** (directory reference)

### Understand the data exploration notebooks
→ Read **[NOTEBOOKS.md](NOTEBOOKS.md)**

### See what was improved and why
→ Read **[OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md)**

---

## 🚀 Quick Command Reference

### Install Everything
```bash
conda env create -f PIE_environment.yml
conda activate PIE
```

### Add Your API Key
```bash
# Edit and add your API key
vim configs/model_configs/groq_config.json
```

### Run One Experiment
```bash
python main.py \
    --model_config_path configs/model_configs/groq_config.json \
    --model_name llama-3.1-8b-instant \
    --task_config_path configs/task_configs/synthetic.json
```

### Quick Diagnostics
```bash
python scripts/test_models.py      # Verify API connectivity
python scripts/check_defense.py    # Test defenses
python scripts/score.py            # Quick evaluation
```

### Run Multiple Experiments
```bash
# Edit run.py to configure
vim run.py

# Execute
python run.py
```

---

## 📚 Documentation Map

```
Entry Points:
├── README.md ..................... Project overview
├── QUICKSTART.md ................. 5-min setup
├── GETTING_STARTED.md ............ This file

Understanding:
├── ARCHITECTURE.md ............... Technical design
├── MANIFEST.md ................... File structure
├── NOTEBOOKS.md .................. Data exploration

Usage:
├── USAGE.md ...................... All options explained  
├── scripts/README.md ............. Utility scripts

Contributing:
├── CONTRIBUTING.md ............... Development guide
├── OPTIMIZATION_SUMMARY.md ....... What changed and why

Configuration Templates:
├── configs/model_configs/template_api_model.json
├── configs/model_configs/template_local_model.json
└── configs/task_configs/template.json
```

---

## 🛠️ For Different Users

### For Researchers Running Experiments

1. Read: [QUICKSTART.md](QUICKSTART.md)
2. Read: [USAGE.md](USAGE.md) - All configuration options
3. Run experiments using `main.py` or `run.py`
4. Analyze results with `evaluate.py` and `scripts/score.py`

### For Developers Extending Code

1. Read: [ARCHITECTURE.md](ARCHITECTURE.md)
2. Read: [CONTRIBUTING.md](CONTRIBUTING.md)
3. Check examples in subdirectories (models/, defense/, etc.)
4. Test with `scripts/` utilities before committing

### For LLMs/AI Assistants

1. Start with: [README.md](README.md)
2. Then: [ARCHITECTURE.md](ARCHITECTURE.md)
3. For specifics: [USAGE.md](USAGE.md)
4. For extending: [CONTRIBUTING.md](CONTRIBUTING.md)
5. Full context: Everything in this repo

---

## ⏱️ Time to First Result

| Goal | Time | Next Step |
|------|------|-----------|
| Understand project | 5 min | README.md |
| Run first experiment | 15 min | QUICKSTART.md |
| Understand design | 20 min | ARCHITECTURE.md |
| Configure batch run | 30 min | USAGE.md + run.py |
| Add custom defense | 45 min | CONTRIBUTING.md |
| Full expert understanding | 2 hours | Read all docs + explore code |

---

## 🎓 Learning Path Depending on Your Role

### Research Scientist
```
README → QUICKSTART → USAGE → Notebooks → Run Experiments
```

### ML Engineer
```
README → ARCHITECTURE → CONTRIBUTING → Add Extensions
```

### DevOps/Admin
```
README → MANIFEST → Setup tools → Docker support
```

### Curious Learner / LLM
```
README → ARCHITECTURE → USAGE → CONTRIBUTING → Code review
```

---

## ❓ Common Questions

**Q: Where do I put my API key?**
A: `configs/model_configs/{provider}_config.json`
See: [QUICKSTART.md](QUICKSTART.md#2-set-up-api-key)

**Q: How do I run my first experiment?**
A: Follow [QUICKSTART.md](QUICKSTART.md#3-run-first-experiment)

**Q: What parameters can I configure?**
A: See [USAGE.md](USAGE.md#parameter-definitions)

**Q: How do I add a new model?**
A: See [ARCHITECTURE.md](ARCHITECTURE.md#adding-a-new-model) and [CONTRIBUTING.md](CONTRIBUTING.md#adding-a-new-model)

**Q: What do experiment results look like?**
A: See [USAGE.md](USAGE.md#interpreting-results)

**Q: Where are my results saved?**
A: `result/{provider}_{model}/{dataset}_{defense}_{prompt}_{icl}_{adaptive}/`
See: [MANIFEST.md](MANIFEST.md#results-result)

**More Q&A:** See specific documentation files listed above

---

## 🔗 Key Files (For Reference)

| Purpose | File |
|---------|------|
| Run experiment | [`main.py`](main.py) |
| Batch orchestration | [`run.py`](run.py) |
| Evaluation | [`evaluate.py`](evaluate.py) |
| Core package | [`LLMPersonalInfoExtraction/`](LLMPersonalInfoExtraction/) |
| Configs | [`configs/`](configs/) |
| Data | [`data/`](data/) |
| Utilities | [`scripts/`](scripts/) |

---

## 📝 Citation

If you use this code:

```bibtex
@inproceedings{liu2025llminfoextraction,
    title={Evaluating Large Language Model based Personal Information Extraction and Countermeasures},
    author={Yupei Liu and Yuqi Jia and Jinyuan Jia and Neil Zhenqiang Gong},
    booktitle={USENIX Security Symposium},
    year={2025}
}
```

**Paper**: https://arxiv.org/abs/2408.07291
**Artifact**: https://zenodo.org/records/14737200

---

## 🤝 Questions or Issues?

1. Check the relevant documentation file above
2. Run utility scripts: `python scripts/test_models.py`
3. Open an issue with full context
4. See [CONTRIBUTING.md](CONTRIBUTING.md#reporting-issues)

---

**Ready?** → Start with [QUICKSTART.md](QUICKSTART.md) 🚀
