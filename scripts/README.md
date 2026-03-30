# Utility Scripts

This directory contains supplementary scripts for data analysis, testing, and performance evaluation.

## Scripts Overview

### `test_models.py`
Tests availability and connectivity with LLM providers (Groq, OpenAI, etc.).
```bash
python scripts/test_models.py
```

### `score.py`
Calculates ROUGE and exact match scores from experimental results.
```bash
python scripts/score.py --defense <defense_type> --provider <provider> --model <model_name>
```

### `analyze_scrape.py`
Analyzes downloaded HTML data for completeness and quality.
```bash
python scripts/analyze_scrape.py
```

### `check_defense.py`
Tests specific defense mechanisms on sample data.
```bash
python scripts/check_defense.py --defense <defense_type>
```

## Design Note

These scripts are exploratory and test-oriented. They support quick validation and iteration during research.
Use them for:
- Debugging specific model/defense combinations
- Quick scoring without full evaluation pipelines
- Data quality assurance
- Provider connectivity checks

