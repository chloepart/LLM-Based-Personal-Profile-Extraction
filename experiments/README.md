# Experiments & Entry Points

This folder contains the main entry points for running experiments:

## Scripts

- **main.py** - Single experiment runner
  ```bash
  python main.py --model_config_path ../configs/model_configs/groq_config.json \
                 --task_config_path ../configs/task_configs/synthetic.json
  ```

- **run.py** - Batch orchestrator for multiple experiment combinations
  ```bash
  python run.py
  ```

- **evaluate.py** - Post-hoc evaluation metrics (BERT-Score, ROUGE, etc.)
  ```bash
  python evaluate.py --provider groq --dataset synthetic
  ```

- **senate_scraper.py** - Scrape senator profile data

## Notebooks

- **senate_llm_pipeline.ipynb** - Interactive analysis pipeline for senator profiles
- **visualize_results.ipynb** - Visualization of results and metrics

## Configuration

All scripts use configuration files from `../configs/`. API keys should be set in the `.env` file at the project root.

See `../docs/USAGE.md` for detailed examples.
