# Outputs

This folder contains all generated outputs and results from experiments:

## Structure

- **result/** - Raw extraction results from model runs
  - Organized by provider and model name
  - Contains per-dataset metrics

- **log/** - Execution logs and outputs
  - One log file per experiment configuration
  - Useful for debugging and tracking progress

- **senate_results/** - Results specific to Senate profile extraction task
  - task1_pii.csv - Structured information extraction results
  - task2_ideology.csv - Ideological inference results
  - baselines.csv - Comparison with traditional methods (regex, spaCy)
  - model_comparison.csv - Results from different model sizes

## Usage

Results are automatically saved here when running experiments via `main.py` or `run.py`.

To analyze results, see `../docs/USAGE.md` or use `../experiments/evaluate.py`.

## .gitignore

Large output files are excluded from version control. 
Only keep results you explicitly want to track.
