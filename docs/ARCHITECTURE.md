# Architecture Guide

Comprehensive documentation of the codebase structure, module organization, and design patterns.

## Overview

The codebase is organized as a modular Python package `LLMPersonalInfoExtraction` (PIE) with clear separation of concerns:

```
PIE/
├── models/          # LLM provider abstraction layer
├── attacker/        # Attack strategies
├── defense/         # Defense mechanisms
├── evaluator/       # Evaluation metrics
├── tasks/           # Dataset/task management
└── utils/           # Utilities & helpers
```

Each module is independently extensible and follows a factory pattern for instantiation.

## 1. Models Module (`LLMPersonalInfoExtraction/models/`)

Abstracts different LLM providers behind a unified interface.

### Base Class: `Model`

```python
class Model:
    """Base interface for all LLM models."""
    
    def query(self, msg: str, image=None) -> str:
        """Send a prompt to the model and get response."""
        raise NotImplementedError
    
    def print_model_info(self) -> None:
        """Print model details."""
```

### Implementations

| Class | Provider | Type | Features |
|-------|----------|------|----------|
| `GPT` | OpenAI | API | Text + image support |
| `Gemini` | Google | API | Text + image support |
| `Groq` | Groq | API | Optimized for speed |
| `PaLM2` | Google | API | Deprecated model |
| `Llama` | Meta | Local | Requires GPU |
| `Vicuna` | UC Berkeley | Local | Requires GPU |
| `Flan` | Google | Local | CPU-friendly |
| `Internlm` | InternLM | Local | Requires GPU |

### Factory Pattern

```python
# LLMPersonalInfoExtraction/models/__init__.py
def create_model(config: dict) -> Model:
    """Factory function to instantiate models."""
    provider = config["model_info"]["provider"]
    if provider == 'gpt':
        return GPT(config)
    elif provider == 'gemini':
        return Gemini(config)
    # ... etc
```

### Configuration Format

```json
{
  "model_info": {
    "provider": "gpt",
    "name": "gpt-4",
    "type": "api"
  },
  "api_key_info": {
    "api_keys": ["key1", "key2"],
    "api_key_use": 0
  },
  "params": {
    "temperature": 0.7,
    "max_output_tokens": 100,
    "seed": 42,
    "gpus": ["0"]
  }
}
```

### Adding a New Model

1. **Create a new file** `LLMPersonalInfoExtraction/models/MyModel.py`:

```python
from .Model import Model

class MyModel(Model):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your model
        self.client = MyAPIClient(api_key=config['api_key_info']['api_keys'][0])
    
    def query(self, msg: str, image=None) -> str:
        """Implement query logic."""
        response = self.client.call(msg)
        return response.text
```

2. **Register in** `LLMPersonalInfoExtraction/models/__init__.py`:

```python
from .MyModel import MyModel

def create_model(config):
    # ... existing code ...
    elif provider == 'mymodel':
        return MyModel(config)
```

3. **Create config** `configs/model_configs/mymodel_config.json` with template.

## 2. Tasks Module (`LLMPersonalInfoExtraction/tasks/`)

Manages datasets and generates prompts for information extraction tasks.

### `TaskManager` Class

```python
class TaskManager:
    """Loads and manages dataset samples."""
    
    def __init__(self, config: dict):
        # Load dataset from config["dataset_info"]["path"]
        
    def __len__(self) -> int:
        # Return total number of profiles
        
    def __getitem__(self, idx: int) -> tuple:
        # Return (raw_html_list, label_dict) for profile[idx]
```

### `ICLManager` Class

```python
class ICLManager:
    """In-context learning examples for few-shot prompting."""
    
    def __init__(self, config: dict):
        # Load ICL examples from config["dataset_info"]["icl_path"]
        
    def __getitem__(self, idx: int) -> tuple:
        # Return (parsed_profile, label) for ICL example[idx]
```

### Configuration Format

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

### Dataset Structure

Each dataset directory should contain:

```
data/{dataset}/
├── profile1.html
├── profile2.html
├── ...
└── labels.json          # Maps filename -> {info_type: value}
```

Example `labels.json`:

```json
{
  "profile1": {
    "email": "john@example.com",
    "phone": "555-1234",
    "name": "John Doe",
    "birth_year": "1990"
  }
}
```

## 3. Attacker Module (`LLMPersonalInfoExtraction/attacker/`)

Implements attack strategies for information extraction.

### `Attacker` Class

```python
class Attacker:
    """Executes attacks on models."""
    
    def __init__(self, model: Model, prompt_type: str, ...):
        self.model = model
        self.prompt_type = prompt_type
        # ICL examples, templates, etc.
    
    def attack(self, profile: str, info_type: str) -> str:
        """Execute attack to extract specific info type."""
        prompt = self._construct_prompt(profile, info_type)
        return self.model.query(prompt)
    
    def _construct_prompt(self, profile: str, info_type: str) -> str:
        """Build prompt based on attack type."""
```

### Supported Prompt Types

- **direct**: Simple question
  ```
  What is the person's email?
  ```

- **icl**: In-context learning (few-shot)
  ```
  Example 1: [profile] -> email: [email]
  ...
  Example N: [profile] -> email: ?
  ```

- **pseudocode**: Structured reasoning
  ```
  Step 1: Look for email patterns...
  ```

- **Adaptive attacks**: Adversarial techniques
  - `sandwich`: Wrap instructions with benign text
  - `xml`: Use XML tags for structure
  - `delimiters`: Custom delimiters
  - `random_seq`: Random character injection
  - `instructional`: Contradictory instructions
  - `paraphrasing`: Rephrase the question
  - `retokenization`: Break tokens strategically

### Factory Pattern

```python
def create_attacker(model: Model, 
                   adaptive_attack: str = 'no',
                   icl_manager = None,
                   prompt_type: str = 'direct') -> Attacker:
    """Create attacker with specified configuration."""
```

## 4. Defense Module (`LLMPersonalInfoExtraction/defense/`)

Implements defensive mechanisms to protect against attacks.

### Base Class: `Defense`

```python
class Defense:
    """Base class for all defenses."""
    
    def __init__(self):
        self.defense = 'base'
    
    def apply(self, profile: str, label: dict) -> str:
        """Apply defense to profile before querying model."""
        raise NotImplementedError
```

### Implemented Defenses

| Type | Name | Mechanism |
|------|------|-----------|
| **Trivial** | `replace_at` | Replace `@` with `AT` |
| | `replace_dot` | Replace `.` with `_` |
| | `replace_at_dot` | Both replacements |
| | `hyperlink` | Convert email/phone to links |
| | `mask` | Replace text with `[REDACTED]` |
| **Smart** | `pi_ci` | Context-aware obfuscation |
| | `pi_id` | Identity-aware obfuscation |
| | `pi_ci_id` | Combination of both |
| **Image** | `image` | Convert HTML to image |

### Adding a New Defense

1. **Create** `LLMPersonalInfoExtraction/defense/MyDefense.py`:

```python
from .Defense import Defense

class MyDefense(Defense):
    def __init__(self):
        super().__init__()
        self.defense = 'my_defense'
    
    def apply(self, profile: str, label: dict) -> str:
        """Apply your defense logic."""
        # Transform profile to obscure PII
        modified = profile.replace(label['email'], '[HIDDEN]')
        return modified
```

2. **Register in** `LLMPersonalInfoExtraction/defense/__init__.py`:

```python
from .MyDefense import MyDefense

def create_defense(defense_type: str) -> Defense:
    # ... existing code ...
    elif defense_type == 'my_defense':
        return MyDefense()
```

3. **Update config** `configs/task_configs/` to include your defense type.

## 5. Evaluator Module (`LLMPersonalInfoExtraction/evaluator/`)

Measures attack success and defense effectiveness.

### `Evaluator` Class

```python
class Evaluator:
    """Evaluates extraction accuracy."""
    
    def __init__(self, provider: str, info_categories: list, metric_2: str = None):
        self.info_categories = info_categories
        # Set up metric functions
    
    def update(self, prediction: str, label: dict, 
               info_type: str, defense: Defense, verbose: int = 1) -> dict:
        """Evaluate single prediction."""
        
    def print_result(self) -> None:
        """Print aggregated results."""
```

### Metrics

- **Exact Match**: For short fields (email, phone)
  ```
  score = predicted.lower() == label.lower()
  ```

- **ROUGE-1**: For longer text (biography, notes)
  ```
  F-score based on unigram overlap
  ```

- **BERT-Score** (optional): Semantic similarity
  ```
  Contextual embeddings comparison
  ```

## 6. Utils Module (`LLMPersonalInfoExtraction/utils/`)

Helper functions for parsing, configuration, and data processing.

### Key Functions

```python
# Configuration
open_config(path: str) -> dict
print_config(config: dict) -> None

# Data
open_txt(path: str) -> list[str]
open_json(path: str) -> dict

# Parsing
get_parser(dataset: str, include_link: bool = True) -> HTMLParser
parsed_data_to_string(dataset: str, data: dict, model_name: str) -> str

# Prompting
load_instruction(prompt_type: str, info_cats: list) -> dict
load_image(image_path: str) -> Image

# Text
remove_symbols(text: str) -> str
```

### HTMLParser

Custom HTML parser that extracts structured data from profiles:

```python
parser = get_parser('synthetic')
parser.feed(raw_html)
structured_data = parser.data
# Returns: {'name': '...', 'email': '...', 'phone': '...', ...}
```

## Workflow: From Raw Profile to Evaluation

```
1. Load Profile
   TaskManager[idx] -> raw_html_list, label_dict
   
2. Apply Defense
   defense.apply(raw_html, label) -> protected_profile
   
3. Parse to Text
   HTMLParser -> structured_data -> parsed_text
   
4. Construct Attack Prompt
   attacker._construct_prompt(parsed_text, info_type)
   
5. Query Model
   model.query(prompt) -> raw_response
   
6. Evaluate
   evaluator.update(raw_response, label, info_type, defense)
```

## Main Execution Flow

### Single Experiment (`main.py`)

```python
1. Load configs (model, task, defense)
2. Create components
   - model = create_model(model_config)
   - task_manager, icl_manager = create_task(task_config)
   - defense = create_defense(args.defense)
   - attacker = create_attacker(model, ...)
   - evaluator = create_evaluator(...)

3. For each profile in dataset:
   - Get raw_html, label
   - Apply defense
   - For each info_type:
     - Construct attack prompt (with ICL, adaptive, etc.)
     - Query model
     - Evaluate prediction
   
4. Save results
   - Raw responses -> result/.../all_raw_responses.npz
   - Metrics -> log files
```

### Batch Experiments (`run.py`)

```python
For each combination of:
  - Model
  - Dataset
  - Defense mechanism
  - Prompt type
  - ICL settings
  - Adaptive attack
  
  -> Execute main.py with specific args
  -> Log results
```

### Post-hoc Evaluation (`evaluate.py`)

```python
1. Load saved raw_responses.npz
2. Compute additional metrics (BERT-Score)
3. Print aggregate statistics
```

## Design Patterns Used

### 1. Factory Pattern

Used in `create_model()`, `create_defense()`, `create_task()`, `create_attacker()`.

**Benefit**: Easy to add new implementations without modifying calling code.

### 2. Strategy Pattern

- Attack types (direct, ICL, adaptive, etc.) as strategies
- Defense mechanisms as strategies

**Benefit**: Runtime switching between different approaches.

### 3. Configuration-Driven Design

All behavior controlled via JSON configs, not code changes.

**Benefit**: Reproducibility and easy experiment variation.

### 4. Data Manager Pattern

`TaskManager` and `ICLManager` abstract dataset complexity.

**Benefit**: Support multiple datasets with minimal code changes.

## Extension Points

### Add a New Dataset

1. Create `data/{mydata}/` with HTML files and `labels.json`
2. Create `configs/task_configs/mydata.json`
3. If custom HTML parsing needed, extend `HTMLParser` in `utils/parser.py`

### Add a New Model Provider

1. Create `LLMPersonalInfoExtraction/models/MyProvider.py` extending `Model`
2. Register in `models/__init__.py`
3. Create `configs/model_configs/myprovider_config.json`

### Add a New Defense

1. Create `LLMPersonalInfoExtraction/defense/MyDefense.py` extending `Defense`
2. Register in `defense/__init__.py`
3. Use in `run.py` or `main.py` via `--defense mydefense`

### Add a New Metric

1. Implement in `Evaluator.update()` method
2. Extend `evaluator.print_result()` to display metric
3. Use via `evaluate.py --m2 mymetric`

## Performance Considerations

### Memory
- Batch process profiles to manage memory
- Use `npz` format for efficient result storage (compresses well)

### Speed
- API-based models can be throttled by rate limits
- Use `groq_config.json` for free fast inference
- Cache model responses when possible

### Scalability
- `TaskManager.__getitem__()` loads profiles on-demand
- Results streamed to disk incrementally
- Easily parallelizable across multiple machines

## Testing

### Unit Tests
Place tests in `tests/` directory:
```bash
python -m pytest tests/
```

### Integration Tests
Run on sample dataset:
```bash
python main.py --task_config_path configs/task_configs/synthetic.json
```

### Validation Scripts
Use utilities in `scripts/`:
- `test_models.py` - Verify Model implementations
- `check_defense.py` - Verify Defense mechanisms
- `score.py` - Verify Evaluator output

## Code Style

- Follow PEP 8
- Use type hints where possible
- Document public APIs with docstrings
- Use meaningful variable names

## Further Reading

- **Paper**: [https://arxiv.org/abs/2408.07291](https://arxiv.org/abs/2408.07291)
- **Experiment Log Format**: See `log/` directory samples
- **Result Format**: `.npz` files contain dicts; load with `np.load(..., allow_pickle=True)`
