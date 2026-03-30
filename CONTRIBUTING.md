# Contributing Guide

Thank you for your interest in contributing to this LLM-based Personal Information Extraction research!

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions.

## How to Contribute

### 1. Reporting Issues

**Found a bug?** Open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment info:
  ```bash
  python --version
  conda env export
  ```
- Last 20 lines of error output

**Have a feature request?** Describe:
- Use case and motivation
- How it benefits research
- Rough implementation idea

### 2. Proposing an Enhancement

Before implementing, open an issue to discuss:
- What problem it solves
- Design approach
- Expected effort
- Any breaking changes

### 3. Contributing Code

#### Setup Development Environment

```bash
# Clone and setup
git clone <your-fork>
cd LLM-Based-Personal-Profile-Extraction
conda env create -f PIE_environment.yml
conda activate PIE

# Install dev dependencies (if you add any)
pip install pytest black pylint
```

#### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make focused changes**
   - One feature per PR
   - Clear commit messages
   - Reference issue numbers

3. **Follow code style**
   - PEP 8 compliance
   - Type hints where possible
   - Docstrings for public APIs
   - Meaningful variable names

4. **Test your changes**
   - For new models: test with `scripts/test_models.py`
   - For new defenses: test with `scripts/check_defense.py`
   - For core logic: add unit tests in `tests/`

5. **Update documentation**
   - Update README if user-facing
   - Update ARCHITECTURE.md for major changes
   - Add docstrings to new functions
   - Update USAGE.md for new features

#### Example: Adding a New Model

```python
# 1. Create LLMPersonalInfoExtraction/models/MyProvider.py
from .Model import Model

class MyProvider(Model):
    """
    Integration with MyProvider LLM API.
    
    Supports:
    - Text and image inputs
    - Stream and async queries
    - Rate limiting
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.client = MyProviderClient(config['api_key_info']['api_keys'][0])
    
    def query(self, msg: str, image=None) -> str:
        """Query the model with text or image input."""
        if image:
            return self.client.query_with_image(msg, image)
        return self.client.query_text(msg)

# 2. Register in models/__init__.py
from .MyProvider import MyProvider

def create_model(config):
    # ... existing code ...
    elif provider == 'myprovider':
        return MyProvider(config)

# 3. Create config template configs/model_configs/myprovider_config.json
{
  "model_info": {
    "provider": "myprovider",
    "name": "model-identifier",
    "type": "api"
  },
  "api_key_info": {
    "api_keys": ["your-api-key"],
    "api_key_use": 0
  },
  "params": {
    "temperature": 0.7,
    "max_output_tokens": 500,
    "seed": 42,
    "gpus": []
  }
}

# 4. Test
python scripts/test_models.py

# 5. Document in ARCHITECTURE.md
# Add row to models table and implementation example
```

#### Example: Adding a New Defense

```python
# 1. Create LLMPersonalInfoExtraction/defense/MyDefense.py
from .Defense import Defense

class MyDefense(Defense):
    """
    Custom privacy defense mechanism.
    
    Applies [description of technique] to obscure PII.
    Effectiveness: [estimate] reduction in extraction success.
    """
    
    def __init__(self):
        super().__init__()
        self.defense = 'my_defense'
    
    def apply(self, profile: str, label: dict) -> str:
        """Apply defense to profile."""
        modified = profile
        
        # Apply defenses
        if 'email' in label:
            modified = modified.replace(label['email'], '[REDACTED_EMAIL]')
        if 'phone' in label:
            modified = modified.replace(label['phone'], '[REDACTED_PHONE]')
        
        return modified

# 2. Register in defense/__init__.py
from .MyDefense import MyDefense

def create_defense(defense_type):
    # ... existing code ...
    elif defense_type == 'my_defense':
        return MyDefense()

# 3. Test
python scripts/check_defense.py --defense my_defense

# 4. Run experiments
python main.py --defense my_defense ...
```

#### Example: Adding a New Dataset

See [USAGE.md - Add Your Own Dataset](USAGE.md#add-your-own-dataset)

### 4. Pull Request Process

1. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create pull request** with:
   - Clear title describing change
   - Link to related issue
   - Summary of changes
   - Any breaking changes
   - Test results if applicable

3. **Address feedback**
   - Respond to review comments
   - Make requested changes
   - Push updates to same branch

4. **Merge**
   - Ensure CI passes
   - Get approval
   - Squash commits if requested

## Development Guidelines

### Code Organization

- **Package structure**: Follow existing patterns
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes
- **Imports**: Group by standard, external, local
- **Comments**: Explain *why*, not *what*

### Documentation

- **Docstrings**: Google style
  ```python
  def extract_info(text: str, info_type: str) -> str:
      """Extract specific information from text.
      
      Args:
          text: Input text to extract from
          info_type: Type of information (email, phone, etc.)
      
      Returns:
          Extracted information or empty string if not found
      
      Raises:
          ValueError: If info_type is invalid
      """
  ```

- **Type hints**: Use for public APIs
  ```python
  def query(self, msg: str, image=None) -> str:
  ```

- **Inline comments**: For complex logic
  ```python
  # Skip profiles with missing labels (corrupted data)
  if not label:
      continue
  ```

### Testing

- Use `scripts/` utilities for quick validation
- Document any new test scripts
- Include both success and error cases

### Performance

- Consider memory for large datasets
- Cache expensive computations where possible
- Document assumptions about input sizes
- Provide performance benchmarks for major changes

### Error Handling

Good:
```python
try:
    response = model.query(prompt)
except APIError as e:
    logger.error(f"API error: {e}")
    raise
```

Avoid:
```python
try:
    response = model.query(prompt)
except:  # Too broad!
    pass
```

## Areas Needing Help

### High Priority
- [ ] Add BERT-Score evaluation for all models
- [ ] Optimize memory usage for large datasets
- [ ] Add more open-source models (LLaMA2, Mistral, etc.)
- [ ] Create unit test suite

### Medium Priority
- [ ] Add result visualization dashboard
- [ ] Support multi-modal inputs (document images)
- [ ] Add adaptive attack strategy generation
- [ ] Create web interface for running experiments

### Low Priority
- [ ] Additional datasets (if public sources identified)
- [ ] Optimize API calls for cost reduction
- [ ] Create Docker environments
- [ ] Add model quantization support

## Communication

- **Issues**: For bugs, features, questions
- **Discussions**: For design decisions, research ideas
- **Email**: For sensitive issues or private contact

## Citations

When contributing research, include:
- Your name and affiliation
- Related publications
- Datasets (with proper attribution)

## Licensing

- Code: Apache 2.0
- Datasets: See individual licenses
- Contributions: Must be compatible with Apache 2.0

## Questions?

- Check [ARCHITECTURE.md](ARCHITECTURE.md) for design details
- Review [USAGE.md](USAGE.md) for examples
- Look at similar implementations for patterns
- Open an issue for clarification

## Acknowledgments

Contributors will be:
- Listed in CONTRIBUTORS.md
- Thanked in release notes
- Credited for significant contributions

Thank you for making this research accessible and reproducible!
