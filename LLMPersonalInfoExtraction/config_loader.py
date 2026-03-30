"""
Configuration loader for API keys and environment variables.
Loads from .env file or environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)


def get_api_key(provider: str) -> str:
    """
    Get API key for a provider from environment variables.
    
    Args:
        provider: One of 'gemini', 'groq', 'openai', 'anthropic'
    
    Returns:
        API key string
        
    Raises:
        ValueError: If API key not found in environment
    """
    key_mapping = {
        'gemini': 'GEMINI_API_KEY',
        'groq': 'GROQ_API_KEY',
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
    }
    
    env_var = key_mapping.get(provider.lower())
    if not env_var:
        raise ValueError(f"Unknown provider: {provider}")
    
    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(
            f"API key for {provider} not found. "
            f"Set {env_var} in .env file or environment variables."
        )
    
    return api_key


def get_debug() -> bool:
    """Get debug mode from environment variables."""
    return os.getenv('DEBUG', 'false').lower() == 'true'
