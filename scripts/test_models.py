#!/usr/bin/env python3
"""
Test available LLM models from a provider (e.g., Groq).

This script verifies that API keys are set up correctly and that models
are accessible through their respective APIs.
"""
import sys
sys.path.insert(0, '.')
from LLMPersonalInfoExtraction.utils import open_config
from groq import Groq as GroqClient

def test_groq_models():
    """Test connectivity and availability of Groq models."""
    config = open_config('./configs/model_configs/groq_config.json')
    key = config['api_key_info']['api_keys'][0]
    client = GroqClient(api_key=key)

    models = [
        'llama-3.3-70b-versatile',
        'llama3-70b-8192',
        'mixtral-8x7b-32768',
        'gemma2-9b-it',
    ]

    for model in models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': 'Say the word: hello'}],
                temperature=0.1,
                max_tokens=20
            )
            print('OK: ' + model + ' -> ' + response.choices[0].message.content.strip())
        except Exception as e:
            print('FAIL: ' + model + ' -> ' + str(e)[:60])

if __name__ == '__main__':
    test_groq_models()
