"""
Core API functions for LLM extraction pipeline.

Provides high-level interface for calling LLM APIs (Groq, etc.) with
automatic retry logic, JSON parsing, and prompt style orchestration.
"""

import json
import time
import re
from typing import Dict, Any, Optional


def call_groq(
    prompt_template: str,
    text: str,
    config=None,
    model_override: str = None,
    max_retries: int = 5
) -> Dict[str, Any]:
    """
    Call Groq API with given prompt and text, parse JSON response.
    Supports exponential backoff retry logic for quota/rate limit errors.
    
    Args:
        prompt_template: Prompt string (can include {text} placeholder or will be appended)
        text: Input text to extract PII from
        config: SessionConfig object (required; must have api_client, model, temperature, max_tokens)
        model_override: Optional model name override (default: use config model)
        max_retries: Maximum number of retry attempts (default 5)
        
    Returns:
        Dict with extracted data (parsed from JSON response) or {"error": "..."} on failure
    """
    if config is None:
        raise ValueError("config parameter is required (SessionConfig object)")
    
    # Prepare full prompt (either format it or append)
    if "{text}" in prompt_template:
        full_prompt = prompt_template.format(text=text)
    else:
        full_prompt = prompt_template + "\n\n" + text
    
    for attempt in range(max_retries):
        try:
            model_name = model_override or config.model
            response_text = None
            
            # API call to Groq (with configured model, temperature, max_tokens)
            response = config.api_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            response_text = response.choices[0].message.content.strip()
            
            # Clean up markdown fences if model wrapped response in ```json or ```
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            # Parse JSON and return (if this succeeds, we're done)
            return json.loads(response_text)
        
        except json.JSONDecodeError as e:
            # Model returned non-JSON — return error marker for later filtering
            return {
                "error": "Failed to parse JSON response",
                "raw": response_text[:200] if response_text else "No response"
            }
        
        except Exception as e:
            error_str = str(e)
            
            # Detect quota/rate-limit errors (will trigger exponential backoff)
            is_quota_error = any([
                "RESOURCE_EXHAUSTED" in error_str,
                "quota" in error_str.lower(),
                "rate_limit" in error_str.lower(),
                "rate limit" in error_str.lower(),
                "too many requests" in error_str.lower(),
                "429" in error_str,  # HTTP 429
                "503" in error_str,  # HTTP 503
            ])
            
            # If quota error and retries remain, backoff exponentially
            if is_quota_error and attempt < max_retries - 1:
                backoff_seconds = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
                print(f"  Quota limit hit (attempt {attempt + 1}/{max_retries}). "
                      f"Backing off {backoff_seconds}s...")
                time.sleep(backoff_seconds)
                continue
            
            # If not quota error or no retries left, return error
            return {
                "error": str(e)[:100],
                "attempt": attempt + 1,
                "max_retries": max_retries
            }
    
    # Fallback (should rarely reach here)
    return {"error": "Max retries exceeded", "max_retries": max_retries}


def run_pipeline(
    text: str,
    config,
    model_override: str = None
) -> Dict[str, Any]:
    """
    Execute complete extraction pipeline on senator profile text.
    
    Orchestrates all configured prompt styles and returns structured results.
    Includes automatic rate limiting between style calls.
    
    Args:
        text: Cleaned senator biography text
        config: SessionConfig object with model, API settings, and prompt_styles/prompt_map
        model_override: Optional model name for comparison
        
    Returns:
        Dict with structure:
        {
            "task1_pii": {
                "direct": {...extraction results...},
                "pseudocode": {...},
                "icl": {...},
            } OR {single_style: {...results...}},
            "prompt_style": "all_styles" or style name
        }
    """
    # Multi-style extraction
    if len(config.prompt_styles) > 1:
        results = {}
        for style_name in config.prompt_styles:
            prompt = config.prompt_map[style_name]
            results[style_name] = call_groq(
                prompt,
                text,
                config=config,
                model_override=model_override,
                max_retries=5
            )
            # Rate limit between styles (3s) to reduce quota exhaustion
            time.sleep(3)
        
        return {
            "task1_pii": results,
            "prompt_style": "all_styles"
        }
    else:
        # Single style extraction
        style_name = config.prompt_styles[0]
        prompt = config.prompt_map[style_name]
        task1 = call_groq(
            prompt,
            text,
            config=config,
            model_override=model_override,
            max_retries=5
        )
        
        return {
            "task1_pii": {style_name: task1},
            "prompt_style": style_name
        }
