"""
Core API functions for LLM extraction pipeline.

Provides high-level interface for calling LLM APIs (Groq, etc.) with
automatic retry logic, JSON parsing, and prompt style orchestration.
"""

import json
import time
import re
import logging
from typing import Dict, Any, Optional

# Get logger for this module
logger = logging.getLogger("senate_pipeline.api")


def attempt_json_recovery(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to extract JSON from a response that may be wrapped in markdown,
    contain extra text, or have other formatting issues.
    
    Tries strategies in order:
    1. Direct JSON parse
    2. Extract from ```json...``` blocks
    3. Extract from ```...``` blocks
    4. Find JSON object {...} or array [...]
    5. Remove common wrapping text and retry
    
    Args:
        response_text: Raw response from LLM
        
    Returns:
        Parsed JSON dict/list if successful, None otherwise
    """
    if not response_text or not isinstance(response_text, str):
        logger.warning("attempt_json_recovery: received non-string or empty input")
        return None
    
    original_text = response_text.strip()
    
    # Strategy 1: Direct parse
    try:
        return json.loads(original_text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract from ```json...```
    if "```json" in original_text:
        try:
            json_part = original_text.split("```json")[1].split("```")[0].strip()
            return json.loads(json_part)
        except (json.JSONDecodeError, IndexError):
            pass
    
    # Strategy 3: Extract from ```...```
    if "```" in original_text:
        try:
            json_part = original_text.split("```")[1].split("```")[0].strip()
            return json.loads(json_part)
        except (json.JSONDecodeError, IndexError):
            pass
    
    # Strategy 4: Find JSON object {...} or array [...]
    # Match from first { or [ to last matching } or ]
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start_idx = original_text.find(start_char)
        if start_idx != -1:
            end_idx = original_text.rfind(end_char)
            if end_idx != -1 and end_idx > start_idx:
                try:
                    json_part = original_text[start_idx:end_idx+1]
                    return json.loads(json_part)
                except json.JSONDecodeError:
                    pass
    
    # Strategy 5: Remove common wrapping text
    # e.g., "Here's the extracted data: {...}" or "The result is: {...}"
    patterns_to_remove = [
        (r"^[^{[]*", ""),  # Remove text before first { or [
        (r"[^}\]]*$", ""),  # Remove text after last } or ]
    ]
    for pattern, replacement in patterns_to_remove:
        try:
            cleaned = re.sub(pattern, replacement, original_text).strip()
            if cleaned and cleaned[0] in "{[":
                return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
    
    logger.warning(f"attempt_json_recovery failed: could not extract JSON from response")
    return None


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
    Includes comprehensive logging for debugging.
    
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
        logger.error("call_groq: config parameter is required (SessionConfig object)")
        raise ValueError("config parameter is required (SessionConfig object)")
    
    # Prepare full prompt (either format it or append)
    if "{text}" in prompt_template:
        full_prompt = prompt_template.format(text=text)
    else:
        full_prompt = prompt_template + "\n\n" + text
    
    model_name = model_override or config.model
    logger.debug(f"call_groq: starting extraction with model={model_name}, "
                f"prompt_len={len(prompt_template)}, text_len={len(text)}")
    
    for attempt in range(max_retries):
        try:
            response_text = None
            
            # API call to Groq (with configured model, temperature, max_tokens)
            logger.debug(f"call_groq: attempt {attempt + 1}/{max_retries} - calling API")
            response = config.api_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            response_text = response.choices[0].message.content.strip()
            logger.debug(f"call_groq: received response ({len(response_text)} chars)")
            
            # Try direct parse first, then JSON recovery
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                logger.debug("call_groq: direct JSON parse failed, attempting recovery")
                recovered_json = attempt_json_recovery(response_text)
                
                if recovered_json is not None:
                    logger.info("call_groq: successfully recovered JSON from malformed response")
                    return recovered_json
                else:
                    # Recovery failed — return error with raw response
                    logger.warning(f"call_groq: JSON recovery failed, returning error")
                    return {
                        "error": "Failed to parse JSON response (recovery attempted)",
                        "raw": response_text[:200] if response_text else "No response"
                    }
        
        except Exception as e:
            error_str = str(e)
            logger.warning(f"call_groq: attempt {attempt + 1} failed with error: {error_str[:100]}")
            
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
                backoff_seconds = min(2 ** attempt, 120)  # Cap at 120s
                logger.info(f"call_groq: quota limit detected (attempt {attempt + 1}/{max_retries}). "
                           f"Backing off {backoff_seconds}s...")
                time.sleep(backoff_seconds)
                continue
            
            # If not quota error or no retries left, return error
            logger.error(f"call_groq: unrecoverable error after {attempt + 1} attempt(s): {error_str[:100]}")
            return {
                "error": error_str[:100],
                "attempt": attempt + 1,
                "max_retries": max_retries
            }
    
    # Fallback (should rarely reach here)
    logger.error(f"call_groq: max retries ({max_retries}) exceeded")
    return {"error": "Max retries exceeded", "max_retries": max_retries}


def run_pipeline(
    text: str,
    config,
    model_override: str = None
) -> Dict[str, Any]:
    """
    Execute complete extraction pipeline on senator profile text.
    
    Orchestrates all configured prompt styles and returns structured results.
    Uses configured rate limits between style calls.
    Includes comprehensive logging for debugging.
    
    Args:
        text: Cleaned senator biography text
        config: SessionConfig or PipelineConfig object with model, API settings, prompt_styles/prompt_map
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
    logger.info(f"run_pipeline: starting extraction with {len(config.prompt_styles)} prompt style(s)")
    logger.debug(f"run_pipeline: styles={config.prompt_styles}, text_len={len(text)}")
    
    # Multi-style extraction
    if len(config.prompt_styles) > 1:
        results = {}
        
        # Get inter-style delay from config if available, else use default
        inter_style_delay = 3.0
        if hasattr(config, 'rate_limit_config') and config.rate_limit_config:
            inter_style_delay = config.rate_limit_config.inter_style_delay
            logger.debug(f"run_pipeline: using rate_limit_config.inter_style_delay={inter_style_delay}s")
        else:
            logger.debug(f"run_pipeline: using default inter_style_delay={inter_style_delay}s")
        
        for i, style_name in enumerate(config.prompt_styles):
            logger.info(f"run_pipeline: extracting with style '{style_name}' ({i+1}/{len(config.prompt_styles)})")
            prompt = config.prompt_map[style_name]
            results[style_name] = call_groq(
                prompt,
                text,
                config=config,
                model_override=model_override,
                max_retries=5
            )
            
            # Check for errors
            if "error" in results[style_name]:
                logger.warning(f"run_pipeline: style '{style_name}' returned error: {results[style_name]['error']}")
            else:
                logger.debug(f"run_pipeline: style '{style_name}' succeeded")
            
            # Rate limit between styles (except after last style)
            if i < len(config.prompt_styles) - 1:
                logger.debug(f"run_pipeline: rate limiting {inter_style_delay}s before next style")
                time.sleep(inter_style_delay)
        
        logger.info(f"run_pipeline: multi-style extraction complete ({len(results)} styles)")
        return {
            "task1_pii": results,
            "prompt_style": "all_styles"
        }
    else:
        # Single style extraction
        style_name = config.prompt_styles[0]
        logger.info(f"run_pipeline: extracting with single style '{style_name}'")
        prompt = config.prompt_map[style_name]
        task1 = call_groq(
            prompt,
            text,
            config=config,
            model_override=model_override,
            max_retries=5
        )
        
        if "error" in task1:
            logger.warning(f"run_pipeline: style '{style_name}' returned error: {task1['error']}")
        else:
            logger.debug(f"run_pipeline: style '{style_name}' succeeded")
        
        logger.info(f"run_pipeline: single-style extraction complete")
        return {
            "task1_pii": {style_name: task1},
            "prompt_style": style_name
        }