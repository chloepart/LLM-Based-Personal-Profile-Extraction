"""
HTTP utilities with retry logic and rate limiting.
Consolidates all web requests and API error handling.
"""

import re
import time
import json
import logging
from typing import Dict, Optional, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry as UrllibRetry


# ─────────────────────────────────────────────────────────────────────────────
# HTTP SESSION WITH AUTOMATIC RETRY
# ─────────────────────────────────────────────────────────────────────────────

def create_session(
    max_retries: int = 3,
    backoff_factor: float = 0.5,
    status_forcelist: tuple = (429, 500, 502, 503, 504)
) -> requests.Session:
    """
    Create a requests session with automatic retry on network errors.
    
    Args:
        max_retries: Number of retries
        backoff_factor: Exponential backoff factor
        status_forcelist: HTTP status codes to retry on
        
    Returns:
        Configured requests.Session
        
    Example:
        >>> session = create_session()
        >>> response = session.get("https://example.com")
    """
    session = requests.Session()
    
    retry_strategy = UrllibRetry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["GET", "POST"],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


# ─────────────────────────────────────────────────────────────────────────────
# SAFE HTTP REQUESTS
# ─────────────────────────────────────────────────────────────────────────────

def safe_get(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 10,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    logger: Optional[logging.Logger] = None,
) -> Optional[requests.Response]:
    """
    Make a safe HTTP GET request with exponential backoff retry.
    
    Args:
        url: URL to fetch
        headers: Optional HTTP headers
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        backoff_factor: Exponential backoff multiplier
        logger: Optional logger for warnings
        
    Returns:
        Response object on success, None on failure
        
    Example:
        >>> response = safe_get("https://en.wikipedia.org/wiki/Dan_Sullivan")
        >>> if response:
        ...     html = response.text
    """
    default_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    headers = {**default_headers, **(headers or {})}
    
    session = create_session(
        max_retries=max_retries,
        backoff_factor=backoff_factor
    )
    
    for attempt in range(max_retries):
        try:
            response = session.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            msg = f"HTTP {status_code} on attempt {attempt + 1}/{max_retries}: {url}"
            if logger:
                logger.warning(msg)
            else:
                print(f"⚠️  {msg}")
            
            if status_code == 404:
                return None  # Don't retry 404s
            
            if attempt < max_retries - 1:
                wait_time = backoff_factor ** attempt
                time.sleep(wait_time)
        
        except requests.exceptions.Timeout:
            msg = f"Timeout on attempt {attempt + 1}/{max_retries}: {url}"
            if logger:
                logger.warning(msg)
            else:
                print(f"⚠️  {msg}")
            
            if attempt < max_retries - 1:
                wait_time = backoff_factor ** attempt
                time.sleep(wait_time)
        
        except requests.exceptions.RequestException as e:
            msg = f"Request failed on attempt {attempt + 1}/{max_retries}: {url} ({e})"
            if logger:
                logger.warning(msg)
            else:
                print(f"⚠️  {msg}")
            
            if attempt < max_retries - 1:
                wait_time = backoff_factor ** attempt
                time.sleep(wait_time)
    
    return None  # All retries exhausted


# ─────────────────────────────────────────────────────────────────────────────
# LLM API CALLS WITH RETRY
# ─────────────────────────────────────────────────────────────────────────────

def is_quota_error(error_str: str) -> bool:
    """
    Check if an exception string indicates a quota/rate limit error.
    
    Args:
        error_str: Error message string
        
    Returns:
        True if likely a quota/rate limit error
    """
    quota_indicators = [
        "RESOURCE_EXHAUSTED",
        "quota",
        "rate_limit",
        "rate limit",
        "too many requests",
        "please retry",
        "429",  # HTTP 429 Too Many Requests
        "503",  # HTTP 503 Service Unavailable
        "503",  # HTTP 502 Bad Gateway
    ]
    return any(indicator.lower() in error_str.lower() for indicator in quota_indicators)


def extract_retry_delay(error_str: str) -> Optional[float]:
    """
    Extract retry delay from error message if available.
    
    Args:
        error_str: Error message string
        
    Returns:
        Delay in seconds, or None if not found
        
    Example:
        >>> extract_retry_delay("Error: retry in 3.5 seconds")
        3.5
    """
    match = re.search(r"retry\s+in\s+(\d+(?:\.\d+)?)\s*s(?:ec)?", error_str, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 2.0  # Add 2s buffer
    return None


def call_llm_with_retry(
    api_call_fn,
    prompt: str,
    text: str,
    max_retries: int = 5,
    base_backoff: float = 1.0,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Call an LLM API with exponential backoff retry on quota errors.
    
    Args:
        api_call_fn: Callable that makes the API call and returns parsed response
                     Signature: (prompt, text) -> Dict
        prompt: Prompt template for extraction
        text: Input text to extract from
        max_retries: Maximum retry attempts
        base_backoff: Base backoff in seconds (multiplied by 2^attempt)
        logger: Optional logger for warnings
        
    Returns:
        Dict with "result" key on success, "error" key on failure
        
    Example:
        >>> def my_groq_call(prompt, text):
        ...     response = client.chat.completions.create(...)
        ...     return json.loads(response.choices[0].message.content)
        >>> result = call_llm_with_retry(my_groq_call, PROMPT, text)
    """
    for attempt in range(max_retries):
        try:
            result = api_call_fn(prompt, text)
            return {"result": result, "attempt": attempt + 1}
        
        except Exception as e:
            error_str = str(e)
            
            if is_quota_error(error_str) and attempt < max_retries - 1:
                # Extract suggested delay or use exponential backoff
                delay = extract_retry_delay(error_str)
                if not delay:
                    delay = min(base_backoff * (2 ** attempt), 120)  # Cap at 120s
                
                msg = f"⏳ Rate limit (attempt {attempt + 1}/{max_retries}). Waiting {delay:.1f}s..."
                if logger:
                    logger.warning(msg)
                else:
                    print(msg)
                
                time.sleep(delay)
                continue
            else:
                # Either not a quota error or out of retries
                return {
                    "error": error_str,
                    "attempt": attempt + 1,
                    "is_quota": is_quota_error(error_str),
                }
    
    return {"error": f"Failed after {max_retries} retries", "attempt": max_retries}


# ─────────────────────────────────────────────────────────────────────────────
# JSON RESPONSE PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_json_response(
    response_text: str,
    remove_markdown_fences: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict]:
    """
    Parse JSON from LLM response, handling markdown code fences.
    
    Args:
        response_text: Raw response from LLM
        remove_markdown_fences: If True, strip ```json ... ``` wrappers
        logger: Optional logger for errors
        
    Returns:
        Parsed dict on success, None on failure
        
    Example:
        >>> response = "```json\n{\"name\": \"John\"}\n```"
        >>> parse_json_response(response)
        {"name": "John"}
    """
    text = response_text.strip()
    
    # Remove markdown code fences
    if remove_markdown_fences:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1].strip()
    
    # Parse JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        if logger:
            logger.error(f"JSON parse error: {e}\nText: {text[:200]}")
        else:
            print(f"❌ JSON parse error: {e}")
        return None
