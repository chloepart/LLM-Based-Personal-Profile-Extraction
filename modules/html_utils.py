"""
HTML processing utilities — extracted from HTMLProcessor static methods.
Provides functions for parsing, cleaning, and extracting data from HTML.
"""

import re
from bs4 import BeautifulSoup


def extract_readable_text(html: str) -> str:
    """
    Extract readable text from HTML, removing script/style tags and excessive whitespace.
    
    Args:
        html: Raw HTML string
        
    Returns:
        Cleaned text string
    """
    try:
        soup = BeautifulSoup(html, 'html.parser')
    except Exception:
        return ""

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Get text
    text = soup.get_text()

    # Break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())

    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

    # Drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text


def extract_infobox(html: str) -> dict:
    """
    Extract infobox data from Wikipedia-style HTML.
    
    Args:
        html: Raw HTML string
        
    Returns:
        Dictionary of infobox fields
    """
    try:
        soup = BeautifulSoup(html, 'html.parser')
    except Exception:
        return {}

    # Find infobox (table with class containing "infobox")
    infobox = soup.find('table', {'class': re.compile('infobox', re.IGNORECASE)})
    if not infobox:
        return {}

    result = {}
    rows = infobox.find_all('tr')
    for row in rows:
        cells = row.find_all(['th', 'td'])
        if len(cells) >= 2:
            key = cells[0].get_text(strip=True)
            val = cells[1].get_text(strip=True)
            if key and val:
                result[key] = val

    return result


def extract_wikipedia_profile(html: str, name: str = "") -> dict:
    """
    Extract biographical profile data from Wikipedia HTML.
    
    Args:
        html: Raw HTML string
        name: Person's name (optional, for logging)
        
    Returns:
        Dictionary with infobox and intro text
    """
    readable_text = extract_readable_text(html)
    infobox_data = extract_infobox(html)
    
    return {
        "name": name,
        "text": readable_text,
        "infobox": infobox_data
    }
