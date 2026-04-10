"""
Ground Truth Builder — Multi-source scraping and data merging.

Functions for building senate_ground_truth.csv by combining data from:
  - Wikipedia (birthdate, gender, race_ethnicity)
  - Ballotpedia (committee_roles)
  - Pew Research (religion via fuzzy matching)
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import yaml
import re
import os
import logging
from rapidfuzz import process, fuzz
from datetime import datetime


def normalize_birthdate(date_str):
    """
    Normalize birthdate to YYYY-MM-DD format.
    
    Handles multiple input formats:
      - "Month DD, YYYY" (e.g., "January 15, 1980")
      - "DD Month YYYY" (e.g., "15 January 1980")
      - "YYYY-MM-DD" (already normalized)
      - "Month DD YYYY" (e.g., "January 15 1980")
      - etc.
    
    Args:
        date_str: Birthdate string in various formats
        
    Returns:
        str: Normalized date in YYYY-MM-DD format, or None if parsing fails
    """
    if not date_str or not isinstance(date_str, str):
        return None
    
    date_str = date_str.strip()
    
    # Already in YYYY-MM-DD format
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str
    
    # Month names for parsing
    month_map = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    try:
        # Remove extra parentheses or brackets
        date_str = re.sub(r'[()[\]]', '', date_str).strip()
        
        # Try parsing with dateutil or manual parsing
        # Pattern 1: "Month DD, YYYY" or "Month DD YYYY"
        match = re.match(r'(\w+)\s+(\d{1,2}),?\s+(\d{4})', date_str)
        if match:
            month_str, day, year = match.groups()
            month = month_map.get(month_str.lower())
            if month:
                return f"{year}-{month:02d}-{int(day):02d}"
        
        # Pattern 2: "DD Month YYYY"
        match = re.match(r'(\d{1,2})\s+(\w+)\s+(\d{4})', date_str)
        if match:
            day, month_str, year = match.groups()
            month = month_map.get(month_str.lower())
            if month:
                return f"{year}-{month:02d}-{int(day):02d}"
        
        # Pattern 3: Use datetime.strptime for other formats
        for fmt in ['%B %d, %Y', '%b %d, %Y', '%B %d %Y', '%b %d %Y', '%d %B %Y', '%d %b %Y']:
            try:
                parsed = datetime.strptime(date_str, fmt)
                return parsed.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # If nothing worked, return None
        return None
        
    except Exception as e:
        logging.debug(f"Error normalizing birthdate '{date_str}': {str(e)}")
        return None


def fetch_wikipedia_text(url, name):
    """
    Fetch and return raw Wikipedia page text (for caching and reuse).
    Uses multiple fallback strategies for better coverage.
    
    Args:
        url: Wikipedia URL for the senator
        name: Senator's name for logging
        
    Returns:
        str: Clean Wikipedia text
        str: "error" on failure
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    
    # Try original URL first
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Remove script, style, nav, footer — keep the actual content
        for tag in soup(["script", "style", "nav", "footer", "noscript"]):
            tag.decompose()
        
        # Extract clean text (first 5000 chars should be sufficient)
        wiki_text = soup.get_text(separator=" ", strip=True)[:5000]
        
        if len(wiki_text) > 100:  # Basic sanity check - should have meaningful content
            return wiki_text
        else:
            logging.warning(f"Wikipedia text too short for {name} ({len(wiki_text)} chars)")
            # Fall through to fallback
        
    except requests.exceptions.RequestException as e:
        logging.warning(f"Initial Wikipedia fetch failed for {name}: {str(e)}")
    except Exception as e:
        logging.warning(f"Error parsing Wikipedia page for {name}: {str(e)}")
    
    # Fallback strategy 1: Try with underscores instead of spaces in URL
    try:
        fallback_url_underscore = url.replace(" ", "_") if " " in url else None
        if fallback_url_underscore:
            response = requests.get(fallback_url_underscore, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "noscript"]):
                tag.decompose()
            wiki_text = soup.get_text(separator=" ", strip=True)[:5000]
            if len(wiki_text) > 100:
                logging.info(f"Successfully fetched {name} using underscore URL")
                return wiki_text
    except Exception as e:
        logging.debug(f"Underscore fallback failed for {name}: {str(e)}")
    
    # Fallback strategy 2: Construct Wikipedia URL from name
    try:
        # Standard Wikipedia URL format
        constructed_url = "https://en.wikipedia.org/wiki/" + name.replace(" ", "_")
        response = requests.get(constructed_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "noscript"]):
            tag.decompose()
        wiki_text = soup.get_text(separator=" ", strip=True)[:5000]
        if len(wiki_text) > 100:
            logging.info(f"Successfully fetched {name} using constructed URL")
            return wiki_text
    except Exception as e:
        logging.debug(f"Constructed URL fallback failed for {name}: {str(e)}")
    
    # All strategies failed
    logging.warning(f"All fetch strategies exhausted for {name}")
    return "error"


def scrape_wikipedia(url, name):
    """
    Scrape Wikipedia for a senator's profile.
    
    Args:
        url: Wikipedia URL for the senator
        name: Senator's name for logging
        
    Returns:
        dict with keys: full_name, birthdate, gender, race_ethnicity
        On failure, all fields are returned as None
    """
    result = {
        "full_name": None,
        "birthdate": None,
        "gender": None,
        "race_ethnicity": None
    }
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extract full_name from page heading (h1)
        h1_tag = soup.find("h1")
        if h1_tag:
            page_title = h1_tag.get_text(strip=True)
            # Remove disambiguation suffix if present
            result["full_name"] = re.sub(r'\s*\(.*?\)\s*$', '', page_title).strip()
        
        # Find infobox (table with class containing "infobox")
        infobox = soup.find("table", {"class": lambda x: x and "infobox" in x})
        
        if infobox:
            # Use a more robust parsing for infobox data
            result = _extract_infobox_data(infobox, name, result)
        
        # If birthdate not found in infobox, search first paragraph for birth date context
        if not result["birthdate"]:
            for p in soup.find_all("p"):
                text = p.get_text()
                # Try full date first
                birth_match = re.search(
                    r'born\s+(?:\()?(\w+\s+\d{1,2},?\s+\d{4})(?:\))?',
                    text, re.IGNORECASE
                )
                if birth_match:
                    result["birthdate"] = normalize_birthdate(birth_match.group(1))
                    break
                # Year-only fallback
                year_match = re.search(r'born\s+(?:in\s+)?(\d{4})', text, re.IGNORECASE)
                if year_match:
                    result["birthdate"] = year_match.group(1)
                    break
        
        # If gender not found in infobox, check first paragraph for pronouns
        if not result["gender"]:
            for p in soup.find_all("p"):
                text = p.get_text()
                if len(text) > 50:
                    text_lower = text.lower()
                    # Look for gender pronouns (be careful with possessives and contractions)
                    if any(p in text_lower for p in [" she ", " her ", " hers "]):
                        result["gender"] = "Female"
                    elif any(p in text_lower for p in [" he ", " him ", " his "]):
                        result["gender"] = "Male"
                    if result["gender"]:
                        break
        
    except Exception as e:
        logging.warning(f"Wikipedia scrape failed for {name}: {str(e)}")
    
    return result


def _extract_infobox_data(infobox_table, name, result):
    """
    Extract data from Wikipedia infobox table using multiple strategies.
    More robust than simple label matching.
    
    Args:
        infobox_table: BeautifulSoup table object (the infobox)
        name: Senator name for logging
        result: Current result dict to update
        
    Returns:
        Updated result dict with extracted values
    """

    bday = infobox_table.find("span", {"class": "bday"})
    if bday:
        result["birthdate"] = bday.get_text(strip=True)  # already YYYY-MM-DD

    rows = infobox_table.find_all("tr")
    
    for row in rows:
        cells = row.find_all(["th", "td"])
        if len(cells) < 2:
            continue
        
        # Get label and value, handling nested tags
        label_cell = cells[0]
        value_cell = cells[1]
        
        label = label_cell.get_text(strip=True).lower()
        value = value_cell.get_text(strip=True)
        
        if not value:
            continue
        
        # ── BIRTHDATE EXTRACTION (multiple patterns) ────────────────────
        # Pattern 1: "Born" or "Birth date" labels
        if any(pattern in label for pattern in ["born", "birth date", "birthdate", "date of birth", "b."]):
            if not result["birthdate"]:
                # Extract first date-like pattern from value
                # Handles: "Month DD, YYYY", "DD Month YYYY", "YYYY-MM-DD", "(Month DD, YYYY)"
                date_patterns = [
                    r'(\w+\s+\d{1,2},?\s+\d{4})',  # Month DD, YYYY or Month DD YYYY
                    r'(\d{1,2}\s+\w+\s+\d{4})',     # DD Month YYYY
                    r'(\d{4}-\d{2}-\d{2})',          # YYYY-MM-DD
                ]
                
                for pattern in date_patterns:
                    date_match = re.search(pattern, value)
                    if date_match:
                        result["birthdate"] = normalize_birthdate(date_match.group(1))
                        break
        
        # ── GENDER EXTRACTION ──────────────────────────────────────────
        # Pattern 1: Explicit "Gender" or "Sex" label
        if any(pattern in label for pattern in ["gender", "sex"]):
            value_lower = value.lower()
            if "female" in value_lower or "woman" in value_lower:
                result["gender"] = "Female"
            elif "male" in value_lower or "man" in value_lower:
                result["gender"] = "Male"
        
        # ── RACE/ETHNICITY EXTRACTION ──────────────────────────────────
        # Pattern 1: Explicit "Ethnicity" or "Race" label
        if any(pattern in label for pattern in ["ethnicity", "race", "ethnic", "ancestry"]):
            # Clean up the value (remove citations, extra whitespace)
            clean_value = re.sub(r'\[.*?\]', '', value).strip()
            if clean_value and len(clean_value) < 200:  # Sanity check
                result["race_ethnicity"] = clean_value
    
    return result


# Validation function removed — V1 scraper uses session-year targeting (2025-2026) instead


def _extract_committees_from_soup(soup):
    """
    Extract committee assignments from a BeautifulSoup object.
    Finds "Committee assignments" heading, traverses to "2025-2026" session year,
    and collects all committee list items.
    
    Args:
        soup: BeautifulSoup object of parsed Ballotpedia page
        
    Returns:
        dict with keys:
            - committees: list of committee strings
            - session_year_found: str ("2025-2026") or None
            - status: str describing extraction result
    """
    result = {
        "committees": [],
        "session_year_found": None,
        "status": "unknown"
    }
    
    # Find "Committee assignments" h2
    committee_heading = None
    for h in soup.find_all("h2"):
        if "committee assignments" in h.get_text().lower():
            committee_heading = h
            break
    
    if not committee_heading:
        result["status"] = "no_committee_heading_found"
        return result
    
    # Look for session year heading (h4)
    current = committee_heading.find_next("h4")
    while current:
        heading_text = current.get_text()
        if "2025-2026" in heading_text:
            result["session_year_found"] = "2025-2026"
            sib = current.find_next()
            # Extract all list items until next h4/h3/h2
            while sib and sib.name not in ["h4", "h3", "h2"]:
                if sib.name == "li":
                    text = sib.get_text(strip=True)
                    if text:
                        result["committees"].append(text)
                sib = sib.find_next()
            break
        current = current.find_next("h4")
    
    if not result["session_year_found"]:
        result["status"] = "session_year_2025-2026_not_found"
    else:
        result["status"] = "success" if result["committees"] else "session_found_but_no_committees"
    
    return result


def load_committees_from_yaml(yaml_path, senator_names=None):
    """
    senator_names: optional list of names from your CSV to fuzzy-match against
    """
    import yaml
    from rapidfuzz import process, fuzz

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # Build raw name -> pipe-delimited committees
    senator_committees = {}
    for committee_id, members in data.items():
        if not isinstance(members, list):
            continue
        for member in members:
            name = member.get("name")
            if not name:
                continue
            if name not in senator_committees:
                senator_committees[name] = []
            entry = committee_id
            if member.get("title"):
                entry = f"{committee_id} ({member['title']})"
            senator_committees[name].append(entry)

    raw_map = {name: "|".join(c) for name, c in senator_committees.items()}

    # If no senator_names provided, return raw map
    if not senator_names:
        return raw_map

    # Fuzzy match your CSV names against YAML names
    yaml_names = list(raw_map.keys())
    matched_map = {}
    for name in senator_names:
        result = process.extractOne(name, yaml_names, scorer=fuzz.token_sort_ratio)
        if result and result[1] >= 85:
            matched_map[name] = raw_map[result[0]]
        else:
            logging.warning(f"No committee match for {name} (best score: {result[1] if result else 0})")
            matched_map[name] = None

    return matched_map


def build_committee_lookup(committees_yaml_path):
    """
    Build a thomas_id -> full name lookup from committees-current.yaml.
    Covers both full committees and subcommittees.
    """
    import yaml
    with open(committees_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    lookup = {}
    for committee in data:
        tid = committee.get("thomas_id")
        name = committee.get("name")
        if tid and name:
            lookup[tid] = name
        for sub in committee.get("subcommittees", []):
            sub_id = tid + sub["thomas_id"].zfill(2)
            lookup[sub_id] = sub["name"]

    return lookup


def resolve_committee_roles(committee_roles_str, lookup):
    """
    Convert pipe-delimited committee IDs to readable names.
    e.g. "SSEG|SSEG01 (Ranking Member)" -> "Senate Committee on Energy and Natural Resources|Energy (Ranking Member)"
    """
    if not committee_roles_str:
        return None

    resolved = []
    for entry in committee_roles_str.split("|"):
        parts = entry.split(" (", 1)
        code = parts[0].strip()
        role = f" ({parts[1]}" if len(parts) > 1 else ""
        name = lookup.get(code, code)  # fall back to raw code if not found
        resolved.append(name + role)

    return "|".join(resolved)

def test_committee_scraping(url, name, verbose=True):
    """
    Test committee extraction from a single URL (debug helper).
    Uses the same session-year targeting approach as scrape_ballotpedia().
    
    Args:
        url: Ballotpedia senator URL
        name: Senator name
        verbose: Print detailed extraction info
        
    Returns:
        dict with extraction results and debug info
    """
    result = {
        "url": url,
        "name": name,
        "status": "unknown",
        "committee_count": 0,
        "committees": [],
        "session_year_found": None
    }
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        extraction_result = _extract_committees_from_soup(soup)
        
        # Merge extraction results into debug dict
        result["status"] = extraction_result["status"]
        result["session_year_found"] = extraction_result["session_year_found"]
        result["committees"] = extraction_result["committees"]
        result["committee_count"] = len(extraction_result["committees"])
        
        # Print verbose output if requested
        if verbose:
            if result["status"] == "no_committee_heading_found":
                print(f"✗ {name}: No 'Committee assignments' heading found")
            elif result["status"] == "session_year_2025-2026_not_found":
                print(f"⚠️  {name}: Found Committee assignments but no 2025-2026 session year")
            elif result["status"] == "success":
                print(f"✓ {name}")
                print(f"  Session year: {result['session_year_found']}")
                print(f"  Found {result['committee_count']} committees")
                if result['committee_count'] <= 5:
                    for c in result["committees"]:
                        print(f"    • {c[:80]}")
                else:
                    print(f"    • {result['committees'][0][:80]}")
                    print(f"    ... and {result['committee_count']-1} more")
            elif result["status"] == "session_found_but_no_committees":
                print(f"⚠️  {name}: Session year found but no committee list items")
        
    except Exception as e:
        result["status"] = f"error: {str(e)}"
        if verbose:
            print(f"✗ {name}: {str(e)}")
    
    return result


def merge_pew(senators_df, pew_path):
    """
    Merge Pew religion data using fuzzy name matching.

    Args:
        senators_df: DataFrame with column 'name'
        pew_path: Path to Pew CSV with columns: name, state, religion

    Returns:
        Series of religion values aligned to senators_df index
        Returns all None if pew_path file doesn't exist
    """
    religion_series = pd.Series(index=senators_df.index, dtype=object)
    religion_series[:] = None

    # ── Senators with no Pew entry (appointed after Pew data was collected) ──
    # These will never match correctly — force null rather than accept bad match.
    NO_PEW_ENTRY = {
        "Alan Armstrong",   # appointed March 2026
        "Ashley Moody",     # appointed January 2025
        "Jon Husted",       # appointed January 2025
    }

    # ── Nickname → Pew formal name mapping ───────────────────────────────────
    # Pew uses formal/legal names; senate bios often use nicknames.
    NAME_ALIASES = {
        "Chuck Grassley":  "Charles Grassley",
        "Chuck Schumer":   "Charles Schumer",
        "Dick Durbin":     "Richard Durbin",
        "Ed Markey":       "Edward Markey",
        "Chris Murphy":    "Christopher Murphy",
        "Mike Crapo":      "Michael Crapo",
    }

    # Check if Pew CSV exists
    if not os.path.exists(pew_path):
        logging.warning(f"Pew religion CSV not found at {pew_path} — skipping merge")
        print(f"⚠️  Pew religion file not found: {pew_path}")
        print(f"   Religion column will be empty. To populate:")
        print(f"   1. Create {pew_path} with columns: name, state, religion")
        print(f"   2. Re-run this cell\n")
        return religion_series

    try:
        pew_df = pd.read_csv(pew_path)
        pew_names = pew_df["name"].tolist()

        for idx, row in senators_df.iterrows():
            senator_name = row["name"]

            # Skip senators with no Pew entry
            if senator_name in NO_PEW_ENTRY:
                logging.info(f"No Pew entry for {senator_name} — skipping")
                continue

            # Apply alias if nickname is used
            lookup_name = NAME_ALIASES.get(senator_name, senator_name)

            # Fuzzy match against Pew names
            best_match, score, _ = process.extractOne(
                lookup_name, pew_names, scorer=fuzz.token_sort_ratio
            )

            if score >= 85:
                pew_match = pew_df[pew_df["name"] == best_match].iloc[0]
                religion_series.loc[idx] = pew_match["religion"]
            else:
                logging.warning(f"Pew match failed for {senator_name} (score={score})")

    except Exception as e:
        logging.warning(f"Error merging Pew data: {str(e)}")
        print(f"⚠️  Error reading Pew CSV: {str(e)}\n")

    return religion_series


def detect_religion_signal(wiki_text, religion):
    """
    Classify whether a biography explicitly mentions religious affiliation.
    
    Used to stratify evaluation metrics in Liu et al. Section 6.1.4.
    
    Args:
        wiki_text (str): Wikipedia biography text
        religion (str): Religion from Pew data (ground truth)
        
    Returns:
        str: One of "explicit", "not_explicit", or "error"
        
    Logic:
        - "error": wiki_text is "error" or missing (fetch failed)
        - "explicit": Wiki text contains direct religious keywords like:
          "church", "parish", "attends", "denomination", "baptist", "catholic",
          "methodist", "presbyterian", "jewish", "muslim", "buddhist", etc.
        - "not_explicit": No explicit religious keywords found in text
    """
    # Handle error states
    if not wiki_text or str(wiki_text).strip() == "" or wiki_text == "error":
        return "error"
    
    text_lower = str(wiki_text).lower()
    
    # Explicit religion keywords — direct mentions in biographical text
    explicit_keywords = [
        # Church / place of worship
        "church", "parish", "synagogue", "mosque", "temple", "cathedral",
        "chapel", "congregation", "diocese",
        
        # Religious practices / beliefs
        "catholic", "protestant", "baptist", "methodist", "presbyterian",
        "episcopal", "lutheran", "pentecostal", "evangelical", "mormon",
        "lds", "church of jesus christ", "jewish", "judaism", "orthodox",
        "jewish orthodox", "muslim", "islam", "buddhist", "buddhism",
        "hindu", "hinduism", "sikh", "sikhism", "unitarian",
        "unitarian universalist", "quaker", "mennonite",
        
        # Religious actions / associations
        "faith", "denomination", "ordained", "reverend", "rabbi", "priest", 
        "imam", "pastor", "bishop", "cardinal", "confirmed", "baptized", 
        "convert", "conversion", "religious", "devout", "pious", "faithful",
        
        # Religious education / organization
        "theological", "seminary", "divinity school",
        "christian", "christian denomination",
    ]
    
    # Check if any explicit keyword appears in the text
    for keyword in explicit_keywords:
        if keyword in text_lower:
            return "explicit"
    
    # No explicit keywords found
    return "not_explicit"
