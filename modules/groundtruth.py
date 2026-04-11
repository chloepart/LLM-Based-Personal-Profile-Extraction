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


# ---------------------------------------------------------------------------
# Birthdate normalization
# ---------------------------------------------------------------------------

def normalize_birthdate(date_str):
    """
    Normalize birthdate to YYYY-MM-DD format.

    Handles:
      - "Month DD, YYYY" / "Month DD YYYY"
      - "DD Month YYYY"
      - "YYYY-MM-DD" (already normalized — returned as-is)

    Returns str in YYYY-MM-DD, or None if parsing fails.
    """
    if not date_str or not isinstance(date_str, str):
        return None

    date_str = re.sub(r'[()[\]]', '', date_str).strip()

    # Already normalized
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str

    # FIX: single strptime loop — no redundant regex pre-pass
    for fmt in [
        '%B %d, %Y', '%b %d, %Y',
        '%B %d %Y',  '%b %d %Y',
        '%d %B %Y',  '%d %b %Y',
    ]:
        try:
            return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue

    logging.debug(f"normalize_birthdate: could not parse '{date_str}'")
    return None


# ---------------------------------------------------------------------------
# Wikipedia scraping
# FIX: removed fetch_wikipedia_text (dead code — scrape_wikipedia never called it)
# ---------------------------------------------------------------------------

def scrape_wikipedia(url, name):
    """
    Scrape Wikipedia for a senator's profile.

    Returns dict with keys: full_name, birthdate, gender, race_ethnicity.
    Falls back to pronoun heuristic for gender if infobox is absent.
    All fields default to None on failure.
    """
    result = {
        "full_name": None,
        "birthdate": None,
        "gender": None,
        "race_ethnicity": None,
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    # FIX: consolidated fallback chain — try URL as-is, then constructed URL from name
    urls_to_try = [url, "https://en.wikipedia.org/wiki/" + name.replace(" ", "_")]
    # deduplicate while preserving order
    seen = set()
    urls_to_try = [u for u in urls_to_try if not (u in seen or seen.add(u))]

    soup = None
    for attempt_url in urls_to_try:
        try:
            response = requests.get(attempt_url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            break
        except requests.exceptions.RequestException as e:
            logging.warning(f"Wikipedia fetch failed for {name} at {attempt_url}: {e}")

    if soup is None:
        logging.warning(f"All Wikipedia fetch attempts exhausted for {name}")
        return result

    try:
        # Full name from h1
        h1_tag = soup.find("h1")
        if h1_tag:
            result["full_name"] = re.sub(r'\s*\(.*?\)\s*$', '', h1_tag.get_text(strip=True)).strip()

        # Infobox
        infobox = soup.find("table", {"class": lambda x: x and "infobox" in x})
        if infobox:
            result = _extract_infobox_data(infobox, name, result)

        # Birthdate fallback — scan paragraphs
        if not result["birthdate"]:
            for p in soup.find_all("p"):
                text = p.get_text()
                birth_match = re.search(
                    r'born\s+(?:\()?(\w+\s+\d{1,2},?\s+\d{4})(?:\))?',
                    text, re.IGNORECASE
                )
                if birth_match:
                    result["birthdate"] = normalize_birthdate(birth_match.group(1))
                    break
                year_match = re.search(r'born\s+(?:in\s+)?(\d{4})', text, re.IGNORECASE)
                if year_match:
                    result["birthdate"] = year_match.group(1)
                    break

        # Gender fallback — pronoun heuristic
        if not result["gender"]:
            for p in soup.find_all("p"):
                text = p.get_text()
                if len(text) > 50:
                    text_lower = text.lower()
                    if any(tok in text_lower for tok in [" she ", " her ", " hers "]):
                        result["gender"] = "Female"
                    elif any(tok in text_lower for tok in [" he ", " him ", " his "]):
                        result["gender"] = "Male"
                    if result["gender"]:
                        break

    except Exception as e:
        logging.warning(f"Wikipedia parse failed for {name}: {e}")

    return result


def _extract_infobox_data(infobox_table, name, result):
    """
    Extract birthdate, gender, and race/ethnicity from a Wikipedia infobox.
    """
    # Structured bday span — most reliable path
    bday = infobox_table.find("span", {"class": "bday"})
    if bday:
        result["birthdate"] = bday.get_text(strip=True)  # already YYYY-MM-DD

    for row in infobox_table.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) < 2:
            continue

        label = cells[0].get_text(strip=True).lower()
        value = cells[1].get_text(strip=True)

        if not value:
            continue

        # Birthdate
        if not result["birthdate"] and any(
            p in label for p in ["born", "birth date", "birthdate", "date of birth", "b."]
        ):
            for pattern in [
                r'(\w+\s+\d{1,2},?\s+\d{4})',
                r'(\d{1,2}\s+\w+\s+\d{4})',
                r'(\d{4}-\d{2}-\d{2})',
            ]:
                m = re.search(pattern, value)
                if m:
                    result["birthdate"] = normalize_birthdate(m.group(1))
                    break

        # Gender
        if any(p in label for p in ["gender", "sex"]):
            v = value.lower()
            if "female" in v or "woman" in v:
                result["gender"] = "Female"
            elif "male" in v or "man" in v:
                result["gender"] = "Male"

        # Race / ethnicity
        if any(p in label for p in ["ethnicity", "race", "ethnic", "ancestry"]):
            clean = re.sub(r'\[.*?\]', '', value).strip()
            if clean and len(clean) < 200:
                result["race_ethnicity"] = clean

    return result


# ---------------------------------------------------------------------------
# Ballotpedia committee scraping
# ---------------------------------------------------------------------------

def _extract_committees_from_soup(soup):
    """
    Extract 2025-2026 committee assignments from a parsed Ballotpedia page.

    Returns dict: {committees, session_year_found, status}
    """
    result = {
        "committees": [],
        "session_year_found": None,
        "status": "unknown",
    }

    committee_heading = None
    for h in soup.find_all("h2"):
        if "committee assignments" in h.get_text().lower():
            committee_heading = h
            break

    if not committee_heading:
        result["status"] = "no_committee_heading_found"
        return result

    current = committee_heading.find_next("h4")
    while current:
        if "2025-2026" in current.get_text():
            result["session_year_found"] = "2025-2026"
            sib = current.find_next()
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


def test_committee_scraping(url, name, verbose=True):
    """
    Debug helper: test committee extraction from a single Ballotpedia URL.
    """
    result = {
        "url": url,
        "name": name,
        "status": "unknown",
        "committee_count": 0,
        "committees": [],
        "session_year_found": None,
    }

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        extraction = _extract_committees_from_soup(soup)

        result.update({
            "status": extraction["status"],
            "session_year_found": extraction["session_year_found"],
            "committees": extraction["committees"],
            "committee_count": len(extraction["committees"]),
        })

        if verbose:
            status = result["status"]
            if status == "no_committee_heading_found":
                print(f"✗ {name}: No 'Committee assignments' heading found")
            elif status == "session_year_2025-2026_not_found":
                print(f"⚠️  {name}: Found Committee assignments but no 2025-2026 session year")
            elif status == "success":
                print(f"✓ {name} — {result['committee_count']} committees")
                for c in result["committees"][:3]:
                    print(f"    • {c[:80]}")
                if result["committee_count"] > 3:
                    print(f"    ... and {result['committee_count'] - 3} more")
            elif status == "session_found_but_no_committees":
                print(f"⚠️  {name}: Session year found but no committee list items")

    except Exception as e:
        result["status"] = f"error: {e}"
        if verbose:
            print(f"✗ {name}: {e}")

    return result


# ---------------------------------------------------------------------------
# Committee YAML loading and resolution
# ---------------------------------------------------------------------------

# Module-level constants for committee name mapping and URL overrides
COMMITTEE_NAME_ALIASES = {
    "Angus King":    "Angus S. King, Jr.",
    "Bernie Sanders": "Bernard Sanders",
    "Chuck Schumer":  "Charles E. Schumer",
    "Chris Coons":    "Christopher A. Coons",
    "Chris Murphy":   "Christopher Murphy",
    "Ed Markey":      "Edward J. Markey",
    "Jim Justice":    "James C. Justice",
    "Jim Risch":      "James E. Risch",
    "Katie Britt":    "Katie Boyd Britt",
    "Maggie Hassan":  "Margaret Wood Hassan",
    "Dick Durbin":    "Richard J. Durbin",
}

NO_COMMITTEE_ENTRY = {"Alan Armstrong"}

WIKI_URL_OVERRIDES = {
    "Dan Sullivan": "https://en.wikipedia.org/wiki/Dan_Sullivan_(U.S._senator)",
    "Jack Reed":    "https://en.wikipedia.org/wiki/Jack_Reed_(Rhode_Island_politician)",
}


def load_committees_from_yaml(yaml_path, senator_names=None):
    """
    Load committee membership YAML and return a name -> pipe-delimited ID string map.
    Fuzzy-matches against senator_names if provided.
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    senator_committees = {}
    for committee_id, members in data.items():
        if not isinstance(members, list):
            continue
        for member in members:
            name = member.get("name")
            if not name:
                continue
            entry = committee_id
            if member.get("title"):
                entry = f"{committee_id} ({member['title']})"
            senator_committees.setdefault(name, []).append(entry)

    raw_map = {name: "|".join(c) for name, c in senator_committees.items()}

    if not senator_names:
        return raw_map

    yaml_names = list(raw_map.keys())
    matched_map = {}
    for name in senator_names:
        if name in NO_COMMITTEE_ENTRY:
            matched_map[name] = None
            continue

        lookup_name = COMMITTEE_NAME_ALIASES.get(name, name)  # apply alias
        result = process.extractOne(lookup_name, yaml_names, scorer=fuzz.token_sort_ratio)
        if result and result[1] >= 85:
            matched_map[name] = raw_map[result[0]]
        else:
            logging.warning(f"No committee match for {name} (best score: {result[1] if result else 0})")
            matched_map[name] = None

    return matched_map


def build_committee_lookup(committees_yaml_path):
    """
    Build thomas_id -> full name lookup from committees-current.yaml.
    Covers full committees and subcommittees.
    """
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
    Convert pipe-delimited committee IDs to readable names using the lookup.
    e.g. "SSEG|SSEG01 (Ranking Member)" -> "Senate Committee on Energy...|Energy (Ranking Member)"
    """
    if not committee_roles_str:
        return None

    resolved = []
    for entry in committee_roles_str.split("|"):
        parts = entry.split(" (", 1)
        code = parts[0].strip()
        role = f" ({parts[1]}" if len(parts) > 1 else ""
        resolved.append(lookup.get(code, code) + role)

    return "|".join(resolved)


# ---------------------------------------------------------------------------
# Pew religion merge
# ---------------------------------------------------------------------------

# FIX: "Don't know/refused" mapped to None so it scores as NaN (unscored),
# not as an incorrect prediction against any LLM output.
_PEW_NULL_VALUES = {"Don't know/refused", "don't know/refused", "Refused"}

def merge_pew(senators_df, pew_path):
    """
    Merge Pew religion data using fuzzy name matching.

    Args:
        senators_df: DataFrame with column 'name'
        pew_path: Path to Pew CSV with columns: name, state, religion

    Returns:
        Series of religion values aligned to senators_df index.
        "Don't know/refused" Pew values are coerced to None.
        Returns all None if pew_path doesn't exist.
    """
    religion_series = pd.Series(index=senators_df.index, dtype=object)
    religion_series[:] = None

    # Senators appointed after Pew data collection — no valid entry exists
    NO_PEW_ENTRY = {
        "Alan Armstrong",
        "Ashley Moody",
        "Jon Husted",
    }

    # Pew uses formal names; bios often use nicknames
    NAME_ALIASES = {
        "Chuck Grassley": "Charles Grassley",
        "Chuck Schumer":  "Charles Schumer",
        "Dick Durbin":    "Richard Durbin",
        "Ed Markey":      "Edward Markey",
        "Chris Murphy":   "Christopher Murphy",
        "Mike Crapo":     "Michael Crapo",
    }

    if not os.path.exists(pew_path):
        logging.warning(f"Pew religion CSV not found at {pew_path} — skipping merge")
        print(f"⚠️  Pew religion file not found: {pew_path}")
        print(f"   Religion column will be empty.")
        return religion_series

    try:
        pew_df = pd.read_csv(pew_path)
        pew_names = pew_df["name"].tolist()

        for idx, row in senators_df.iterrows():
            senator_name = row["name"]

            if senator_name in NO_PEW_ENTRY:
                continue

            lookup_name = NAME_ALIASES.get(senator_name, senator_name)
            best_match, score, _ = process.extractOne(
                lookup_name, pew_names, scorer=fuzz.token_sort_ratio
            )

            if score >= 85:
                raw_religion = pew_df[pew_df["name"] == best_match].iloc[0]["religion"]
                # Coerce survey non-answers to None so they score as NaN, not wrong
                religion_series.loc[idx] = None if raw_religion in _PEW_NULL_VALUES else raw_religion
            else:
                logging.warning(f"Pew match failed for {senator_name} (score={score})")

    except Exception as e:
        logging.warning(f"Error merging Pew data: {e}")
        print(f"⚠️  Error reading Pew CSV: {e}")

    return religion_series


# ---------------------------------------------------------------------------
# Religion signal detection
# ---------------------------------------------------------------------------

# FIX: use a frozenset for O(1) membership testing instead of linear scan
_EXPLICIT_RELIGION_KEYWORDS = frozenset([
    # Place of worship
    "church", "parish", "synagogue", "mosque", "temple", "cathedral",
    "chapel", "congregation", "diocese",
    # Denominations / traditions
    "catholic", "protestant", "baptist", "methodist", "presbyterian",
    "episcopal", "lutheran", "pentecostal", "evangelical", "mormon",
    "lds", "church of jesus christ", "jewish", "judaism", "orthodox",
    "muslim", "islam", "buddhist", "buddhism", "hindu", "hinduism",
    "sikh", "sikhism", "unitarian", "unitarian universalist",
    "quaker", "mennonite", "christian",
    # Religious roles / actions
    "faith", "denomination", "ordained", "reverend", "rabbi", "priest",
    "imam", "pastor", "bishop", "cardinal", "confirmed", "baptized",
    "convert", "conversion", "religious", "devout", "pious", "faithful",
    # Religious education
    "theological", "seminary", "divinity school",
])


def detect_religion_signal(wiki_text, religion):
    """
    Classify whether a biography explicitly mentions religious affiliation.

    Returns:
        "error"        — fetch failed or text missing
        "explicit"     — text contains a direct religious keyword
        "not_explicit" — no explicit keyword found
    """
    if not wiki_text or str(wiki_text).strip() == "" or wiki_text == "error":
        return "error"

    text_lower = str(wiki_text).lower()

    # FIX: set-based check — any() over a generator against a frozenset is O(k)
    # where k = number of keywords, but each `in text_lower` is still a substring
    # search. For the keyword count here this is fine; consider regex alternation
    # if the keyword list grows significantly.
    if any(kw in text_lower for kw in _EXPLICIT_RELIGION_KEYWORDS):
        return "explicit"

    return "not_explicit"
