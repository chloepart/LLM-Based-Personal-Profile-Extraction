"""
Test script for education extraction from Wikipedia infobox.

This script extracts senator education from Wikipedia infobox data and displays
results for manual validation before integrating into groundtruth.py.

Usage:
    python scripts/validate_education_extraction.py

Workflow:
    1. Select sample senators from external_data/senate_html/
    2. Load their HTML and parse infobox
    3. Extract education fields (degree, institution, year)
    4. Display Wikipedia link + extracted education for manual review
    5. Iterate based on user feedback
"""

import os
import re
import json
from bs4 import BeautifulSoup
from pathlib import Path
import sys
import requests
import time
import logging

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.utils.parsing import DegreeNormalizer, SchoolNormalizer

# Configure logging
logging.basicConfig(level=logging.WARNING)


def extract_education_from_infobox(infobox_table):
    """
    Extract education from a Wikipedia infobox table.
    
    Returns:
        List of dicts with keys: degree, institution, year
        Example: [{"degree": "B.S.", "institution": "MIT", "year": 2005}]
    """
    education_items = []
    
    if not infobox_table:
        return education_items
    
    # Education field label variants
    education_labels = [
        "education", "alma mater", "educated at"
    ]
    
    # Labels to explicitly skip
    skip_labels = [
        "committee", "position", "profession", "occupation"
    ]
    
    for row in infobox_table.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) < 2:
            continue
        
        label = cells[0].get_text(strip=True).lower()
        value_cell = cells[1]
        
        # Check if this is an education row
        is_education = any(edu_label in label for edu_label in education_labels)
        if not is_education:
            continue
        
        # Skip rows that are clearly not education
        if any(skip in label for skip in skip_labels):
            continue
        
        # Check if value_cell contains a list (ul/ol with li elements)
        list_items = value_cell.find_all("li")
        
        if list_items:
            # Process each list item as a separate education entry
            for li in list_items:
                entry = parse_education_list_item(li)
                if entry and entry.get("institution"):
                    education_items.append(entry)
        else:
            # No list structure; check for <br> separators
            br_tags = value_cell.find_all("br")
            
            if br_tags:
                # Multiple entries separated by <br> tags
                # Split by <br> manually
                entries_html = []
                current = ""
                
                for child in value_cell.children:
                    if isinstance(child, str):
                        current += str(child).strip()
                    elif hasattr(child, 'name'):
                        if child.name == "br":
                            if current.strip():
                                entries_html.append(current)
                            current = ""
                        else:
                            # Collect text from nested tags
                            current += child.get_text()
                
                # Add the last entry
                if current.strip():
                    entries_html.append(current)
                
                # Parse each entry separately
                for entry_text in entries_html:
                    entry_text = entry_text.strip()
                    if entry_text:
                        line_entries = parse_education_value(None, entry_text)
                        education_items.extend(line_entries)
            else:
                # No <br> tags, just extract text normally
                value_text = value_cell.get_text(strip=True)
                
                if not value_text:
                    continue
                
                # Parse the value cell for multiple entries
                entries = parse_education_value(value_cell, value_text)
                education_items.extend(entries)
    
    return education_items


def parse_education_list_item(li_element):
    """
    Parse a single <li> element from an education list.
    
    Example structure:
    <li><a href="/wiki/University_of_Louisville">University of Louisville</a> (<a href="/wiki/Bachelor_of_Arts">BA</a>)</li>
    
    Returns:
        Dict with keys: degree, institution, year
    """
    result = {
        "degree": None,
        "institution": None,
        "year": None,
    }
    
    text = li_element.get_text(strip=True)
    
    # Extract year if present
    year_match = re.search(r'\(?(\d{4})\)?', text)
    if year_match:
        result["year"] = int(year_match.group(1))
    
    # Look for degree patterns in parentheses or after institution
    # Common patterns: (BA), (J.D.), etc.
    degree_patterns = [
        (r'\(B\.S\.?\)', "B.S."),
        (r'\(B\.A\.?\)', "B.A."),
        (r'\(BS\)', "B.S."),
        (r'\(BA\)', "B.A."),
        (r'\(M\.S\.?\)', "M.S."),
        (r'\(M\.A\.?\)', "M.A."),
        (r'\(MS\)', "M.S."),
        (r'\(MA\)', "M.A."),
        (r'\(Ph\.?D\.?\)', "Ph.D."),
        (r'\(PhD\)', "Ph.D."),
        (r'\(J\.?D\.?\)', "J.D."),
        (r'\(JD\)', "J.D."),
        (r'\(M\.?B\.?A\.?\)', "M.B.A."),
        (r'\(MBA\)', "M.B.A."),
        (r'\(M\.?D\.?\)', "M.D."),
        (r'\(MD\)', "M.D."),
        (r'\(L\.?L\.?B\.?\)', "L.L.B."),
        (r'\(LLB\)', "L.L.B."),
    ]
    
    for pattern, normalized in degree_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["degree"] = normalized
            # Remove degree from text for institution extraction
            text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
            break
    
    # What remains should be the institution
    # Remove parentheses and extra whitespace
    institution = re.sub(r'[()]+', '', text).strip()
    institution = re.sub(r'\s+', ' ', institution)
    
    if institution and len(institution) > 1:
        result["institution"] = institution
    
    # Normalize
    if result["degree"]:
        result["degree"] = DegreeNormalizer.normalize(result["degree"])
    
    if result["institution"]:
        result["institution"] = SchoolNormalizer.normalize(result["institution"])
    
    return result


def parse_education_value(value_cell, value_text):
    """
    Parse education value cell (fallback for non-list structures).
    
    Handles:
    - Multi-line entries separated by newlines
    - Degrees in parentheses: "MIT (B.S.)", "Harvard (BA, JD)"
    - Entries without degrees: "George Washington University"
    - Multiple degrees from same institution: "MIT (B.S., M.S.)"
    
    Returns:
        List of dicts with degree, institution, year
    """
    entries = []
    
    # First, ALWAYS split by newlines if there are multiple lines
    # This handles cases like:
    # George Washington University
    # University of Houston (BS)
    # Rutgers University (JD)
    lines = value_text.split('\n')
    
    if len(lines) > 1:
        # Multi-line entry - process each line separately
        for line in lines:
            line = line.strip()
            if not line or len(line) < 2:
                continue
            
            # Try parentheses format first for this line
            entries_from_line = parse_education_with_parens(line)
            if entries_from_line:
                entries.extend(entries_from_line)
            else:
                # Try parsing as plain institution name (no degree)
                parsed = parse_single_education_entry(line)
                if parsed and parsed.get("institution"):
                    entries.append(parsed)
    else:
        # Single line or no newlines - try parentheses pattern on full text
        entries_from_parens = parse_education_with_parens(value_text)
        if entries_from_parens:
            return entries_from_parens
        
        # Otherwise, try parsing as single entry
        parsed = parse_single_education_entry(value_text)
        if parsed and parsed.get("institution"):
            entries.append(parsed)
    
    return entries


def parse_education_with_parens(text):
    """
    Parse education text with degrees in parentheses.
    
    Examples:
    - "MIT (B.S.)" → [{degree: "B.S.", institution: "MIT", year: null}]
    - "MIT (B.S., M.S.)" → [{degree: "B.S.", ..}, {degree: "M.S.", ..}]
    - "Harvard (BA), Yale (JD)" → [{degree: "BA", institution: "Harvard"}, {degree: "JD", institution: "Yale"}]
    - Multi-line with one without degree:
      "George Washington University\nUniversity of Houston (BS)"
      → [{degree: null, institution: "George Washington", ..}, {degree: "BS", institution: "University of Houston", ..}]
    
    Returns:
        List of parsed entries, or empty if pattern doesn't match
    """
    entries = []
    
    # If text contains newlines, process line by line
    if '\n' in text:
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Recursively parse each line
            line_entries = parse_education_with_parens(line)
            entries.extend(line_entries)
        return entries
    
    # Pattern: Text (degrees/years)
    # Match institution in parentheses with degree info
    pattern = r'([^()]+)\s*\(([^)]+)\)'
    
    matches = list(re.finditer(pattern, text))
    
    if not matches:
        return []
    
    # If we have matches, process each one
    for match in matches:
        institution_text = match.group(1).strip()
        paren_content = match.group(2).strip()
        
        # Extract degree patterns from the parenthetical content
        degree_pattern = r'\b(B\.S\.?|BS|B\.A\.?|BA|M\.S\.?|MS|M\.A\.?|MA|Ph\.D\.?|PhD|J\.D\.?|JD|MBA|M\.B\.A\.|M\.D\.?|MD|LLB|L\.L\.B\.)\b'
        degrees = re.findall(degree_pattern, paren_content, re.IGNORECASE)
        
        # Extract year if present
        year_match = re.search(r'(\d{4})', paren_content)
        year = None
        if year_match:
            year = int(year_match.group(1))
        
        if degrees:
            # Create one entry per degree
            for degree_str in degrees:
                entry = {
                    "degree": DegreeNormalizer.normalize(degree_str),
                    "institution": SchoolNormalizer.normalize(institution_text),
                    "year": year,
                }
                entries.append(entry)
        else:
            # No degree found, just have institution
            entry = {
                "degree": None,
                "institution": SchoolNormalizer.normalize(institution_text),
                "year": year,
            }
            entries.append(entry)
    
    return entries


def split_by_degrees(text):
    """
    Split text by degree keywords to separate multiple education entries.
    
    Example:
    "BA Louisville University Kentucky JD University of Kentucky"
    -> ["BA Louisville University Kentucky", "JD University of Kentucky"]
    """
    degree_patterns = [
        r'\b(B\.S\.|BS|B\.A\.|BA|M\.S\.|MS|M\.A\.|MA|Ph\.D\.|PhD|J\.D\.|JD|MBA|M\.B\.A\.|LLB|L\.L\.B\.|MD|M\.D\.)\b'
    ]
    
    text = text.strip()
    
    # Find all degrees and their positions
    degrees = []
    for pattern in degree_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            degrees.append((match.start(), match.group(0)))
    
    if not degrees:
        # No degrees found, return as single entry
        return [text]
    
    # Sort by position
    degrees = sorted(degrees, key=lambda x: x[0])
    
    if len(degrees) == 1:
        # Single degree, return as-is
        return [text]
    
    # Multiple degrees found - split between them
    entries = []
    for i, (pos, degree) in enumerate(degrees):
        if i < len(degrees) - 1:
            next_pos = degrees[i + 1][0]
            entry = text[pos:next_pos].strip()
        else:
            entry = text[pos:].strip()
        
        if entry:
            entries.append(entry)
    
    return entries


def split_education_entries(text):
    """
    Split education text into individual entries using delimiters.
    
    Handles:
    - Semicolon-separated: "degree1; degree2"
    - Newline-separated
    - Single entry
    """
    # Try splitting by semicolon first (most reliable separation)
    if ";" in text:
        return [e.strip() for e in text.split(";")]
    
    # Try splitting by newline
    if "\n" in text:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(lines) > 1:
            return lines
    
    # Return as single entry
    return [text]


def parse_single_education_entry(entry_text):
    """
    Parse a single education entry into degree, institution, year.
    
    Examples:
    - "B.S. in Computer Science, MIT, 2005" → {degree: "B.S.", institution: "MIT", year: 2005}
    - "Harvard (1995)" → {degree: None, institution: "Harvard", year: 1995}
    - "University of Michigan, M.S., 1990" → {degree: "M.S.", institution: "University of Michigan", year: 1990}
    - "BA Louisville University Kentucky" → {degree: "BA", institution: "Louisville University"}
    
    Returns:
        Dict with keys: degree, institution, year (all can be None except institution)
    """
    entry = entry_text.strip()
    
    result = {
        "degree": None,
        "institution": None,
        "year": None,
    }
    
    # Extract year (4-digit number, typically in parentheses or at end)
    year_match = re.search(r'\(?(\d{4})\)?', entry)
    if year_match:
        result["year"] = int(year_match.group(1))
        # Remove year from entry for further processing
        entry = re.sub(r'\s*\(?(\d{4})\)?\s*', ' ', entry).strip()
    
    # Extract degree (common patterns: B.S., B.A., M.A., J.D., Ph.D., etc.)
    degree_patterns = [
        (r'\b(B\.S\.?)\b', "B.S."),
        (r'\b(B\.A\.?)\b', "B.A."),
        (r'\b(BS)\b', "B.S."),
        (r'\b(BA)\b', "B.A."),
        (r'\b(M\.S\.?)\b', "M.S."),
        (r'\b(M\.A\.?)\b', "M.A."),
        (r'\b(MS)\b', "M.S."),
        (r'\b(MA)\b', "M.A."),
        (r'\b(Ph\.?D\.?|PhD)\b', "Ph.D."),
        (r'\b(J\.?D\.?|JD)\b', "J.D."),
        (r'\b(M\.?B\.?A\.?|MBA)\b', "M.B.A."),
        (r'\b(M\.?D\.?|MD)\b', "M.D."),
        (r'\b(L\.?L\.?B\.?|LLB)\b', "L.L.B."),
    ]
    
    for pattern, normalized in degree_patterns:
        match = re.search(pattern, entry, re.IGNORECASE)
        if match:
            result["degree"] = normalized
            # Remove degree from entry
            entry = re.sub(pattern, '', entry, flags=re.IGNORECASE).strip()
            break
    
    # What remains should be institution(s)
    # Clean up extra spaces and punctuation
    institution = entry.strip()
    institution = re.sub(r'\s+', ' ', institution)  # Collapse multiple spaces
    institution = re.sub(r'[,;\(\)]+', '', institution).strip()  # Remove delimiters
    
    # If institution is too long, it might contain multiple institutions concatenated
    # Try to extract the most likely single institution
    if institution and len(institution) > 50:
        # Look for common university keywords and truncate after them
        # This is a heuristic for cases like "stanford universityuniversity california"
        matches = list(re.finditer(r'university|college|institute|school|academy', institution, re.IGNORECASE))
        if len(matches) > 1:
            # Keep text up to and including the first occurrence of a university keyword
            first_match = matches[0]
            institution = institution[:first_match.end()].strip()
    
    if institution and len(institution) > 1:
        result["institution"] = institution
    
    # Normalize degree and institution
    if result["degree"]:
        result["degree"] = DegreeNormalizer.normalize(result["degree"])
    
    if result["institution"]:
        result["institution"] = SchoolNormalizer.normalize(result["institution"])
    
    return result


def extract_from_html_file(html_path):
    """
    Load HTML file and extract education from infobox.
    
    Returns:
        Dict with keys: senator_name, wikipedia_url, education, status
    """
    result = {
        "senator_name": None,
        "wikipedia_url": None,
        "education": [],
        "status": "unknown",
    }
    
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract senator name from filename
        filename = os.path.basename(html_path)
        senator_name = filename.replace('.html', '')
        result["senator_name"] = senator_name
        
        # Try to extract Wikipedia URL from page (if available)
        # Look for canonical link
        canonical = soup.find("link", {"rel": "canonical"})
        if canonical and canonical.get("href"):
            result["wikipedia_url"] = canonical.get("href")
        else:
            # Fallback: construct URL from name
            result["wikipedia_url"] = f"https://en.wikipedia.org/wiki/{senator_name.replace(' ', '_')}"
        
        # Find infobox
        infobox = soup.find("table", {"class": lambda x: x and "infobox" in x})
        
        if infobox:
            education = extract_education_from_infobox(infobox)
            result["education"] = education
            result["status"] = "success" if education else "no_education_found"
        else:
            result["status"] = "no_infobox_found"
        
    except Exception as e:
        result["status"] = f"error: {str(e)}"
    
    return result


def fetch_and_extract_wikipedia(senator_name):
    """
    Fetch senator's Wikipedia page and extract education from infobox.
    
    Returns:
        Dict with keys: senator_name, wikipedia_url, education, status
    """
    result = {
        "senator_name": senator_name,
        "wikipedia_url": None,
        "education": [],
        "status": "unknown",
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    
    # Try multiple URL formats
    urls_to_try = [
        f"https://en.wikipedia.org/wiki/{senator_name.replace(' ', '_')}",
    ]
    
    soup = None
    used_url = None
    
    for url in urls_to_try:
        try:
            logging.info(f"Fetching: {url}")
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            used_url = url
            result["wikipedia_url"] = url
            break
        except Exception as e:
            logging.warning(f"Failed to fetch {url}: {e}")
            time.sleep(1)  # Be respectful to Wikipedia
    
    if not soup:
        result["status"] = "failed_to_fetch_wikipedia"
        return result
    
    # Find infobox
    infobox = soup.find("table", {"class": lambda x: x and "infobox" in x})
    
    if infobox:
        education = extract_education_from_infobox(infobox)
        result["education"] = education
        result["status"] = "success" if education else "no_education_found"
    else:
        result["status"] = "no_infobox_found"
    
    return result


def select_sample_senators(sample_names=None):
    """
    Get sample senators to test. Can be provided or defaults to common senators.
    
    Returns:
        List of senator names
    """
    if sample_names:
        return sample_names
    
    # Default diverse sample
    defaults = [
        "Bernie Sanders",
        "Mitch McConnell",
        "Kamala Harris",  # Former senator
        "JD Vance",
        "Bill Cassidy",
        "Elizabeth Warren",
        "Chuck Schumer",
        "Lindsey Graham",
    ]
    
    return defaults


def format_education_output(education_items):
    """Format education items for display (JSON format)."""
    if not education_items:
        return "[]"
    
    return json.dumps(education_items, indent=2)


def main():
    """Main validation script."""
    
    # Set up paths
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Select sample senators
    print("=" * 80)
    print("EDUCATION EXTRACTION VALIDATION")
    print("=" * 80)
    print("\nFetching Wikipedia pages for sample senators...\n")
    
    # Use sample senators (can be customized)
    sample_senators = select_sample_senators()
    
    # Process each senator
    results = []
    for i, senator_name in enumerate(sample_senators, 1):
        print(f"[{i}/{len(sample_senators)}] Processing: {senator_name}")
        
        result = fetch_and_extract_wikipedia(senator_name)
        results.append(result)
        
        # Add small delay to be respectful to Wikipedia
        if i < len(sample_senators):
            time.sleep(1)
    
    # Display results
    print("\n" + "=" * 80)
    print("EXTRACTION RESULTS")
    print("=" * 80 + "\n")
    
    for result in results:
        print("-" * 80)
        print(f"Senator: {result['senator_name']}")
        print(f"Wikipedia: {result['wikipedia_url']}")
        print(f"Status: {result['status']}")
        
        if result['education']:
            print(f"Education ({len(result['education'])} entries):")
            print(format_education_output(result['education']))
        else:
            print("Education: No entries extracted")
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    total_count = len(results)
    total_entries = sum(len(r['education']) for r in results)
    
    print(f"\nProcessed: {total_count} senators")
    print(f"Successful extractions: {success_count}/{total_count}")
    print(f"Total education entries: {total_entries}")
    
    # Show unique status values
    statuses = {}
    for r in results:
        status = r['status']
        statuses[status] = statuses.get(status, 0) + 1
    
    print("\nStatus breakdown:")
    for status, count in sorted(statuses.items()):
        print(f"  - {status}: {count}")
    
    # Save detailed results to JSON
    output_file = os.path.join(repo_root, "outputs", "education_extraction_results.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    print("\nNext steps:")
    print("1. Review the extracted education above")
    print("2. Visit the Wikipedia links for spot-checking")
    print("3. Check the infobox education section on Wikipedia")
    print("4. Compare with extracted values shown above")
    print("5. Note any mismatches or formatting issues")
    print("6. Report back with findings for refinement")


if __name__ == "__main__":
    main()
