#!/usr/bin/env python3
"""
Clean ground truth CSV by extracting only actual committee roles from the noisy committee_roles column.

The committee_roles column currently contains mixed content:
- Legitimate committee assignments (e.g., "U.S. Senate Alabama")
- Full bill descriptions (e.g., "The National Defense Authorization Act...")
- Election information (e.g., "U.S. Senate, Alaska General Election, 2010")
- Website/Social Media links (e.g., "WebsiteFacebookXInstagram")

This script:
1. Parses each row and splits the pipe-delimited committee_roles
2. Filters to keep only legitimate committee assignments
3. Removes bill descriptions, election info, and other noise
4. Saves a cleaned CSV
"""

import pandas as pd
import re
from pathlib import Path

def is_likely_committee(text):
    """
    Heuristic to determine if text is a committee role vs noise.
    
    Returns True if text looks like a committee assignment, False otherwise.
    """
    if not text or not isinstance(text, str):
        return False
    
    text = text.strip()
    
    # Too short or too long is usually noise
    if len(text) < 3 or len(text) > 150:
        return False
    
    # Keywords that indicate this is NOT a committee role
    noise_keywords = [
        'bill', 'act', 'passed', 'voted', 'election', 'facebook', 'instagram',
        'website', 'youtube', 'twitter', 'tiktok', 'candidate', 'officeholder',
        'required', 'majority', 'vote', 'senate and house', 'conference report',
        'general election', 'facebook', 'contact', 'phone', 'email'
    ]
    
    text_lower = text.lower()
    
    # Check for noise keywords
    if any(keyword in text_lower for keyword in noise_keywords):
        return False
    
    # Check for common committee/office patterns
    valid_patterns = [
        r'u\.?s\.?\s+(senate|house|representative)',
        r'committee',
        r'district\s+\d+',
        r'seat|representative',
    ]
    
    if any(re.search(pattern, text_lower) for pattern in valid_patterns):
        return True
    
    return False

def clean_committee_roles(committee_str):
    """
    Extract only legitimate committee roles from pipe-delimited string.
    
    Args:
        committee_str: Pipe-delimited string of (mixed) committee info
        
    Returns:
        Pipe-delimited string of cleaned committee roles, or empty string if none found
    """
    if not committee_str or pd.isna(committee_str):
        return ""
    
    # Split by pipe
    items = str(committee_str).split('|')
    
    # Filter to legitimate committees
    committees = [item.strip() for item in items if is_likely_committee(item)]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_committees = []
    for c in committees:
        c_lower = c.lower()
        if c_lower not in seen:
            seen.add(c_lower)
            unique_committees.append(c)
    
    # Return as pipe-delimited string
    return '|'.join(unique_committees)

def main():
    # Load ground truth CSV
    input_path = Path('/Users/chloe/LLM-Based-Personal-Profile-Extraction/external_data/ground_truth/senate_ground_truth.csv')
    output_path = Path('/Users/chloe/LLM-Based-Personal-Profile-Extraction/external_data/ground_truth/senate_ground_truth_cleaned.csv')
    
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"Original shape: {df.shape}")
    print(f"Sample committee_roles before cleaning:")
    print(f"  Length: {len(str(df.iloc[0]['committee_roles']))} chars")
    print(f"  First 200 chars: {str(df.iloc[0]['committee_roles'])[:200]}")
    print()
    
    # Clean committee_roles column
    print("Cleaning committee_roles column...")
    df['committee_roles'] = df['committee_roles'].apply(clean_committee_roles)
    
    print(f"Sample committee_roles after cleaning:")
    print(f"  Length: {len(str(df.iloc[0]['committee_roles']))} chars")
    print(f"  Full: {df.iloc[0]['committee_roles']}")
    print()
    
    # Show statistics
    print("=" * 70)
    print("CLEANING RESULTS")
    print("=" * 70)
    print(f"Senators with committee data: {(df['committee_roles'].str.len() > 0).sum()}/{len(df)}")
    print(f"Average committee_roles length: {df['committee_roles'].str.len().mean():.1f} chars")
    print(f"Max committee_roles length: {df['committee_roles'].str.len().max()} chars")
    print()
    
    # Show sample of cleaned data
    print("SAMPLE CLEANED DATA (first 3 rows):")
    print("-" * 70)
    for idx in range(min(3, len(df))):
        row = df.iloc[idx]
        committees = row['committee_roles']
        print(f"{row['name']:20s} ({row['state']})")
        if committees:
            for comm in committees.split('|'):
                print(f"  • {comm}")
        else:
            print(f"  (no committees)")
    print()
    
    # Save cleaned CSV
    df.to_csv(output_path, index=False)
    print(f"✓ Saved cleaned ground truth to: {output_path}")
    print(f"  Shape: {df.shape}")

if __name__ == "__main__":
    main()
