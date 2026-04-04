"""
audit_pipeline_inputs.py
Diagnostic script for Senate LLM Pipeline — DSBA 6010
Checks:
  1. HTML file quality (text length, key field presence)
  2. Pew religion fuzzy match audit
Usage:
  python audit_pipeline_inputs.py --html_dir ../external_data/senate_html --pew pew_religion.csv
"""

import argparse
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from rapidfuzz import process, fuzz
import json

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--html_dir", default="../external_data/senate_html", help="Path to senate HTML files")
parser.add_argument("--pew", default="pew_religion.csv", help="Path to pew_religion.csv")
parser.add_argument("--min_chars", type=int, default=300, help="Minimum cleaned text length before flagging")
parser.add_argument("--match_threshold", type=int, default=85, help="Fuzzy match score threshold (0-100)")
args = parser.parse_args()

HTML_DIR = Path(args.html_dir)
PEW_PATH = Path(args.pew)

# ── HTML cleaning (mirrors pipeline) ─────────────────────────────────────────
def extract_readable_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)

# ── Normalize name from filename stem ────────────────────────────────────────
def stem_to_name(stem: str) -> str:
    """Bernie_Moreno_OH -> Bernie Moreno"""
    parts = stem.split("_")
    # Drop last part if it looks like a state abbreviation (2 uppercase letters)
    if parts[-1].isupper() and len(parts[-1]) == 2:
        parts = parts[:-1]
    return " ".join(parts)

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1: HTML Quality Audit
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CHECK 1: HTML QUALITY AUDIT")
print("=" * 70)

html_files = sorted(HTML_DIR.glob("*.html"))
print(f"Total HTML files found: {len(html_files)}\n")

KEY_SIGNALS = ["committee", "education", "born", "senator", "degree", "university", "college"]

html_audit = []
for f in html_files:
    html = f.read_text(encoding="utf-8", errors="ignore")
    text = extract_readable_text(html)
    text_lower = text.lower()

    signals_found = [s for s in KEY_SIGNALS if s in text_lower]
    row = {
        "file": f.stem,
        "raw_html_chars": len(html),
        "cleaned_text_chars": len(text),
        "flagged_short": len(text) < args.min_chars,
        "signals_found": ", ".join(signals_found) if signals_found else "NONE",
        "signal_count": len(signals_found),
    }
    html_audit.append(row)

df_html = pd.DataFrame(html_audit)

# Summary
flagged = df_html[df_html["flagged_short"]]
no_signals = df_html[df_html["signal_count"] == 0]

print(f"Files with cleaned text < {args.min_chars} chars (suspicious): {len(flagged)}")
if not flagged.empty:
    print(flagged[["file", "cleaned_text_chars"]].to_string(index=False))

print(f"\nFiles with NO key signals found: {len(no_signals)}")
if not no_signals.empty:
    print(no_signals[["file", "cleaned_text_chars"]].to_string(index=False))

print(f"\nText length distribution:")
print(df_html["cleaned_text_chars"].describe().round(0).to_string())

df_html.to_csv("html_quality_audit.csv", index=False)
print(f"\nFull audit saved to: html_quality_audit.csv")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2: Pew Fuzzy Match Audit
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CHECK 2: PEW RELIGION FUZZY MATCH AUDIT")
print("=" * 70)

df_pew = pd.read_csv(PEW_PATH)
pew_names = df_pew["name"].tolist()

# Build name list from HTML stems
html_names = [(f.stem, stem_to_name(f.stem)) for f in html_files]

match_audit = []
for stem, normalized_name in html_names:
    match = process.extractOne(normalized_name, pew_names, scorer=fuzz.token_sort_ratio)
    if match:
        matched_name, score, _ = match
        matched_row = df_pew[df_pew["name"] == matched_name].iloc[0]
        match_audit.append({
            "html_stem": stem,
            "normalized_name": normalized_name,
            "pew_match": matched_name,
            "match_score": score,
            "religion": matched_row["religion"],
            "below_threshold": score < args.match_threshold,
        })
    else:
        match_audit.append({
            "html_stem": stem,
            "normalized_name": normalized_name,
            "pew_match": None,
            "match_score": 0,
            "religion": None,
            "below_threshold": True,
        })

df_matches = pd.DataFrame(match_audit)

# Summary
below = df_matches[df_matches["below_threshold"]]
print(f"Match threshold: {args.match_threshold}")
print(f"Total HTML files: {len(df_matches)}")
print(f"Matched above threshold: {len(df_matches) - len(below)}")
print(f"Below threshold (risky matches): {len(below)}")

if not below.empty:
    print("\nRisky matches (spot-check these):")
    print(below[["normalized_name", "pew_match", "match_score", "religion"]].to_string(index=False))

print(f"\nSample of high-confidence matches:")
print(df_matches[~df_matches["below_threshold"]].head(10)[
    ["normalized_name", "pew_match", "match_score", "religion"]
].to_string(index=False))

# Check Pew senators with no HTML match
matched_pew_names = set(df_matches["pew_match"].dropna())
unmatched_pew = df_pew[~df_pew["name"].isin(matched_pew_names)]
print(f"\nPew senators with no HTML file match: {len(unmatched_pew)}")
if not unmatched_pew.empty:
    print(unmatched_pew[["name", "state", "religion"]].to_string(index=False))

df_matches.to_csv("pew_match_audit.csv", index=False)
print(f"\nFull match audit saved to: pew_match_audit.csv")

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
