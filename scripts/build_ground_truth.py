"""
Ground Truth Builder Script

Regenerates senate_ground_truth.csv by scraping Wikipedia, Ballotpedia, and Pew data.

Usage:
    python build_ground_truth.py [--html-dir /path/to/senate_html] [--output /path/to/output.csv]

Dependencies:
    - modules.groundtruth (Wikipedia, Ballotpedia, Pew scrapers)
    - pandas, requests, beautifulsoup4, rapidfuzz
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import time
import re
from modules.data.groundtruth import WIKI_URL_OVERRIDES

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s | %(levelname)s | %(message)s'
)


def get_senator_urls_from_html_dir(html_dir):
    """
    Extract senator names and construct Wikipedia/Ballotpedia URLs from HTML directory.

    Args:
        html_dir: Path to directory containing senator HTML files named "FirstName_LastName_ST.html"

    Returns:
        list of dicts with keys: name, wiki_url, ballotpedia_url
    """
    html_dir = Path(html_dir)
    senators = []

    for html_file in sorted(html_dir.glob("*.html")):
        raw = html_file.stem  # e.g., "Adam_Schiff_CA"
        # Strip trailing _XX state code, replace underscores with spaces
        name = re.sub(r'_[A-Z]{2}$', '', raw).replace("_", " ")  # "Adam Schiff"

        # FIX: each URL constructed exactly once — no redundant overwrite
        wiki_url = WIKI_URL_OVERRIDES.get(name, "https://en.wikipedia.org/wiki/" + name.replace(" ", "_"))
        ballotpedia_url = "https://ballotpedia.org/" + name.replace(" ", "_") + "_(U.S._Senate)"

        senators.append({
            "senator_id": raw,
            "name": name,
            "wiki_url": wiki_url,
            "ballotpedia_url": ballotpedia_url,
        })

    return senators


def build_ground_truth(html_dir, pew_path, output_path, committee_yaml_path,
                       committees_yaml_path=None, resume=True):
    """
    Build ground truth CSV by scraping Wikipedia, Ballotpedia, and merging Pew data.

    Args:
        html_dir: Path to directory with senator HTML files
        pew_path: Path to Pew religion CSV
        output_path: Where to save the output CSV
        committee_yaml_path: Path to committee-membership-current.yaml
        committees_yaml_path: Path to committees-current.yaml for ID resolution
        resume: If True, skip senators already present in output_path

    Returns:
        DataFrame with columns: name, full_name, birthdate, gender,
                                race_ethnicity, committee_roles, religion
    """
    from modules.data.groundtruth import (
        scrape_wikipedia, merge_pew, normalize_birthdate,
        load_committees_from_yaml, build_committee_lookup, resolve_committee_roles,
        WIKI_URL_OVERRIDES
    )

    # FIX: tqdm — prefer CLI (works everywhere); notebook tqdm only as upgrade
    try:
        get_ipython  # noqa: F821 — defined in Jupyter kernels
        from tqdm.notebook import tqdm
    except NameError:
        from tqdm import tqdm

    output_path = Path(output_path)

    # Resume from checkpoint
    if resume and output_path.exists():
        print(f"📋 Resuming from checkpoint: {output_path}")
        df_progress = pd.read_csv(output_path)
        completed_names = set(df_progress["name"].tolist())
        print(f"   Already completed: {len(completed_names)} senators")
    else:
        df_progress = None
        completed_names = set()

    print(f"📂 Reading senator HTML files from {html_dir}...")
    senators = get_senator_urls_from_html_dir(html_dir)
    print(f"   Found: {len(senators)} senators")

    senators_todo = [s for s in senators if s["name"] not in completed_names]
    print(f"   To process: {len(senators_todo)} senators\n")

    if not senators_todo:
        print("✓ All senators already processed. Skipping rebuild.")
        return df_progress if df_progress is not None else pd.DataFrame()

    senator_names = [s["name"] for s in senators_todo]
    committee_map = load_committees_from_yaml(committee_yaml_path, senator_names=senator_names)
    committee_lookup = build_committee_lookup(committees_yaml_path)

    rows = []
    for senator_info in tqdm(senators_todo, desc="Scraping senators"):
        name = senator_info["name"]

        wiki_data = scrape_wikipedia(senator_info["wiki_url"], name)

        if wiki_data.get("birthdate"):
            wiki_data["birthdate"] = normalize_birthdate(wiki_data["birthdate"])

        rows.append({
            "senator_id": senator_info["senator_id"],
            "name": name,
            "full_name": wiki_data.get("full_name"),
            "birthdate": wiki_data.get("birthdate"),
            "gender": wiki_data.get("gender"),
            "race_ethnicity": wiki_data.get("race_ethnicity"),
            "committee_roles": resolve_committee_roles(
                committee_map.get(name), committee_lookup
            ),
        })

        time.sleep(0.5)

    # Combine with checkpoint
    if df_progress is not None:
        df_combined = pd.concat([df_progress, pd.DataFrame(rows)], ignore_index=True)
    else:
        df_combined = pd.DataFrame(rows)

    # Merge Pew religion
    print("\n🔗 Merging Pew religion data...")
    if not df_combined.empty:
        df_combined["religion"] = merge_pew(df_combined, pew_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(output_path, index=False)
    print(f"\n✓ Ground truth saved to: {output_path}")
    print(f"  Total senators: {len(df_combined)}")
    print(f"  Columns: {', '.join(df_combined.columns.tolist())}\n")

    return df_combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build ground truth CSV from Wikipedia, Ballotpedia, and Pew data"
    )
    parser.add_argument(
        "--html-dir", type=str,
        default="../external_data/senate_html",
        help="Path to directory containing senator HTML files"
    )
    parser.add_argument(
        "--pew-path", type=str,
        default="../external_data/ground_truth/pew_religion.csv",
        help="Path to Pew religion CSV"
    )
    parser.add_argument(
        "--output", type=str,
        default="../external_data/ground_truth/senate_ground_truth.csv",
        help="Output path for ground truth CSV"
    )
    parser.add_argument(
        "--committee-yaml", type=str,
        default="../external_data/committee-membership-current.yaml",
        help="Path to committee membership YAML"
    )
    parser.add_argument(
        "--committees-yaml", type=str,
        default="../external_data/committees-current.yaml",
        help="Path to committees-current.yaml for ID resolution"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh instead of resuming from checkpoint"
    )

    args = parser.parse_args()

    build_ground_truth(
        html_dir=args.html_dir,
        pew_path=args.pew_path,
        output_path=args.output,
        committee_yaml_path=args.committee_yaml,
        # FIX: was args.committees_yml (misspelled) — silently passed None
        committees_yaml_path=args.committees_yaml,
        resume=not args.no_resume,
    )
