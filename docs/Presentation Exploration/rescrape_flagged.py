"""
rescrape_flagged.py
-------------------
Targeted re-scrape for senators whose HTML files failed quality audit
(too short, no bio content, or homepage fallback).

Usage:
    python rescrape_flagged.py

Edit FLAGGED_SENATORS below to add any senators identified by audit.
Output overwrites existing HTML files in senate_html/.
"""

import os
import time
import requests
from bs4 import BeautifulSoup

OUTPUT_DIR = "../external_data/senate_html"
DELAY_SECONDS = 2.0
TIMEOUT = 15
MIN_CHARS = 300  # minimum cleaned text length to consider a page valid

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; academic-research-scraper/1.0; "
        "contact: your@email.edu)"
    )
}

# ── Add flagged senators here ─────────────────────────────────────────────────
# Format: (filename_stem, [list of URLs to try in order])
# URLs are tried sequentially — first one that passes the content check wins.
FLAGGED_SENATORS = [
    # ("Tina_Smith_MN", [
    #     "https://www.smith.senate.gov/about-tina/biography/",
    #     "https://www.smith.senate.gov/about-tina/",
    #     "https://www.smith.senate.gov/about/biography/",
    # ]),

    # ("Andy_Kim_NJ", [
    #     "https://www.kim.senate.gov/about/biography/",
    #     "https://www.kim.senate.gov/about/",
    #     "https://www.kim.senate.gov/"
    # ]),

    # ("Amy_Klobuchar_MN", [
    #     "https://www.klobuchar.senate.gov/public/index.cfm/about-amy"
    # ]),

    ("Deb_Fischer_NE", [
        "https://www.fischer.senate.gov/public/index.cfm/biography",
        "https://www.fischer.senate.gov/public/index.cfm/extended-biography",
    ]),
    ("Jerry_Moran_KS", [
        "https://www.moran.senate.gov/public/index.cfm/biography",
    ]),
    ("John_Kennedy_LA", [
        "https://www.kennedy.senate.gov/public/biography",
    ]),
    ("John_Thune_SD", [
        "https://www.thune.senate.gov/public/index.cfm/biography",
    ]),
    ("Lindsey_Graham_SC", [
        "https://www.lgraham.senate.gov/public/index.cfm/biography",
    ]),
    ("Mark_Warner_VA", [
        "https://www.warner.senate.gov/public/index.cfm/biography",
        "https://www.warner.senate.gov/public/index.cfm/about",
    ]),
    ("Mitch_McConnell_KY", [
        "https://www.mcconnell.senate.gov/public/index.cfm/biography",
        "https://www.mcconnell.senate.gov/public/index.cfm/about",
    ]),
]

# ── Content quality check (mirrors pipeline) ──────────────────────────────────
def extract_readable_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)

def is_content_valid(html: str) -> tuple[bool, int]:
    text = extract_readable_text(html)
    return len(text) >= MIN_CHARS, len(text)

# ── Re-scrape loop ────────────────────────────────────────────────────────────
def rescrape():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session = requests.Session()

    print(f"Re-scraping {len(FLAGGED_SENATORS)} flagged senators\n")
    print("=" * 60)

    results = []
    for stem, url_candidates in FLAGGED_SENATORS:
        filepath = os.path.join(OUTPUT_DIR, f"{stem}.html")
        print(f"\n{stem}")

        success = False
        for url in url_candidates:
            print(f"  Trying: {url}")
            try:
                r = session.get(url, timeout=TIMEOUT, headers=HEADERS)
                if r.status_code != 200:
                    print(f"    ✗ Status {r.status_code}")
                    continue

                valid, char_count = is_content_valid(r.text)
                if not valid:
                    print(f"    ✗ Too short ({char_count} chars after cleaning)")
                    continue

                # Success — overwrite existing file
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(r.text)

                print(f"    ✓ Saved ({char_count} chars cleaned) → {stem}.html")
                results.append((stem, url, char_count, "SUCCESS"))
                success = True
                break

            except Exception as e:
                print(f"    ✗ Error: {e}")

        if not success:
            print(f"  ✗ All URLs failed for {stem} — manual review needed")
            results.append((stem, "N/A", 0, "FAILED"))

        time.sleep(DELAY_SECONDS)

    # Summary
    print("\n" + "=" * 60)
    print("RESCRAPE SUMMARY")
    print("=" * 60)
    for stem, url, chars, status in results:
        print(f"  {status:8s}  {stem:35s}  {chars} chars")

if __name__ == "__main__":
    rescrape()
