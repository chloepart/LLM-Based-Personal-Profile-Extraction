"""
senate_scraper.py
-----------------
Fetches raw HTML from the official About/Biography page of all 100 U.S. Senators.
Mirrors the Liu et al. (2025) data collection approach: minimal preprocessing,
raw web content as LLM input.

Usage:
    python senate_scraper.py

Output:
    ./senate_html/<LAST_FIRST_STATE>.html  — one file per senator
    ./senate_html/scrape_log.csv           — success/failure log
    ./senate_html/senators_index.csv       — senator metadata (name, state, url)

Dependencies:
    pip install requests pyyaml
"""

import csv
import os
import time
import requests
import yaml

# ---------------------------------------------------------------------------
# STEP 1: Load senator list
# Source: https://github.com/unitedstates/congress-legislators (public domain)
# The YAML is fetched at runtime. If GitHub is unavailable, falls back to the
# hardcoded seed list below.
# ---------------------------------------------------------------------------

YAML_URL = (
    "https://raw.githubusercontent.com/unitedstates/"
    "congress-legislators/main/legislators-current.yaml"
)

# Fallback: manually curated seed list (name, state, official website).
# Covers all 100 senators as of early 2025. Update as needed.
FALLBACK_SENATORS = [
    ("Katie Britt", "AL", "https://www.britt.senate.gov"),
    ("Tommy Tuberville", "AL", "https://www.tuberville.senate.gov"),
    ("Lisa Murkowski", "AK", "https://www.murkowski.senate.gov"),
    ("Dan Sullivan", "AK", "https://www.sullivan.senate.gov"),
    ("Mark Kelly", "AZ", "https://www.kelly.senate.gov"),
    ("Ruben Gallego", "AZ", "https://www.gallego.senate.gov"),
    ("Tom Cotton", "AR", "https://www.cotton.senate.gov"),
    ("John Boozman", "AR", "https://www.boozman.senate.gov"),
    ("Alex Padilla", "CA", "https://www.padilla.senate.gov"),
    ("Adam Schiff", "CA", "https://www.schiff.senate.gov"),
    ("John Hickenlooper", "CO", "https://www.hickenlooper.senate.gov"),
    ("Michael Bennet", "CO", "https://www.bennet.senate.gov"),
    ("Chris Murphy", "CT", "https://www.murphy.senate.gov"),
    ("Richard Blumenthal", "CT", "https://www.blumenthal.senate.gov"),
    ("Lisa Blunt Rochester", "DE", "https://www.bluntrochester.senate.gov"),
    ("Chris Coons", "DE", "https://www.coons.senate.gov"),
    ("Rick Scott", "FL", "https://www.rickscott.senate.gov"),
    ("Marco Rubio", "FL", "https://www.rubio.senate.gov"),
    ("Jon Ossoff", "GA", "https://www.ossoff.senate.gov"),
    ("Raphael Warnock", "GA", "https://www.warnock.senate.gov"),
    ("Mazie Hirono", "HI", "https://www.hirono.senate.gov"),
    ("Brian Schatz", "HI", "https://www.schatz.senate.gov"),
    ("Mike Crapo", "ID", "https://www.crapo.senate.gov"),
    ("Jim Risch", "ID", "https://www.risch.senate.gov"),
    ("Dick Durbin", "IL", "https://www.durbin.senate.gov"),
    ("Tammy Duckworth", "IL", "https://www.duckworth.senate.gov"),
    ("Todd Young", "IN", "https://www.young.senate.gov"),
    ("Jim Banks", "IN", "https://www.banks.senate.gov"),
    ("Chuck Grassley", "IA", "https://www.grassley.senate.gov"),
    ("Joni Ernst", "IA", "https://www.ernst.senate.gov"),
    ("Jerry Moran", "KS", "https://www.moran.senate.gov"),
    ("Roger Marshall", "KS", "https://www.marshall.senate.gov"),
    ("Mitch McConnell", "KY", "https://www.mcconnell.senate.gov"),
    ("Rand Paul", "KY", "https://www.paul.senate.gov"),
    ("Bill Cassidy", "LA", "https://www.cassidy.senate.gov"),
    ("John Kennedy", "LA", "https://www.kennedy.senate.gov"),
    ("Susan Collins", "ME", "https://www.collins.senate.gov"),
    ("Angus King", "ME", "https://www.king.senate.gov"),
    ("Ben Cardin", "MD", "https://www.cardin.senate.gov"),
    ("Angela Alsobrooks", "MD", "https://www.alsobrooks.senate.gov"),
    ("Elizabeth Warren", "MA", "https://www.warren.senate.gov"),
    ("Ed Markey", "MA", "https://www.markey.senate.gov"),
    ("Debbie Stabenow", "MI", "https://www.stabenow.senate.gov"),
    ("Gary Peters", "MI", "https://www.peters.senate.gov"),
    ("Amy Klobuchar", "MN", "https://www.klobuchar.senate.gov"),
    ("Tina Smith", "MN", "https://www.smith.senate.gov"),
    ("Cindy Hyde-Smith", "MS", "https://www.hydesmith.senate.gov"),
    ("Roger Wicker", "MS", "https://www.wicker.senate.gov"),
    ("Josh Hawley", "MO", "https://www.hawley.senate.gov"),
    ("Eric Schmitt", "MO", "https://www.schmitt.senate.gov"),
    ("Steve Daines", "MT", "https://www.daines.senate.gov"),
    ("Tim Sheehy", "MT", "https://www.sheehy.senate.gov"),
    ("Deb Fischer", "NE", "https://www.fischer.senate.gov"),
    ("Pete Ricketts", "NE", "https://www.ricketts.senate.gov"),
    ("Catherine Cortez Masto", "NV", "https://www.cortezMasto.senate.gov"),
    ("Jacky Rosen", "NV", "https://www.rosen.senate.gov"),
    ("Jeanne Shaheen", "NH", "https://www.shaheen.senate.gov"),
    ("Maggie Hassan", "NH", "https://www.hassan.senate.gov"),
    ("Cory Booker", "NJ", "https://www.booker.senate.gov"),
    ("Andy Kim", "NJ", "https://www.andykim.senate.gov"),
    ("Martin Heinrich", "NM", "https://www.heinrich.senate.gov"),
    ("Ben Ray Lujan", "NM", "https://www.lujan.senate.gov"),
    ("Chuck Schumer", "NY", "https://www.schumer.senate.gov"),
    ("Kirsten Gillibrand", "NY", "https://www.gillibrand.senate.gov"),
    ("Thom Tillis", "NC", "https://www.tillis.senate.gov"),
    ("Ted Budd", "NC", "https://www.budd.senate.gov"),
    ("John Hoeven", "ND", "https://www.hoeven.senate.gov"),
    ("Kevin Cramer", "ND", "https://www.cramer.senate.gov"),
    ("Sherrod Brown", "OH", "https://www.brown.senate.gov"),
    ("Bernie Moreno", "OH", "https://www.moreno.senate.gov"),
    ("James Lankford", "OK", "https://www.lankford.senate.gov"),
    ("Markwayne Mullin", "OK", "https://www.mullin.senate.gov"),
    ("Ron Wyden", "OR", "https://www.wyden.senate.gov"),
    ("Jeff Merkley", "OR", "https://www.merkley.senate.gov"),
    ("Bob Casey", "PA", "https://www.casey.senate.gov"),
    ("Dave McCormick", "PA", "https://www.mccormick.senate.gov"),
    ("Jack Reed", "RI", "https://www.reed.senate.gov"),
    ("Sheldon Whitehouse", "RI", "https://www.whitehouse.senate.gov"),
    ("Tim Scott", "SC", "https://www.scott.senate.gov"),
    ("Lindsey Graham", "SC", "https://www.lgraham.senate.gov"),
    ("John Thune", "SD", "https://www.thune.senate.gov"),
    ("Mike Rounds", "SD", "https://www.rounds.senate.gov"),
    ("Marsha Blackburn", "TN", "https://www.blackburn.senate.gov"),
    ("Bill Hagerty", "TN", "https://www.hagerty.senate.gov"),
    ("John Cornyn", "TX", "https://www.cornyn.senate.gov"),
    ("Ted Cruz", "TX", "https://www.cruz.senate.gov"),
    ("Mike Lee", "UT", "https://www.lee.senate.gov"),
    ("John Curtis", "UT", "https://www.curtis.senate.gov"),
    ("Peter Welch", "VT", "https://www.welch.senate.gov"),
    ("Bernie Sanders", "VT", "https://www.sanders.senate.gov"),
    ("Tim Kaine", "VA", "https://www.kaine.senate.gov"),
    ("Mark Warner", "VA", "https://www.warner.senate.gov"),
    ("Maria Cantwell", "WA", "https://www.cantwell.senate.gov"),
    ("Patty Murray", "WA", "https://www.murray.senate.gov"),
    ("Joe Manchin", "WV", "https://www.manchin.senate.gov"),
    ("Shelley Moore Capito", "WV", "https://www.capito.senate.gov"),
    ("Tammy Baldwin", "WI", "https://www.baldwin.senate.gov"),
    ("Ron Johnson", "WI", "https://www.ronjohnson.senate.gov"),
    ("John Barrasso", "WY", "https://www.barrasso.senate.gov"),
    ("Cynthia Lummis", "WY", "https://www.lummis.senate.gov"),
]

# Common paths where senator bio pages are typically found.
# We try them in order and keep the first that returns a non-redirect 200.
BIO_PATH_CANDIDATES = [
    "/about/",
    "/about/biography/",
    "/about/biography",
    "/biography/",
    "/senator/",
    "/",   # fallback: homepage
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; academic-research-scraper/1.0; "
        "contact: your@email.edu)"
    )
}

DELAY_SECONDS = 1.5   # polite crawl delay between requests
TIMEOUT = 15
OUTPUT_DIR = "./senate_html"


# ---------------------------------------------------------------------------
# STEP 2: Try to load from GitHub YAML; fall back to hardcoded list
# ---------------------------------------------------------------------------

def load_senators_from_yaml(url: str):
    """Fetch legislators-current.yaml and extract current senators."""
    try:
        r = requests.get(url, timeout=10, headers=HEADERS)
        r.raise_for_status()
        members = yaml.safe_load(r.content)
        senators = []
        for m in members:
            last_term = m["terms"][-1]
            if last_term.get("type") != "sen":
                continue
            full_name = m["name"].get("official_full", "")
            state = last_term.get("state", "")
            website = last_term.get("url", "")
            if full_name and state and website:
                senators.append((full_name, state, website))
        print(f"[INFO] Loaded {len(senators)} senators from GitHub YAML.")
        return senators
    except Exception as e:
        print(f"[WARN] Could not fetch YAML ({e}). Using fallback list.")
        return None


def get_senator_list():
    senators = load_senators_from_yaml(YAML_URL)
    if not senators:
        senators = FALLBACK_SENATORS
        print(f"[INFO] Using fallback list: {len(senators)} senators.")
    return senators


# ---------------------------------------------------------------------------
# STEP 3: Resolve the best bio URL for a given senator website
# ---------------------------------------------------------------------------

def resolve_bio_url(base_url: str, session: requests.Session) -> tuple[str, int]:
    """
    Try known bio path candidates. Return (url, status_code) for the first
    successful hit, or (base_url, last_status) if none resolve cleanly.
    """
    base = base_url.rstrip("/")
    last_status = 0
    for path in BIO_PATH_CANDIDATES:
        candidate = base + path
        try:
            r = session.get(
                candidate, timeout=TIMEOUT, headers=HEADERS, allow_redirects=True
            )
            if r.status_code == 200:
                return candidate, 200
            last_status = r.status_code
        except Exception:
            pass
    return base + "/", last_status


# ---------------------------------------------------------------------------
# STEP 4: Scrape and save
# ---------------------------------------------------------------------------

def safe_filename(name: str, state: str) -> str:
    clean = name.replace(" ", "_").replace("'", "").replace(".", "")
    return f"{clean}_{state}.html"


def scrape_all(senators: list) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, "scrape_log.csv")
    index_path = os.path.join(OUTPUT_DIR, "senators_index.csv")

    session = requests.Session()

    with open(log_path, "w", newline="", encoding="utf-8") as log_f, \
         open(index_path, "w", newline="", encoding="utf-8") as idx_f:

        log_writer = csv.writer(log_f)
        log_writer.writerow(["name", "state", "url_attempted", "status", "filename", "error"])

        idx_writer = csv.writer(idx_f)
        idx_writer.writerow(["name", "state", "base_url", "scraped_url", "filename"])

        total = len(senators)
        for i, (name, state, base_url) in enumerate(senators, 1):
            print(f"[{i:03d}/{total}] {name} ({state}) ... ", end="", flush=True)

            bio_url, status = resolve_bio_url(base_url, session)
            filename = safe_filename(name, state)
            filepath = os.path.join(OUTPUT_DIR, filename)

            try:
                r = session.get(bio_url, timeout=TIMEOUT, headers=HEADERS)
                r.raise_for_status()
                html = r.text

                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(html)

                print(f"OK ({len(html):,} bytes) → {filename}")
                log_writer.writerow([name, state, bio_url, r.status_code, filename, ""])
                idx_writer.writerow([name, state, base_url, bio_url, filename])

            except Exception as e:
                print(f"FAILED — {e}")
                log_writer.writerow([name, state, bio_url, status, "", str(e)])

            time.sleep(DELAY_SECONDS)

    print(f"\n[DONE] HTML files saved to: {OUTPUT_DIR}/")
    print(f"       Log:   {log_path}")
    print(f"       Index: {index_path}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    senators = get_senator_list()
    scrape_all(senators)
