"""
crawl_tamu.py  –  Exhaustive BFS crawler for tamu.edu

Design goals
────────────
1. Start from a curated seed list (key entry-points across all major sub-sites).
2. Follow ALL internal tamu.edu links discovered on each fetched page (BFS).
3. Never visit the same URL twice (SHA-256 dedup).
4. Skip URLs that are unlikely to contain useful text:
     – binary extensions (.pdf handled separately, images/video skipped)
     – query-string / fragment variants of seen canonical paths
     – known low-value patterns (login, logout, search-results, calendar widgets)
5. Enforce a configurable page cap (default 2 000) so the run stays manageable.
6. Respect a polite download delay between requests.
7. Save each page as a cleaned .html file in data/documents/ so create_database.py
   can re-index without any changes.

Usage
─────
    # Basic exhaustive crawl (up to MAX_PAGES pages):
    .venv\\Scripts\\python.exe crawl_tamu.py

    # Custom cap and delay:
    .venv\\Scripts\\python.exe crawl_tamu.py --max-pages 5000 --delay 0.6

    # After crawling, rebuild the vector store:
    .venv\\Scripts\\python.exe create_database.py --reset
"""

import argparse
import hashlib
import os
import re
import time
from collections import deque
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

# ── Configuration ─────────────────────────────────────────────────────────────

# Key entry-points — the BFS will expand from these into the full site.
SEED_URLS = [
    # Main portal
    "https://www.tamu.edu/",
    "https://www.tamu.edu/about/index.html",

    # Admissions
    "https://admissions.tamu.edu/",
    "https://admissions.tamu.edu/freshman/requirements.html",
    "https://admissions.tamu.edu/costs/",
    "https://admissions.tamu.edu/transfer/",

    # Financial Aid
    "https://financialaid.tamu.edu/",
    "https://financialaid.tamu.edu/undergraduate/types-of-aid/",
    "https://financialaid.tamu.edu/undergraduate/scholarships/",

    # Academic Catalog — course descriptions (all departments the bot gets asked about)
    "https://catalog.tamu.edu/undergraduate/course-descriptions/csce/",
    "https://catalog.tamu.edu/undergraduate/course-descriptions/math/",
    "https://catalog.tamu.edu/undergraduate/course-descriptions/biol/",
    "https://catalog.tamu.edu/undergraduate/course-descriptions/phys/",
    "https://catalog.tamu.edu/undergraduate/course-descriptions/ecen/",
    "https://catalog.tamu.edu/undergraduate/course-descriptions/hist/",
    "https://catalog.tamu.edu/undergraduate/course-descriptions/aero/",
    "https://catalog.tamu.edu/undergraduate/course-descriptions/pete/",
    "https://catalog.tamu.edu/undergraduate/course-descriptions/cven/",
    "https://catalog.tamu.edu/undergraduate/course-descriptions/isen/",
    "https://catalog.tamu.edu/undergraduate/course-descriptions/meen/",
    "https://catalog.tamu.edu/undergraduate/course-descriptions/chen/",
    "https://catalog.tamu.edu/undergraduate/course-descriptions/stat/",
    "https://catalog.tamu.edu/undergraduate/course-descriptions/acct/",
    "https://catalog.tamu.edu/undergraduate/course-descriptions/mgmt/",
    "https://catalog.tamu.edu/undergraduate/course-descriptions/pols/",
    "https://catalog.tamu.edu/undergraduate/course-descriptions/psyc/",
    "https://catalog.tamu.edu/undergraduate/course-descriptions/soci/",
    "https://catalog.tamu.edu/graduate/colleges-schools-interdisciplinary/engineering/computer-science/",

    # Policies & student rules
    "https://student-rules.tamu.edu/",
    "https://student-rules.tamu.edu/studentrules",
    "https://registrar.tamu.edu/",
    "https://registrar.tamu.edu/Catalogs,-Policies-Procedures/Academic-Calendar",

    # Colleges & departments
    "https://engineering.tamu.edu/",
    "https://engineering.tamu.edu/cse/index.html",
    "https://engineering.tamu.edu/aerospace/index.html",
    "https://engineering.tamu.edu/civil/index.html",
    "https://engineering.tamu.edu/mechanical/index.html",
    "https://engineering.tamu.edu/electrical/index.html",
    "https://artsci.tamu.edu/",
    "https://bush.tamu.edu/",
    "https://mays.tamu.edu/",
    "https://veterinary.tamu.edu/",
    "https://education.tamu.edu/",
    "https://arch.tamu.edu/",
    "https://law.tamu.edu/",
    "https://nursing.tamu.edu/",
    "https://pharmacy.tamu.edu/",

    # Faculty / personnel directories (key for "who is the head of X" queries)
    "https://engineering.tamu.edu/cse/people/",
    "https://engineering.tamu.edu/aerospace/people.html",
    "https://engineering.tamu.edu/civil/people/",
    "https://engineering.tamu.edu/mechanical/people/",
    "https://engineering.tamu.edu/electrical/people/",
    "https://cse.tamu.edu/people/",

    # Campus life
    "https://studentlife.tamu.edu/",
    "https://msc.tamu.edu/",
    "https://library.tamu.edu/",
    "https://rec.tamu.edu/",
    "https://housing.tamu.edu/",
    "https://dining.tamu.edu/",
    "https://careercenter.tamu.edu/",

    # Research & graduate
    "https://research.tamu.edu/",
    "https://gradcatalog.tamu.edu/",
    "https://grad.tamu.edu/",

    # Newly added high-value targets (eval gaps)
    "https://reslife.tamu.edu/",
    "https://howdy.tamu.edu/",
    "https://aggiehonor.tamu.edu/",
    "https://catalog.tamu.edu/undergraduate/science/",
]

# File extensions that contain no useful text — skip them entirely.
SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico", ".webp",
    ".mp4", ".mp3", ".avi", ".mov", ".wmv",
    ".zip", ".gz", ".tar", ".exe", ".msi",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".js", ".css", ".json", ".xml", ".rss",
}

# URL path patterns that are never useful for Q&A.
SKIP_PATTERNS = re.compile(
    r"/(login|logout|sign-?in|sign-?out|auth|sso|cas|"
    r"cart|checkout|order|pay|donate|give|"
    r"search|query|results|sitemap|robots|"
    r"feed|rss|atom|api/|ajax|wp-json|"
    r"calendar/event|ical|\.ics|"
    r"print/?$|pdf-print|embed)",
    re.IGNORECASE,
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; HowdyAI-Crawler/1.0; "
        "+https://howdyai.tamu.edu/bot)"
    )
}
TIMEOUT = 10
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "documents")
os.makedirs(OUT_DIR, exist_ok=True)

# Robots.txt cache: netloc → RobotFileParser
_robots_cache: dict[str, RobotFileParser] = {}


# ── robots.txt helper ─────────────────────────────────────────────────────────

def can_fetch(url: str) -> bool:
    """Return True if our UA is allowed to fetch this URL per robots.txt."""
    parsed = urlparse(url)
    netloc = parsed.netloc
    if netloc not in _robots_cache:
        rp = RobotFileParser()
        rp.set_url(f"{parsed.scheme}://{netloc}/robots.txt")
        try:
            rp.read()
        except Exception:
            rp.allow_all = True
        _robots_cache[netloc] = rp
    return _robots_cache[netloc].can_fetch("*", url)


# ── URL normalisation & filtering ─────────────────────────────────────────────

def normalise(url: str, base: str) -> str | None:
    """
    Resolve a (possibly relative) href against base, strip fragment/tracking
    params, and return a canonical URL string — or None if it should be skipped.
    """
    try:
        full = urljoin(base, url.strip())
    except Exception:
        return None

    parsed = urlparse(full)

    # Must be http/https
    if parsed.scheme not in ("http", "https"):
        return None

    # Must stay within tamu.edu
    if not parsed.netloc.endswith("tamu.edu"):
        return None

    # Skip unwanted extensions
    path_lower = parsed.path.lower()
    if any(path_lower.endswith(ext) for ext in SKIP_EXTENSIONS):
        return None

    # Skip low-value URL patterns
    if SKIP_PATTERNS.search(parsed.path):
        return None

    # Canonicalise: drop fragment and query string (most TAMU pages are static)
    canonical = urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
    # Normalise trailing slash: treat /foo and /foo/ as the same
    if canonical.endswith("/") and len(canonical) > len(f"{parsed.scheme}://{parsed.netloc}/"):
        canonical = canonical.rstrip("/")
    return canonical


def url_fingerprint(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()


# ── HTML cleaning ─────────────────────────────────────────────────────────────

def clean_html(soup: BeautifulSoup) -> str:
    """Strip boilerplate and return body HTML ready for BSHTMLLoader."""
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "noscript", "aside", "form", "iframe"]):
        tag.decompose()
    # Also strip elements whose class contains common boilerplate class names
    for tag in soup.find_all(class_=re.compile(
            r"cookie|banner|popup|modal|overlay|ad-|advertisement|"
            r"sidebar|breadcrumb|pagination|social|share|widget", re.I)):
        tag.decompose()
    body = soup.find("body") or soup
    return str(body)


# ── Save ──────────────────────────────────────────────────────────────────────

def safe_filename(url: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.replace(".", "_")
    path = re.sub(r"[^a-zA-Z0-9_-]", "_", parsed.path)
    slug = (domain + path).strip("_")[:80]
    uid = hashlib.md5(url.encode()).hexdigest()[:6]
    return f"{slug}_{uid}.html"


def save_doc(doc: dict) -> str:
    fname = safe_filename(doc["url"])
    fpath = os.path.join(OUT_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(f"<!-- SOURCE: {doc['url']} TITLE: {doc['title']} -->\n")
        f.write(doc["html"])
    return fpath


# ── Fetch & parse ─────────────────────────────────────────────────────────────

def fetch(url: str) -> tuple[dict | None, list[str]]:
    """
    Fetch a page.  Returns (doc_dict_or_None, list_of_discovered_hrefs).
    """
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
        if r.status_code != 200:
            print(f"  ✗ {r.status_code}  {url}")
            return None, []
        content_type = r.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return None, []

        soup = BeautifulSoup(r.text, "lxml")
        title = soup.title.string.strip() if soup.title else url

        # Collect all hrefs for BFS expansion
        hrefs = [a.get("href", "") for a in soup.find_all("a", href=True)]

        html = clean_html(soup)

        # Skip pages with very little content (likely login walls or empty pages)
        text_only = BeautifulSoup(html, "lxml").get_text(" ", strip=True)
        if len(text_only) < 200:
            print(f"  ⚠ Too short ({len(text_only)} chars), skipping: {url}")
            return None, hrefs

        return {"url": url, "title": title, "html": html}, hrefs

    except requests.exceptions.Timeout:
        print(f"  ✗ TIMEOUT  {url}")
        return None, []
    except Exception as e:
        print(f"  ✗ ERR  {url}  ({e})")
        return None, []


# ── BFS crawler ───────────────────────────────────────────────────────────────

def run_bfs(seed_urls: list[str], max_pages: int, delay: float) -> None:
    visited: set[str] = set()      # canonical URL strings already enqueued
    queue: deque[str] = deque()

    for url in seed_urls:
        norm = normalise(url, url)
        if norm and norm not in visited:
            visited.add(norm)
            queue.append(norm)

    saved = 0
    skipped = 0
    total_fetched = 0

    print(f"\n{'─'*60}")
    print(f"  HowdyAI Exhaustive Crawler")
    print(f"  Seeds: {len(seed_urls)}  |  Cap: {max_pages}  |  Delay: {delay}s")
    print(f"  Output: {OUT_DIR}")
    print(f"{'─'*60}\n")

    while queue and total_fetched < max_pages:
        url = queue.popleft()
        total_fetched += 1

        print(f"[{total_fetched:04}/{max_pages}] {url}")

        if not can_fetch(url):
            print(f"  ⚠ Blocked by robots.txt")
            skipped += 1
            continue

        doc, hrefs = fetch(url)
        if doc:
            save_doc(doc)
            saved += 1
            print(f"  ✓ {doc['title'][:70]}  ({len(doc['html'])} chars)")
        else:
            skipped += 1

        # Expand the BFS frontier with newly discovered links
        new_links = 0
        for href in hrefs:
            norm = normalise(href, url)
            if norm and norm not in visited:
                visited.add(norm)
                queue.append(norm)
                new_links += 1

        print(f"  → +{new_links} new links queued  (queue size: {len(queue)})")
        time.sleep(delay)

    print(f"\n{'─'*60}")
    print(f"Crawl complete.")
    print(f"  Pages fetched:  {total_fetched}")
    print(f"  Pages saved:    {saved}")
    print(f"  Skipped/failed: {skipped}")
    print(f"  Unique URLs seen: {len(visited)}")
    print(f"\nNext step:  .venv\\Scripts\\python.exe create_database.py --reset")
    print(f"{'─'*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exhaustive BFS crawler for tamu.edu → data/documents/"
    )
    parser.add_argument(
        "--max-pages", type=int, default=2000,
        help="Maximum number of pages to fetch (default: 2000)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Seconds to wait between requests (default: 0.5)"
    )
    args = parser.parse_args()

    run_bfs(SEED_URLS, max_pages=args.max_pages, delay=args.delay)
