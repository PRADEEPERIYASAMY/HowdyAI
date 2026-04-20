"""
crawl_tamu.py - Targeted crawler for key TAMU URLs.
Fetches and cleans important TAMU pages, saves them to data/documents/ for indexing.
"""

import os
import re
import time
import hashlib
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# ── Target seed URLs ────────────────────────────────────────────────────────────
SEED_URLS = [
    # Main
    "https://www.tamu.edu/",
    "https://www.tamu.edu/about/index.html",
    # Admissions
    "https://admissions.tamu.edu/",
    "https://admissions.tamu.edu/freshman/how-to-apply.html",
    "https://admissions.tamu.edu/freshman/requirements.html",
    "https://admissions.tamu.edu/costs/",
    "https://admissions.tamu.edu/transfer/",
    # Financial Aid
    "https://financialaid.tamu.edu/",
    "https://financialaid.tamu.edu/undergraduate/types-of-aid/",
    "https://financialaid.tamu.edu/undergraduate/scholarships/",
    # Academic Catalog - course descriptions
    "https://catalog.tamu.edu/undergraduate/course-descriptions/csce/",
    "https://catalog.tamu.edu/undergraduate/course-descriptions/math/",
    "https://catalog.tamu.edu/undergraduate/course-descriptions/biol/",
    "https://catalog.tamu.edu/undergraduate/course-descriptions/phys/",
    "https://catalog.tamu.edu/undergraduate/course-descriptions/ecen/",
    "https://catalog.tamu.edu/undergraduate/course-descriptions/hist/",
    "https://catalog.tamu.edu/undergraduate/arts-and-sciences/biology/zoology-bs/",
    "https://catalog.tamu.edu/graduate/colleges-schools-interdisciplinary/engineering/computer-science/",
    # Registrar
    "https://registrar.tamu.edu/",
    "https://registrar.tamu.edu/Catalogs,-Policies-Procedures/Academic-Calendar",
    # Campus Life
    "https://studentlife.tamu.edu/",
    "https://msc.tamu.edu/",
    "https://library.tamu.edu/",
    "https://rec.tamu.edu/",
    "https://housing.tamu.edu/",
    "https://dining.tamu.edu/",
    # Departments
    "https://engineering.tamu.edu/cse/index.html",
    "https://engineering.tamu.edu/",
    "https://artsci.tamu.edu/",
    "https://bush.tamu.edu/",
    "https://mays.tamu.edu/",
    "https://veterinary.tamu.edu/",
    # Research / Graduate
    "https://research.tamu.edu/",
    "https://gradcatalog.tamu.edu/",
    # Career
    "https://careercenter.tamu.edu/",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
TIMEOUT = 8
OUT_DIR  = os.path.join(os.path.dirname(__file__), "data", "documents")
os.makedirs(OUT_DIR, exist_ok=True)


def clean_html(soup: BeautifulSoup) -> str:
    """Remove boilerplate and return clean text-preserving HTML."""
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "noscript", "aside", "form", "iframe",
                     "[class~=cookie]", "[class~=banner]"]):
        tag.decompose()
    # Return full body as minimal HTML so DirectoryLoader can parse it
    body = soup.find("body") or soup
    return str(body)


def safe_filename(url: str) -> str:
    domain = urlparse(url).netloc.replace(".", "_")
    path   = re.sub(r"[^a-zA-Z0-9_-]", "_", urlparse(url).path)
    slug   = (domain + path).strip("_")[:80]
    uid    = hashlib.md5(url.encode()).hexdigest()[:6]
    return f"{slug}_{uid}.html"


def crawl(url: str) -> dict | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code != 200:
            print(f"  ✗ {r.status_code}  {url}")
            return None
        soup = BeautifulSoup(r.text, "lxml")
        title = soup.title.string.strip() if soup.title else url
        content = clean_html(soup)
        return {"url": url, "title": title, "html": content}
    except Exception as e:
        print(f"  ✗ ERR  {url}  ({e})")
        return None


def save(doc: dict) -> str:
    fname = safe_filename(doc["url"])
    fpath = os.path.join(OUT_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(f"<!-- SOURCE: {doc['url']} TITLE: {doc['title']} -->\n")
        f.write(doc["html"])
    return fpath


if __name__ == "__main__":
    print(f"Crawling {len(SEED_URLS)} TAMU URLs -> {OUT_DIR}\n")
    saved = 0
    failed = 0
    for i, url in enumerate(SEED_URLS, 1):
        print(f"[{i:02}/{len(SEED_URLS)}] {url}")
        doc = crawl(url)
        if doc:
            path = save(doc)
            print(f"  ✓ Saved: {os.path.basename(path)}  ({len(doc['html'])} chars)")
            saved += 1
        else:
            failed += 1
        time.sleep(0.4)   # be polite — 400ms between requests

    print(f"\n─────────────────────────────────────────")
    print(f"Done! Saved: {saved}   Failed/skipped: {failed}")
    print(f"Documents in: {OUT_DIR}")
    print(f"Next step: python create_database.py --reset")
