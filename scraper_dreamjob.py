# =============================================================================
# scraper_dreamjob.py — Dreamjob.ma job offers via the public WordPress REST API
# =============================================================================
# Dreamjob.ma is a WordPress site, so its posts are available as JSON at
#   https://www.dreamjob.ma/wp-json/wp/v2/posts
# No HTML parsing or headless browser needed. Its robots.txt only disallows
# /wp-admin/, and we keep the load low (one request per second, small pages).
#
# Each post carries category IDs that tell us the city ("Offres d'Emploi à
# Casablanca"), the sector (children of "Secteur") and public vs private.
#
# Usage:
#   from scraper_dreamjob import scrape_dreamjob
#   df = scrape_dreamjob(max_pages=5, db_path="jobs.db")
# =============================================================================

import html
import logging
import re
import time
from datetime import datetime

import pandas as pd
import requests

from scraper_rekrute import init_database, save_job

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

API = "https://www.dreamjob.ma/wp-json/wp/v2"
EMPLOI_MAROC = 1527        # "Emploi Maroc" — skips guides, news, jobs abroad
SECTEUR_PARENT = 3012      # sector categories are children of "Secteur"
EMPLOI_PUBLIC = 77
PER_PAGE = 50
REQUEST_DELAY = 1.0        # seconds between requests

HEADERS = {
    "User-Agent": "job-market-morocco/1.0 (+https://github.com/Aya04LA/job-market-morocco)",
    "Accept": "application/json",
}

_session = requests.Session()
_session.headers.update(HEADERS)


def _get(path, params):
    resp = _session.get(f"{API}/{path}", params=params, timeout=30)
    time.sleep(REQUEST_DELAY)
    return resp


def _strip_html(raw: str) -> str:
    text = re.sub(r"<(br|/p|/li|/h\d)[^>]*>", "\n", raw or "", flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


_category_cache = {}

def _load_categories(ids):
    """Fetch names/parents for category IDs we haven't seen yet."""
    missing = [i for i in ids if i not in _category_cache]
    for start in range(0, len(missing), 100):
        chunk = missing[start:start + 100]
        resp = _get("categories", {"include": ",".join(map(str, chunk)),
                                   "per_page": 100, "_fields": "id,name,parent"})
        resp.raise_for_status()
        for cat in resp.json():
            _category_cache[cat["id"]] = (html.unescape(cat["name"]), cat["parent"])


def _city(cat_names, title):
    for name in cat_names:
        m = re.match(r"Offres d.Emploi à (.+)", name)
        if m:
            return m.group(1)
    # Titles often end with "... à Casablanca"
    m = re.search(r"\bà ([A-ZÉ][\w\-éèâ]+(?: [A-ZÉ][\w\-éèâ]+)?)\s*$", title)
    return m.group(1) if m else "N/A"


def _sector(cat_ids):
    for cid in cat_ids:
        name, parent = _category_cache.get(cid, ("", 0))
        if parent == SECTEUR_PARENT:
            return re.sub(r"^Offres d.Emploi\s*", "", name)
    if EMPLOI_PUBLIC in cat_ids:
        return "Secteur Public"
    return "N/A"


def _contract(title, description):
    text = f"{title} {description[:600]}".lower()
    for keyword in ("stage", "cdi", "cdd", "intérim", "interim", "freelance", "alternance"):
        if re.search(rf"\b{keyword}\b", text):
            return keyword.upper() if keyword in ("cdi", "cdd") else keyword.capitalize()
    return "N/A"


def _company(title):
    # "Mundiapolis recrute un Chargé des Stages à Casablanca"
    # "Marchica Lagoon Resort lance un nouveau recrutement"
    m = re.match(r"(.+?)\s+(?:recrute|lance|ouvre|offre|annonce|organise)\b", title)
    if m:
        return m.group(1).strip()
    # "Offres de stage rémunéré chez Motherson Kénitra", "4 postes ouverts chez TRIGO Maroc à Kénitra"
    m = re.search(r"\bchez (.+?)(?:\s+(?:à|dans)\s.*)?$", title)
    return m.group(1).strip() if m else "N/A"


def parse_post(post):
    title = html.unescape(post["title"]["rendered"]).strip()
    description = _strip_html(post["content"]["rendered"])
    cat_ids = post.get("categories", [])
    cat_names = [_category_cache.get(c, ("", 0))[0] for c in cat_ids]
    return {
        "source":        "dreamjob",
        "title":         title,
        "company":       _company(title),
        "location":      _city(cat_names, title),
        "contract_type": _contract(title, description),
        "sector":        _sector(cat_ids),
        "experience":    "N/A",
        "education":     "N/A",
        "salary":        "N/A",
        "description":   description or "N/A",
        "skills_raw":    "N/A",
        "url":           post["link"],
        "posted_date":   post["date"][:10],
        "scraped_at":    datetime.now().isoformat(),
    }


def scrape_dreamjob(max_pages=5, db_path="jobs.db"):
    """Fetch the newest `max_pages` × 50 Morocco job posts and store new ones."""
    conn = init_database(db_path)
    logger.info(f"Starting Dreamjob.ma — up to {max_pages} pages × {PER_PAGE} posts (REST API)")

    new_jobs, duplicates = [], 0
    for page in range(1, max_pages + 1):
        try:
            resp = _get("posts", {
                "categories": EMPLOI_MAROC,
                "per_page": PER_PAGE,
                "page": page,
                "orderby": "date",
                "_fields": "id,date,link,title,content,categories",
            })
        except requests.RequestException as e:
            logger.error(f"Page {page}: request failed — {e}")
            break
        if resp.status_code == 400:   # past the last page
            break
        resp.raise_for_status()
        posts = resp.json()
        if not posts:
            break

        _load_categories({c for p in posts for c in p.get("categories", [])})

        page_new = 0
        for post in posts:
            job = parse_post(post)
            if save_job(conn, job):
                new_jobs.append(job)
                page_new += 1
            else:
                duplicates += 1
        logger.info(f"Page {page}/{max_pages}: {page_new} new, {len(posts) - page_new} already stored")

        # Newest-first ordering: a page with nothing new means we've caught up
        if page_new == 0:
            break

    conn.close()
    logger.info(f"Done — {len(new_jobs)} new, {duplicates} duplicates")
    return pd.DataFrame(new_jobs)


if __name__ == "__main__":
    df = scrape_dreamjob(max_pages=1)
    if not df.empty:
        print(df[["title", "company", "location", "sector", "contract_type"]].head(20).to_string())
