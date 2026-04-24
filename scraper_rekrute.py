# =============================================================================
# REKRUTE SCRAPER — Job Market Intelligence Platform
# =============================================================================

import requests
from bs4 import BeautifulSoup
import sqlite3
import pandas as pd
import time
import random
import logging
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

REKRUTE_BASE_URL  = "https://www.rekrute.com"
REKRUTE_LISTING_URL = REKRUTE_BASE_URL + "/offres.html?s=3&p={page}&o=1"

# =============================================================================
# DATABASE
# =============================================================================

def init_database(db_path="jobs.db"):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            source        TEXT,
            title         TEXT,
            company       TEXT,
            location      TEXT,
            contract_type TEXT,
            sector        TEXT,
            experience    TEXT,
            education     TEXT,
            salary        TEXT,
            description   TEXT,
            skills_raw    TEXT,
            url           TEXT UNIQUE,
            posted_date   TEXT,
            scraped_at    TEXT
        )
    """)
    conn.commit()
    logger.info(f"DB ready: {db_path}")
    return conn


def save_job(conn, job: dict):
    cur = conn.execute("""
        INSERT OR IGNORE INTO jobs
            (source, title, company, location, contract_type, sector,
             experience, education, salary, description, skills_raw,
             url, posted_date, scraped_at)
        VALUES
            (:source, :title, :company, :location, :contract_type, :sector,
             :experience, :education, :salary, :description, :skills_raw,
             :url, :posted_date, :scraped_at)
    """, job)
    conn.commit()
    return cur.rowcount


# =============================================================================
# HTTP — persistent session with warm-up
# =============================================================================

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]

_session = requests.Session()

def _get_headers(referer=None):
    h = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    if referer:
        h["Referer"] = referer
    return h

def _warm_up(base_url):
    try:
        _session.get(base_url, headers=_get_headers(), timeout=15)
        time.sleep(random.uniform(2, 3))
        logger.info("Session warmed up")
    except Exception as e:
        logger.warning(f"Warm-up failed (non-fatal): {e}")

def fetch_page(url, retries=3, referer=None):
    for attempt in range(retries):
        try:
            time.sleep(random.uniform(1.5, 3.0))
            r = _session.get(url, headers=_get_headers(referer), timeout=15)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt+1} failed for {url}: {e}")
            time.sleep(5 * (attempt + 1))
    logger.error(f"All {retries} attempts failed: {url}")
    return None


# =============================================================================
# PARSERS
# =============================================================================

def _txt(tag):
    return tag.get_text(strip=True) if tag else "N/A"

def _after_colon(text):
    return text.split(":", 1)[1].strip() if ":" in text else text.strip()

def _extract_company_from_description(description):
    """
    Extract company name from Rekrute's description text block.

    Rekrute listing cards intentionally hide company names — they only appear
    on the detail page, either as:
      1. An <a class="recruteur"> link (primary — handled in parse_job_detail)
      2. The "Entreprise : NAME est/recrute..." text block (this function)

    Returns "N/A (confidentiel)" when the post is anonymised.
    """
    if not description or description == "N/A":
        return "N/A"

    # The description is joined with " | " — the first block is the company intro
    block = description.split("|")[0].strip()
    # Strip the "Entreprise :" label
    block = re.sub(r'^Entreprise\s*[:\-]\s*', '', block, flags=re.IGNORECASE).strip()

    # Confidential / generic openers → company deliberately hidden
    if re.match(
        r'^(vous\b|rejoignez\b|nous\b|notre\b|le\b|la\b|les\b|un\b|une\b|'
        r'créé\b|fondé\b|depuis\b|leader\b|dans\b|spécialis|entreprise\b|'
        r'société\b|cabinet\b|groupe\b)',
        block, re.IGNORECASE
    ):
        return "N/A (confidentiel)"

    # Stop at first comma or known stop-verb
    m = re.match(
        r'^([^,\n\r]+?)(?:,|\s+(?:est\b|recrute\b|recherche\b|s\'est\b|a\s+été\b|filiale\b))',
        block, re.IGNORECASE
    )
    if m:
        candidate = m.group(1).strip().rstrip(",.;")
        if 2 <= len(candidate) <= 80:
            return candidate

    # Fallback: leading capitalised words (proper-noun heuristic)
    words = block.split()[:6]
    name_words = []
    for w in words:
        if re.match(r'^[A-Z\d]', w):
            name_words.append(w.rstrip(".,"))
        else:
            break
    if name_words:
        return " ".join(name_words)

    return "N/A"


def parse_job_card(card):
    """
    Parse one job card from the Rekrute listing page.

    The listing page uses <li class="post-id ..."> cards.
    Inside each card:
      - Title is in <h3><a> or <a class="titreJob">
      - Company/location are in <span> tags or <li> items with icons
      - Structured metadata (sector, contract, experience) in <ul><li> text
      - Publication date: "Publication : du DD/MM/YYYY au DD/MM/YYYY"

    If parsing fails, a debug HTML file will be saved by the main scraper.
    """
    try:
        # Title + URL
        title_tag = (
            card.select_one("h3 a")
            or card.select_one("h2 a")
            or card.select_one("a.titreJob")
            or card.select_one(".post-name a")
        )
        if not title_tag:
            return None

        title = _txt(title_tag)
        href  = title_tag.get("href", "")
        url   = (REKRUTE_BASE_URL + href) if href.startswith("/") else (href if href.startswith("http") else None)

        # Company — class="recruteur" or icon la-building
        company = "N/A"
        for sel in [".recruteur", ".company-name", "span.recruteur"]:
            t = card.select_one(sel)
            if t:
                company = _txt(t)
                break
        if company == "N/A":
            for li in card.select("li"):
                if li.select_one("i.la-building, i[class*='building']"):
                    company = li.get_text(strip=True)
                    break

        # Location — class="ville" or icon la-map-marker
        location = "N/A"
        for sel in [".ville", ".location", "span.ville"]:
            t = card.select_one(sel)
            if t:
                location = _txt(t)
                break
        if location == "N/A":
            for li in card.select("li"):
                if li.select_one("i.la-map-marker, i[class*='map']"):
                    location = li.get_text(strip=True)
                    break

        # Structured fields from <li> text
        contract_type = sector = experience = education = posted_date = "N/A"

        for li in card.select("li"):
            raw = li.get_text(" ", strip=True)
            low = raw.lower()
            if any(k in low for k in ["contrat", "type de contrat"]):
                contract_type = _after_colon(raw)
            elif any(k in low for k in ["secteur", "domaine d'activité"]):
                sector = _after_colon(raw)
            elif "expérience" in low and experience == "N/A":
                experience = _after_colon(raw)
            elif any(k in low for k in ["formation", "niveau d'étude", "diplôme"]):
                education = _after_colon(raw)
            elif "publication" in low:
                m = re.search(r"du\s+(\d{2}/\d{2}/\d{4})", raw, re.IGNORECASE)
                posted_date = m.group(1) if m else _after_colon(raw)

        return {
            "title": title, "url": url, "company": company,
            "location": location, "contract_type": contract_type,
            "sector": sector, "experience": experience,
            "education": education, "posted_date": posted_date,
        }
    except Exception as e:
        logger.warning(f"Card parse error: {e}")
        return None


def parse_job_detail(url, referer=None):
    """
    Fetch detail page for full description, skills, salary and structured fields.

    Rekrute detail page structure (verified against real HTML):
      - Sector    → <h2 class="h2italic">
      - Location  → <ul class="featureInfo"><li title="Région">  text after "sur"
      - Experience→ <ul class="featureInfo"><li title="Expérience requise">
      - Education → <ul class="featureInfo"><li title="Niveau d'étude et formation">
      - Contract  → <ul class="featureInfo"><li title="Type de contrat"> → span.tagContrat
      - Skills    → <span class="tagSkills">  (deduplicated)
      - Desc      → <div class="col-md-12 blc"> blocks: Entreprise / Poste / Profil
    """
    r = fetch_page(url, referer=referer)
    if not r:
        return {}

    soup    = BeautifulSoup(r.text, "html.parser")
    details = {}

    # Company — <a class="recruteur"> is the most reliable source on the detail page.
    # Rekrute hides company names on listing cards but renders them here.
    company_tag = (
        soup.select_one("a.recruteur")
        or soup.select_one("span.recruteur")
        or soup.select_one(".company-name a")
        or soup.select_one(".company-name")
    )
    if company_tag:
        details["company"] = _txt(company_tag)

    # Sector
    sector_tag = soup.select_one("h2.h2italic")
    if sector_tag:
        details["sector"] = sector_tag.get_text(" ", strip=True)

    # featureInfo structured fields
    for li in soup.select("ul.featureInfo li"):
        title_attr = li.get("title", "")
        raw = re.sub(r'\s+', ' ', li.get_text(" ", strip=True)).strip()

        if title_attr == "Expérience requise":
            details["experience"] = raw

        elif title_attr == "Région":
            # e.g. "2 poste(s) sur Casablanca et région - Maroc"
            m = re.search(r'\bsur\s+(.+)', raw, re.IGNORECASE)
            details["location"] = m.group(1).strip() if m else raw

        elif "Niveau d" in title_attr or "formation" in title_attr.lower():
            details["education"] = raw

        elif title_attr == "Type de contrat":
            span = li.select_one("span.tagContrat")
            details["contract_type"] = span.get_text(strip=True) if span else raw

    # Skills — deduplicated (same tags appear twice in page source)
    seen, skills = set(), []
    for tag in soup.select("span.tagSkills"):
        s = tag.get_text(strip=True)
        if s and s not in seen:
            seen.add(s)
            skills.append(s)
    if skills:
        details["skills_raw"] = " | ".join(skills)

    # Description — only Entreprise / Poste / Profil / Mission blocks
    desc_parts = []
    for blc in soup.select("div.col-md-12.blc"):
        h2 = blc.select_one("h2")
        if h2 and any(k in h2.get_text().lower()
                      for k in ["entreprise", "poste", "profil", "mission", "description"]):
            desc_parts.append(blc.get_text(" ", strip=True))
    details["description"] = " | ".join(desc_parts) if desc_parts else "N/A"

    # Salary
    for node in soup.find_all(string=re.compile(r"salaire|rémunération", re.I)):
        parent = node.parent
        if parent:
            details["salary"] = re.sub(r'\s+', ' ', parent.get_text(" ", strip=True)).strip()
            break
    details.setdefault("salary", "N/A")

    return details


# =============================================================================
# MAIN SCRAPER
# =============================================================================

def scrape_rekrute(max_pages=5, db_path="jobs.db"):
    conn = init_database(db_path)
    session_jobs, new_count, skip_count = [], 0, 0

    logger.info(f"Starting Rekrute — {max_pages} pages")
    _warm_up(REKRUTE_BASE_URL)

    for page_num in range(1, max_pages + 1):
        url     = REKRUTE_LISTING_URL.format(page=page_num)
        referer = REKRUTE_LISTING_URL.format(page=page_num-1) if page_num > 1 else REKRUTE_BASE_URL
        logger.info(f"Page {page_num}/{max_pages}: {url}")

        r = fetch_page(url, referer=referer)
        if not r:
            continue

        soup  = BeautifulSoup(r.text, "html.parser")
        cards = (
            soup.select("li.post-id")
            or soup.select("li[class*='post-']")
            or soup.select(".job-listing-sec li")
            or soup.select(".job-item")
        )

        if not cards:
            logger.warning(f"No cards on page {page_num} — saving debug HTML")
            with open(f"debug_rekrute_p{page_num}.html", "w", encoding="utf-8") as f:
                f.write(r.text)
            continue

        logger.info(f"  {len(cards)} cards")

        for card in cards:
            basic = parse_job_card(card)
            if not basic or not basic.get("url"):
                continue

            logger.info(f"  → {basic['title'][:55]}")
            detail = parse_job_detail(basic["url"], referer=url)

            # Company: detail-page HTML link > description text extraction > card fallback
            description_text = detail.get("description", "N/A")
            company = (
                detail.get("company")
                or _extract_company_from_description(description_text)
                or basic.get("company", "N/A")
            )

            job = {
                "source":        "rekrute",
                "title":         basic.get("title",         "N/A"),
                "company":       company,
                "location":      detail.get("location")      or basic.get("location",      "N/A"),
                "contract_type": detail.get("contract_type") or basic.get("contract_type", "N/A"),
                "sector":        detail.get("sector")        or basic.get("sector",        "N/A"),
                "experience":    detail.get("experience")    or basic.get("experience",    "N/A"),
                "education":     detail.get("education")     or basic.get("education",     "N/A"),
                "salary":        detail.get("salary",        "N/A"),
                "description":   detail.get("description",  "N/A"),
                "skills_raw":    detail.get("skills_raw",   "N/A"),
                "url":           basic.get("url"),
                "posted_date":   basic.get("posted_date",   "N/A"),
                "scraped_at":    datetime.now().isoformat(),
            }

            if save_job(conn, job):
                new_count += 1
                session_jobs.append(job)
            else:
                skip_count += 1

    logger.info(f"Done — {new_count} new, {skip_count} duplicates")

    df = pd.DataFrame(session_jobs) if session_jobs else pd.DataFrame()
    if not df.empty:
        csv_path = f"rekrute_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"CSV: {csv_path}")

    conn.close()
    return df


# =============================================================================
# EXPLORATION HELPER (shared with emploi_ma scraper)
# =============================================================================

def explore_data(db_path="jobs.db"):
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql("SELECT * FROM jobs", conn)
    conn.close()

    print(f"\n{'='*50}")
    print(f"Total jobs       : {len(df)}")
    print(f"Sources          : {df['source'].value_counts().to_dict()}")
    print(f"\nTop locations:\n{df['location'].value_counts().head(10)}")
    print(f"\nTop sectors:\n{df['sector'].value_counts().head(10)}")
    print(f"\nContract types:\n{df['contract_type'].value_counts().head(10)}")
    print(f"\nSalary available : {df['salary'].ne('N/A').sum()} / {len(df)}")
    print(f"{'='*50}\n")
    return df


if __name__ == "__main__":
    df = scrape_rekrute(max_pages=3)
    if not df.empty:
        explore_data()
        print(df[["title", "company", "location", "contract_type"]].head(10))
