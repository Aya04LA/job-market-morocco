# =============================================================================
# EMPLOI.MA SCRAPER — Job Market Intelligence Platform
# =============================================================================
# Uses Playwright (headless Chromium) because emploi.ma blocks plain requests.
#
# ONE-TIME SETUP — run in terminal inside your venv:
#   pip install playwright nest_asyncio
#   playwright install chromium
#
# Usage in notebook:
#   from scraper_emploi_ma import scrape_emploi_ma
#   df = scrape_emploi_ma(max_pages=3)
# =============================================================================

import asyncio
import sys
import random
import logging
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup

# Windows requires ProactorEventLoop for subprocess support (Playwright needs this)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

from scraper_rekrute import init_database, save_job, explore_data

EMPLOI_BASE_URL    = "https://www.emploi.ma"
EMPLOI_LISTING_URL = EMPLOI_BASE_URL + "/recherche-jobs-maroc"


# =============================================================================
# PLAYWRIGHT FETCH
# Works in both plain .py scripts AND Jupyter notebooks.
# Jupyter has a running event loop so sync_playwright raises an error there —
# we use async_playwright + nest_asyncio to handle both cases transparently.
# =============================================================================

async def _fetch_async(url, wait_selector="div.card.card-job", timeout=20000):
    """Async Playwright fetch. Called by _pw_get() — don't call directly."""
    from playwright.async_api import async_playwright, TimeoutError as PWTimeout

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ]
        )
        ctx = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
            locale="fr-FR",
        )
        page = await ctx.new_page()
        html = ""
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            try:
                await page.wait_for_selector(wait_selector, timeout=timeout)
            except PWTimeout:
                logger.warning(f"Selector not found on {url} — using raw HTML")
            await page.wait_for_timeout(random.randint(1500, 2500))
            html = await page.content()
        except Exception as e:
            logger.error(f"Playwright error on {url}: {e}")
        finally:
            await browser.close()

    return html


def _pw_get(url, wait_selector="div.card.card-job", timeout=20000):
    """
    Fetch a URL with headless Chromium.
    Works in plain scripts, Jupyter notebooks, and Windows.
    Runs Playwright in a separate thread so it gets a clean event loop —
    this sidesteps the Windows ProactorEventLoop/Jupyter conflict entirely.
    """
    import concurrent.futures

    def _run():
        # Each thread gets its own fresh event loop — no conflicts
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(_fetch_async(url, wait_selector, timeout))
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run).result()


# =============================================================================
# PARSERS
# =============================================================================

def _txt(tag):
    return tag.get_text(strip=True) if tag else "N/A"

def _after_colon(text):
    return text.split(":", 1)[1].strip() if ":" in text else text.strip()


def parse_emploi_card(card):
    """
    Parse one job card from the Emploi.ma listing page.

    Confirmed HTML structure (from live page):
      <div class="card card-job [featured]">
        <h3><a href="/offre-emploi-maroc/...">Title</a></h3>
        <a class="company-name" href="/recruteur/...">Company</a>
        <ul>
          <li>Niveau d'études requis : <strong>Bac+5</strong></li>
          <li>Niveau d'expérience : <strong>Expérience 2-5 ans</strong></li>
          <li>Contrat proposé : <strong>CDI</strong></li>
          <li>Région de : <strong>Casablanca-Mohammedia</strong></li>
          <li>Compétences clés : <strong>Python - ML</strong></li>
        </ul>
        <time datetime="2026-04-06">06.04.2026</time>
    """
    try:
        title_tag = card.select_one("h3 a")
        if not title_tag:
            return None

        title = _txt(title_tag)
        href  = title_tag.get("href", "")
        url   = (EMPLOI_BASE_URL + href) if href.startswith("/") else href

        company_tag = card.select_one(".company-name")
        company = _txt(company_tag) if company_tag else "N/A"

        location = education = experience = contract_type = skills_raw = "N/A"
        for li in card.select("ul li"):
            raw    = li.get_text(" ", strip=True)
            low    = raw.lower()
            strong = li.select_one("strong")
            val    = _txt(strong) if strong else _after_colon(raw)

            if "région" in low:
                location = val
            elif "niveau d'études" in low or "niveau d´études" in low:
                education = val
            elif "niveau d'expérience" in low:
                experience = val
            elif "contrat" in low:
                contract_type = val
            elif "compétences" in low:
                skills_raw = val

        date_tag    = card.select_one("time")
        posted_date = date_tag.get("datetime", _txt(date_tag)) if date_tag else "N/A"

        return {
            "title": title, "url": url, "company": company,
            "location": location, "contract_type": contract_type,
            "experience": experience, "education": education,
            "skills_raw": skills_raw, "posted_date": posted_date,
        }
    except Exception as e:
        logger.warning(f"Card parse error: {e}")
        return None


def parse_emploi_detail(url):
    """Fetch detail page for description + sector."""
    html = _pw_get(url, wait_selector=".field-name-body, article, .job-description")
    if not html:
        return {}

    soup    = BeautifulSoup(html, "html.parser")
    details = {}

    for li in soup.select("ul li, .field-label, .criteria li"):
        raw = li.get_text(" ", strip=True)
        if any(k in raw.lower() for k in ["secteur", "domaine d'activité"]):
            strong = li.select_one("strong")
            details["sector"] = _txt(strong) if strong else _after_colon(raw)
            break
    details.setdefault("sector", "N/A")

    for li in soup.select("ul li"):
        if any(k in li.get_text().lower() for k in ["salaire", "rémunération"]):
            details["salary"] = _after_colon(li.get_text(" ", strip=True))
            break
    details.setdefault("salary", "N/A")

    for sel in [".field-name-body .field-items", ".field-name-body",
                "#job-description", ".job-description", "article .field-items"]:
        tag = soup.select_one(sel)
        if tag:
            details["description"] = tag.get_text(" ", strip=True)
            break
    details.setdefault("description", "N/A")

    return details


# =============================================================================
# MAIN SCRAPER
# =============================================================================

def scrape_emploi_ma(max_pages=5, db_path="jobs.db", fetch_details=True):
    """
    Scrape Emploi.ma job listings using headless Chromium.

    Args:
        max_pages:     number of listing pages (~25 jobs each)
        db_path:       shared SQLite DB (same file as Rekrute)
        fetch_details: False = skip detail pages, much faster,
                       you lose description + sector but keep everything else

    Emploi.ma pagination is 0-indexed:
        Page 1 → /recherche-jobs-maroc
        Page 2 → /recherche-jobs-maroc?page=1
        Page N → /recherche-jobs-maroc?page=N-1
    """
    conn = init_database(db_path)
    session_jobs, new_count, skip_count = [], 0, 0

    logger.info(f"Starting Emploi.ma — {max_pages} pages (Playwright headless)")

    for page_idx in range(max_pages):
        url = EMPLOI_LISTING_URL if page_idx == 0 else f"{EMPLOI_LISTING_URL}?page={page_idx}"
        logger.info(f"Page {page_idx+1}/{max_pages}: {url}")

        html = _pw_get(url)
        if not html:
            continue

        soup  = BeautifulSoup(html, "html.parser")
        cards = soup.select("div.card.card-job")

        if not cards:
            logger.warning(f"No cards on page {page_idx+1} — saving debug HTML")
            with open(f"debug_emploi_p{page_idx+1}.html", "w", encoding="utf-8") as f:
                f.write(html)
            continue

        logger.info(f"  {len(cards)} cards")

        for card in cards:
            basic = parse_emploi_card(card)
            if not basic or not basic.get("url"):
                continue

            logger.info(f"  → {basic['title'][:55]}")
            detail = parse_emploi_detail(basic["url"]) if fetch_details else {}

            job = {
                "source":        "emploi_ma",
                "title":         basic.get("title",         "N/A"),
                "company":       basic.get("company",       "N/A"),
                "location":      basic.get("location",      "N/A"),
                "contract_type": basic.get("contract_type", "N/A"),
                "sector":        detail.get("sector",       "N/A"),
                "experience":    basic.get("experience",    "N/A"),
                "education":     basic.get("education",     "N/A"),
                "salary":        detail.get("salary",       "N/A"),
                "description":   detail.get("description",  "N/A"),
                "skills_raw":    basic.get("skills_raw",    "N/A"),
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
        csv_path = f"emploi_ma_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"CSV: {csv_path}")

    conn.close()
    return df


if __name__ == "__main__":
    df = scrape_emploi_ma(max_pages=3)
    if not df.empty:
        explore_data()
