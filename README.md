# 🇲🇦 Job Market Morocco

An end-to-end data pipeline that scrapes Moroccan job boards every week, extracts skills with NLP,
predicts missing sectors with a machine-learning model, and serves everything in an interactive
Streamlit dashboard with a CV analyzer and an AI career chatbot.

**Live demo:** [Hugging Face Space](https://huggingface.co/spaces/yaya11111111111111/job-market-morocco)

## Features

- **Automated scraping** of [Rekrute](https://www.rekrute.com) (requests + BeautifulSoup) and
  [Emploi.ma](https://www.emploi.ma) (Playwright), deduplicated into a SQLite database
- **Skill extraction** with a spaCy `PhraseMatcher` over French job descriptions
- **Sector classification**: TF-IDF + Logistic Regression fills in jobs posted without a sector
- **Experiment tracking** with MLflow (cross-validated F1, skill coverage, sector counts per run)
- **Dashboard** (in French) with five pages:
  - Market overview: top skills, sectors, cities
  - Job explorer with filters and free-text search
  - CV analysis: upload a PDF and see which in-demand skills you have and which you're missing
  - ML performance: model metrics from MLflow
  - Chatbot (Groq LLM) for market questions and CV advice
- **Weekly CI run**: GitHub Actions scrapes, retrains, and publishes fresh data to the Space

## Architecture

```
 Rekrute ──┐                                                 ┌──> MLflow (mlruns/)
           ├─> scrapers ─> jobs.db ─> cleaning + NLP + ML ───┤
 Emploi.ma ┘   (SQLite)                                      └──> jobs_nlp.csv
                                                                  skills_frequency.csv
                                                                        │
                                                       Hugging Face Space (dashboard.py)
```

| File | Role |
| --- | --- |
| `scraper_rekrute.py`, `scraper_emploi_ma.py` | Scrape listings into `jobs.db` |
| `data_cleaning.py` | Normalize titles, cities, contracts, salaries |
| `nlp_pipeline.py` | Skill extraction, sector classifier, skill frequencies |
| `mlflow_tracking.py` | Runs the NLP pipeline as a tracked MLflow experiment |
| `run_pipeline.py` | Orchestrates scrape → NLP → publish |
| `dashboard.py` | Streamlit app |
| `main.ipynb` | Exploration notebook |

## Run locally

Requires Python 3.11.

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

Open the dashboard on the data included in the repo:

```bash
streamlit run dashboard.py
```

Run the full pipeline (scraping takes a while; `MAX_PAGES` limits pages per site):

```bash
MAX_PAGES=2 python run_pipeline.py
```

### Environment variables

| Variable | Used by | Purpose |
| --- | --- | --- |
| `MAX_PAGES` | pipeline | Pages scraped per site (default 5) |
| `DB_PATH` | pipeline | SQLite path (default `jobs.db`) |
| `HF_TOKEN` | pipeline | Push results to the Space (skipped if unset) |
| `HF_SPACE_REPO` | pipeline | Target Space |
| `GROQ_API_KEY` | dashboard | Enables the chatbot |

## Automation

`.github/workflows/weekly_pipeline.yml` runs every Monday at 01:00 UTC (and on demand from the
Actions tab with a custom page count). It keeps `jobs.db` in the Actions cache so listings
accumulate across weeks, and uploads the output CSVs as build artifacts.

## Tech stack

Python · pandas · scikit-learn · spaCy · MLflow · Playwright · BeautifulSoup · Streamlit · Plotly ·
Groq · GitHub Actions · Hugging Face Spaces
