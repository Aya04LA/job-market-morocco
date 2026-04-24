# =============================================================================
# DATA CLEANING — Job Market Intelligence Platform
# =============================================================================
# WHY clean before NLP?
# NLP models are sensitive to noise. If "Casablanca et région - Maroc" and
# "Casablanca-Mohammedia" are treated as different cities, your location-based
# analysis will be garbage. Same for sectors. Clean data = trustworthy insights.
#
# This module handles:
#   1. Location normalization  → "Casablanca et région - Maroc" → "Casablanca"
#   2. Sector normalization    → messy strings → clean categories
#   3. Contract type cleanup   → "CDI " / "cdi" / "Contrat CDI" → "CDI"
#   4. Salary parsing          → "15 000 - 20 000 DH" → low=15000, high=20000
#   5. Salary bucket labeling  → for classification model (Bas/Moyen/Élevé)
#   6. Description cleaning    → strip HTML artifacts, normalize whitespace
# =============================================================================

import re
import sqlite3
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# 1. LOCATION NORMALIZATION
# Strategy: keyword matching — check if any known city name appears in the
# raw location string, return the clean name. Handles 95%+ of cases.
# =============================================================================

# Maps keywords (lowercase) → clean city name
LOCATION_MAP = {
    "casablanca":   "Casablanca",
    "mohammedia":   "Casablanca",   # same metro area
    "rabat":        "Rabat",
    "salé":         "Rabat",
    "sale":         "Rabat",
    "témara":       "Rabat",
    "temara":       "Rabat",
    "kénitra":      "Kénitra",
    "kenitra":      "Kénitra",
    "tanger":       "Tanger",
    "tétouan":      "Tanger",
    "tetouan":      "Tanger",
    "marrakech":    "Marrakech",
    "fès":          "Fès",
    "fes":          "Fès",
    "meknès":       "Meknès",
    "meknes":       "Meknès",
    "agadir":       "Agadir",
    "oujda":        "Oujda",
    "laâyoune":     "Laâyoune",
    "laayoune":     "Laâyoune",
    "el jadida":    "El Jadida",
    "settat":       "Settat",
    "béni mellal":  "Béni Mellal",
    "beni mellal":  "Béni Mellal",
    "nador":        "Nador",
    "taza":         "Taza",
    "safi":         "Safi",
    "essaouira":    "Essaouira",
}

def extract_city_from_title(title: str) -> str:
    """
    Many Rekrute titles embed the city: "Technicien SI | Rabat (Maroc)"
    Extract the part between | and (Maroc) and try to normalize it.
    Returns "" if nothing useful found.
    """
    if not title:
        return ""
    # Pattern: "| CityName (Maroc)" at end of title
    m = re.search(r'\|\s*([^|]+?)\s*\(maroc\)', title, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        # Try to normalize the extracted city
        for keyword, clean_name in LOCATION_MAP.items():
            if keyword in candidate.lower():
                return clean_name
        return candidate
    return ""


def normalize_location(raw: str, title: str = "") -> str:
    """
    Convert messy location string to clean city name.
    Falls back to extracting city from job title when raw location
    is a count like "1 poste(s)".

    Examples:
      "Casablanca et région - Maroc"  → "Casablanca"
      "Rabat-Salé-Kénitra"            → "Rabat"
      "1 poste(s) - Maroc"            → extracted from title
      "Tout le Maroc - Maroc"         → "Tout le Maroc"
    """
    if not raw or raw == "N/A":
        return "Non spécifié"

    low = raw.lower()

    # "tout le maroc" or generic national postings
    if any(k in low for k in ["tout le maroc", "maroc entier", "national"]):
        return "Tout le Maroc"

    # Remote
    if "remote" in low or "télétravail" in low or "teletravail" in low:
        return "Télétravail"

    # Looks like a count, not a city → try title fallback
    if re.match(r'^\d+\s*poste', low):
        from_title = extract_city_from_title(title)
        return from_title if from_title else "Non spécifié"

    # Match against known cities
    for keyword, clean_name in LOCATION_MAP.items():
        if keyword in low:
            return clean_name

    # Fallback: strip "- Maroc" suffix
    cleaned = re.sub(r'\s*-\s*maroc\s*$', '', raw, flags=re.IGNORECASE).strip()
    return cleaned if cleaned else "Non spécifié"


# =============================================================================
# 2. SECTOR NORMALIZATION
# Rekrute sectors come as "LABEL - Secteur NAME" or just messy text.
# We map them to clean business categories that make sense on a dashboard.
# =============================================================================

SECTOR_MAP = {
    # IT & Tech
    "informatique":         "IT & Tech",
    "électronique":         "IT & Tech",
    "electronique":         "IT & Tech",
    "télécommunication":    "IT & Tech",
    "telecommunication":    "IT & Tech",
    "digital":              "IT & Tech",
    "numérique":            "IT & Tech",
    "logiciel":             "IT & Tech",
    "software":             "IT & Tech",
    "data":                 "IT & Tech",
    "ia":                   "IT & Tech",
    "intelligence artificielle": "IT & Tech",

    # Finance & Banking
    "banque":               "Finance & Banque",
    "finance":              "Finance & Banque",
    "assurance":            "Finance & Banque",
    "comptabilité":         "Finance & Banque",
    "audit":                "Finance & Banque",
    "immobilier":           "Finance & Banque",

    # Industry & Engineering
    "industrie":            "Industrie & Ingénierie",
    "ingénierie":           "Industrie & Ingénierie",
    "btp":                  "Industrie & Ingénierie",
    "construction":         "Industrie & Ingénierie",
    "énergie":              "Industrie & Ingénierie",
    "energie":              "Industrie & Ingénierie",
    "mines":                "Industrie & Ingénierie",
    "chimie":               "Industrie & Ingénierie",
    "agroalimentaire":      "Industrie & Ingénierie",
    "agriculture":          "Industrie & Ingénierie",
    "automobile":           "Industrie & Ingénierie",
    "aéronautique":         "Industrie & Ingénierie",

    # Commerce & Marketing
    # NOTE: "commercial" alone is too broad (catches "Responsable Commercial" in IT firms)
    # so we match the sector label specifically, not job titles
    "commerce":             "Commerce & Marketing",
    "marketing":            "Commerce & Marketing",
    "grande distribution":  "Commerce & Marketing",
    "vente et distribution": "Commerce & Marketing",
    "communication":        "Commerce & Marketing",
    "publicité":            "Commerce & Marketing",

    # HR & Admin
    "ressources humaines":  "RH & Administration",
    "rh":                   "RH & Administration",
    "administration":       "RH & Administration",
    "secrétariat":          "RH & Administration",
    "juridique":            "RH & Administration",
    "droit":                "RH & Administration",

    # Health
    "santé":                "Santé",
    "médical":              "Santé",
    "pharmacie":            "Santé",
    "médecin":              "Santé",

    # Call Center / BPO
    "centre d'appels":      "Centre d'Appels / BPO",
    "call center":          "Centre d'Appels / BPO",
    "bpo":                  "Centre d'Appels / BPO",
    "relation client":      "Centre d'Appels / BPO",

    # Education
    "enseignement":         "Éducation & Formation",
    "éducation":            "Éducation & Formation",
    "formation":            "Éducation & Formation",

    # Logistics
    "logistique":           "Logistique & Transport",
    "transport":            "Logistique & Transport",
    "supply chain":         "Logistique & Transport",
}

def normalize_sector(raw: str) -> str:
    """
    Convert messy sector string to a clean business category.

    Examples:
      "Informatique / Electronique - Secteur Informatique" → "IT & Tech"
      "Secteur d´activité"                                  → "Non spécifié"
      "Banque / Assurance - Secteur Finance"                → "Finance & Banque"
    """
    if not raw or raw == "N/A":
        return "Non spécifié"

    # Generic label with no real value
    if re.match(r"^secteur\s+d[´'`]activit", raw, re.IGNORECASE):
        return "Non spécifié"

    low = raw.lower()
    for keyword, clean_name in SECTOR_MAP.items():
        if keyword in low:
            return clean_name

    return "Autre"


# =============================================================================
# 3. CONTRACT TYPE NORMALIZATION
# =============================================================================

CONTRACT_MAP = {
    "cdi":      "CDI",
    "cdd":      "CDD",
    "stage":    "Stage",
    "intérim":  "Intérim",
    "interim":  "Intérim",
    "freelance": "Freelance",
    "independant": "Freelance",
    "alternance": "Alternance",
}

def normalize_contract(raw: str) -> str:
    if not raw or raw == "N/A":
        return "Non spécifié"
    low = raw.lower()
    for keyword, clean in CONTRACT_MAP.items():
        if keyword in low:
            return clean
    return "Autre"


# =============================================================================
# 4. SALARY PARSING
# WHY parse salary? We need numeric values for the ML model.
# Salaries in Moroccan job posts look like:
#   "15 000 - 20 000 DH/mois"
#   "Entre 8000 et 12000 MAD"
#   "NC" / "Non communiqué" / "N/A"
# =============================================================================

def _looks_like_salary(raw: str) -> bool:
    if not raw or raw in ("N/A", "NC", "Non communiqué", "Non spécifié"):
        return False
    # Must have BOTH a 4-6 digit number AND a currency word
    # Use word boundary for "dh" to avoid matching inside words
    has_number   = bool(re.search(r'\b\d{4,6}\b', raw))
    has_currency = bool(re.search(r'\bdh\b|\bdhs\b|\bmad\b|\bdirham\b', raw, re.IGNORECASE))
    return has_number and has_currency


def parse_salary(raw: str):
    """
    Extract (salary_low, salary_high, salary_mid) from raw salary string.
    Returns (None, None, None) when salary is not available OR when the
    field doesn't actually look like a salary (avoids false positives).
    All values in MAD/month.

    WHY return midpoint? The ML model needs a single target value.
    Midpoint is the standard approach for range regression.
    """
    if not _looks_like_salary(raw):
        return None, None, None

    # Remove thousands separators and currency labels, keep digits and separators
    numbers = re.findall(r'\d{4,6}', raw.replace(' ', '').replace('\u202f', ''))
    numbers = [int(n) for n in numbers if 1000 <= int(n) <= 200000]

    if len(numbers) >= 2:
        low, high = min(numbers[:2]), max(numbers[:2])
        return low, high, (low + high) / 2
    elif len(numbers) == 1:
        return numbers[0], numbers[0], float(numbers[0])

    return None, None, None


def label_salary_bucket(mid) -> str:
    import math
    if mid is None or (isinstance(mid, float) and math.isnan(mid)):
        return "Non communiqué"
    if mid < 6000:
        return "Bas"
    elif mid <= 15000:
        return "Moyen"
    else:
        return "Élevé"


# =============================================================================
# 5. DESCRIPTION CLEANING
# Raw descriptions contain HTML artifacts, repeated whitespace, pipe separators
# from our scraper, and sometimes binary garbage. We want clean plain text
# for the NLP model.
# =============================================================================

def clean_description(raw: str) -> str:
    """
    Clean raw description text for NLP processing.

    Steps:
      1. Remove HTML tags (in case any slipped through)
      2. Normalize our " | " block separators to newlines
      3. Collapse whitespace
      4. Strip leading/trailing junk
    """
    if not raw or raw == "N/A":
        return ""

    text = raw
    text = re.sub(r'<[^>]+>', ' ', text)           # remove HTML tags
    text = re.sub(r'\|', '\n', text)               # pipe → newline
    text = re.sub(r'[^\w\s\-\.,;:()\'\"/\n]', ' ', text)  # keep useful punctuation
    text = re.sub(r'[ \t]+', ' ', text)            # collapse spaces
    text = re.sub(r'\n{3,}', '\n\n', text)         # max 2 consecutive newlines
    text = text.strip()

    return text


# =============================================================================
# 6. MAIN CLEANING PIPELINE
# Applies all the above to a full DataFrame loaded from SQLite.
# =============================================================================

def clean_jobs_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to a raw jobs DataFrame.
    Returns a new cleaned DataFrame — original is not modified.
    """
    logger.info(f"Starting cleaning on {len(df)} rows...")
    cleaned = df.copy()

    # Location — pass title as fallback for "1 poste(s)" values
    cleaned["location_clean"] = cleaned.apply(
        lambda row: normalize_location(row["location"], row.get("title", "")), axis=1
    )
    logger.info("✓ Locations normalized")

    # Sector
    cleaned["sector_clean"] = cleaned["sector"].apply(normalize_sector)
    logger.info("✓ Sectors normalized")

    # Contract
    cleaned["contract_clean"] = cleaned["contract_type"].apply(normalize_contract)
    logger.info("✓ Contract types normalized")

    # Salary
    salary_parsed = cleaned["salary"].apply(parse_salary)
    cleaned["salary_low"]    = salary_parsed.apply(lambda x: x[0])
    cleaned["salary_high"]   = salary_parsed.apply(lambda x: x[1])
    cleaned["salary_mid"]    = salary_parsed.apply(lambda x: x[2])
    cleaned["salary_bucket"] = cleaned["salary_mid"].apply(label_salary_bucket)
    logger.info(f"✓ Salaries parsed — {cleaned['salary_mid'].notna().sum()} with numeric values")

    # Description
    cleaned["description_clean"] = cleaned["description"].apply(clean_description)
    logger.info("✓ Descriptions cleaned")

    # Drop rows with no title or no URL (useless records)
    before = len(cleaned)
    cleaned = cleaned[cleaned["title"].notna() & (cleaned["title"] != "N/A")]
    cleaned = cleaned[cleaned["url"].notna()]
    logger.info(f"✓ Dropped {before - len(cleaned)} rows with missing title/URL")

    # Reset index
    cleaned = cleaned.reset_index(drop=True)
    logger.info(f"Cleaning complete — {len(cleaned)} clean records")

    return cleaned


def load_and_clean(db_path="jobs.db") -> pd.DataFrame:
    """
    One-liner to load from DB and return a cleaned DataFrame.
    This is what the NLP pipeline will call.
    """
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql("SELECT * FROM jobs", conn)
    conn.close()

    return clean_jobs_dataframe(df)


def cleaning_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame):
    """Print a before/after comparison — useful to paste in your report."""
    print("\n" + "="*55)
    print("CLEANING REPORT")
    print("="*55)
    print(f"Rows before : {len(df_raw)}")
    print(f"Rows after  : {len(df_clean)}")
    print(f"\nLocation distribution (clean):")
    print(df_clean["location_clean"].value_counts().head(10).to_string())
    print(f"\nSector distribution (clean):")
    print(df_clean["sector_clean"].value_counts().to_string())
    print(f"\nContract types (clean):")
    print(df_clean["contract_clean"].value_counts().to_string())
    print(f"\nSalary buckets:")
    print(df_clean["salary_bucket"].value_counts().to_string())
    print("="*55 + "\n")


# =============================================================================
# RUN IN VS CODE / TERMINAL:
#   python data_cleaning.py
#
# OR import in your main notebook:
#   from data_cleaning import load_and_clean, cleaning_report
#   df_clean = load_and_clean("jobs.db")
#   cleaning_report(df_raw, df_clean)
# =============================================================================

if __name__ == "__main__":
    import sqlite3

    conn    = sqlite3.connect("jobs.db")
    df_raw  = pd.read_sql("SELECT * FROM jobs", conn)
    conn.close()

    df_clean = clean_jobs_dataframe(df_raw)
    cleaning_report(df_raw, df_clean)

    # Save cleaned version to CSV for inspection
    df_clean.to_csv("jobs_cleaned.csv", index=False, encoding="utf-8-sig")
    logger.info("Saved to jobs_cleaned.csv")

    # Preview
    print(df_clean[["title", "location_clean", "sector_clean",
                     "contract_clean", "salary_bucket"]].head(15).to_string())
