import logging, os, sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def push_to_huggingface(files):
    try:
        from huggingface_hub import HfApi, login, create_repo
        
        # 1. Fetch secrets from environment variables
        hf_token = os.environ.get("HF_TOKEN")
        hf_repo = os.environ.get("HF_SPACE_REPO", "yaya11111111111111/job-market-morocco")
        
        if not hf_token:
            logger.warning("HF_TOKEN not set - skipping")
            return False
        
        # 2. Force authentication globally
        logger.info("Authenticating with Hugging Face token...")
        login(token=hf_token)
        
        # 3. Defensive check: verify/create the target space
        try:
            create_repo(
                repo_id=hf_repo,
                repo_type="space",
                space_sdk="streamlit",
                exist_ok=True
            )
            logger.info(f"Target Space '{hf_repo}' verified/created.")
        except Exception as e:
            logger.warning(f"Repository pre-check warning: {e}")
        
        # 4. Upload files
        api = HfApi()
        for f in files:
            if Path(f).exists():
                api.upload_file(path_or_fileobj=f, path_in_repo=Path(f).name,
                                repo_id=hf_repo, repo_type="space", token=hf_token)
                logger.info(f"Pushed {f}")
        return True
    except Exception as e:
        logger.error(f"HF push failed: {e}")
        return False

def main():
    logger.info("DAILY RUN - " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    db_path = os.environ.get("DB_PATH", "jobs.db")

    logger.info("[1/3] SCRAPING")
    try:
        from scraper_rekrute import scrape_rekrute
        from scraper_emploi_ma import scrape_emploi_ma
        df_r = scrape_rekrute(max_pages=5, db_path=db_path)
        df_e = scrape_emploi_ma(max_pages=5, db_path=db_path)
        logger.info(f"Rekrute: {len(df_r)}, Emploi.ma: {len(df_e)} new jobs")
    except Exception as e:
        logger.error(f"Scraping failed: {e}")

    logger.info("[2/3] NLP + MLFLOW")
    try:
        from mlflow_tracking import run_tracked_pipeline
        df, sf, run_id = run_tracked_pipeline(db_path=db_path)
        logger.info(f"Done - {len(df)} total jobs, run: {run_id[:8]}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

    logger.info("[3/3] PUSH TO HUGGINGFACE")
    push_to_huggingface(["jobs_nlp.csv", "skills_frequency.csv"])
    logger.info("COMPLETE")

if __name__ == "__main__":
    main()