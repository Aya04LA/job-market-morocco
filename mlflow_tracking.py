# =============================================================================
# MLFLOW TRACKING — Job Market Intelligence Platform
# =============================================================================
# WHY MLflow?
# Without experiment tracking, every time you retrain the model you lose:
#   - What parameters you used
#   - What metrics you got
#   - Which version of the model is in production
# MLflow solves this by logging everything automatically.
# This is the core of MLOps — treating ML models like software with versions.
#
# What we track:
#   PARAMETERS  → model settings (C value, n-grams, threshold...)
#   METRICS     → CV F1, n_jobs, n_skills_found, avg_skills_per_job
#   ARTIFACTS   → the trained model files (vectorizer + classifier)
#
# To view the UI:
#   mlflow ui
#   → open http://localhost:5000 in your browser
# =============================================================================

import mlflow
import mlflow.sklearn
import logging
import pickle
import pandas as pd
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# MLflow experiment name — all runs for this project go under this name
EXPERIMENT_NAME = "job_market_morocco"


def setup_mlflow(tracking_uri="./mlruns"):
    """
    Configure MLflow to store runs locally in ./mlruns folder.
    WHY local? For development. When you deploy to Railway, you can
    point this to a remote MLflow server or use MLflow on HuggingFace.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow tracking URI: {tracking_uri}")
    logger.info(f"Experiment: {EXPERIMENT_NAME}")


def run_tracked_pipeline(db_path="jobs.db"):
    """
    Run the full NLP pipeline with MLflow tracking.
    Every run is logged with its parameters, metrics, and model artifacts.

    This is what you call instead of run_nlp_pipeline() directly —
    it wraps it with tracking so every execution is recorded.
    """
    from nlp_pipeline import (
        run_skill_extraction,
        train_sector_classifier,
        predict_missing_sectors,
        compute_skill_frequency,
    )
    from data_cleaning import load_and_clean

    setup_mlflow()

    # Parameters we want to track — if you change these, MLflow records it
    params = {
        "tfidf_max_features":  8000,
        "tfidf_ngram_range":   "1-2",
        "tfidf_min_df":        2,
        "logreg_C":            5,
        "logreg_class_weight": "balanced",
        "confidence_threshold": 0.40,
        "spacy_model":         "fr_core_news_md",
        "n_skill_patterns":    160,
        "db_path":             db_path,
        "run_date":            datetime.now().strftime("%Y-%m-%d"),
    }

    # mlflow.start_run() creates a new tracked experiment run
    # Everything inside this block gets logged to MLflow
    with mlflow.start_run(run_name=f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M')}"):

        logger.info("Starting MLflow-tracked pipeline run...")

        # ── Log parameters ──────────────────────────────────────────────────
        mlflow.log_params(params)

        # ── Run pipeline steps ───────────────────────────────────────────────
        df = load_and_clean(db_path)
        mlflow.log_metric("n_jobs_total", len(df))

        df = run_skill_extraction(df)
        mlflow.log_metric("n_jobs_with_skills", int((df["skills_count"] > 0).sum()))
        mlflow.log_metric("avg_skills_per_job",  round(df["skills_count"].mean(), 2))
        mlflow.log_metric("n_jobs_no_skills",    int((df["skills_count"] == 0).sum()))

        vectorizer, clf, le, metrics = train_sector_classifier(df)
        mlflow.log_metric("cv_f1_macro",     round(metrics["cv_f1_mean"], 4))
        mlflow.log_metric("cv_f1_std",       round(metrics["cv_f1_std"],  4))
        mlflow.log_metric("n_training_jobs", metrics["n_train"])

        df = predict_missing_sectors(df, vectorizer, clf, le)
        mlflow.log_metric("n_sectors_predicted", int(df["sector_predicted"].sum()))
        mlflow.log_metric("n_still_unspecified",
                          int((df["sector_final"] == "Non spécifié").sum()))

        skill_freq = compute_skill_frequency(df)
        mlflow.log_metric("n_unique_skills_found", len(skill_freq))

        # ── Log model artifacts ──────────────────────────────────────────────
        # WHY log the model? So you can load any past version later.
        # mlflow.sklearn.log_model saves the full sklearn pipeline (vectorizer + clf)
        mlflow.sklearn.log_model(clf, "sector_classifier")
        mlflow.sklearn.log_model(vectorizer, "tfidf_vectorizer")

        # Also save label encoder (not sklearn-native, use pickle)
        le_path = "label_encoder.pkl"
        with open(le_path, "wb") as f:
            pickle.dump(le, f)
        mlflow.log_artifact(le_path)

        # Save output CSVs as artifacts too
        jobs_path  = "jobs_nlp.csv"
        skills_path = "skills_frequency.csv"
        df.to_csv(jobs_path, index=False, encoding="utf-8-sig")
        skill_freq.to_csv(skills_path, index=False, encoding="utf-8-sig")
        mlflow.log_artifact(jobs_path)
        mlflow.log_artifact(skills_path)

        # ── Log sector distribution as a metric per sector ───────────────────
        for sector, count in df["sector_final"].value_counts().items():
            safe_name = sector.replace(" ", "_").replace("&", "and").replace("/", "_")
            mlflow.log_metric(f"sector_{safe_name}", int(count))

        run_id = mlflow.active_run().info.run_id
        logger.info(f"\n✓ MLflow run complete | run_id: {run_id}")
        logger.info(f"  View UI: mlflow ui  →  http://localhost:5000")

    return df, skill_freq, run_id


def load_model_from_run(run_id: str):
    """
    Load a saved model from a specific MLflow run.
    WHY useful? If a new pipeline run produces worse F1,
    you can roll back to the previous run's model instantly.
    This is production MLOps.
    """
    setup_mlflow()
    import pickle, os

    clf_uri        = f"runs:/{run_id}/sector_classifier"
    vectorizer_uri = f"runs:/{run_id}/tfidf_vectorizer"

    clf        = mlflow.sklearn.load_model(clf_uri)
    vectorizer = mlflow.sklearn.load_model(vectorizer_uri)

    # Find and load label encoder artifact
    client   = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)
    le_artifact = next((a for a in artifacts if "label_encoder" in a.path), None)

    le = None
    if le_artifact:
        local_path = client.download_artifacts(run_id, le_artifact.path)
        with open(local_path, "rb") as f:
            le = pickle.load(f)

    logger.info(f"✓ Model loaded from run: {run_id}")
    return clf, vectorizer, le


def compare_runs():
    """
    Print a comparison table of all MLflow runs for this experiment.
    Useful when you've run the pipeline multiple times with different settings.
    """
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    exp    = client.get_experiment_by_name(EXPERIMENT_NAME)

    if not exp:
        logger.warning("No runs found yet. Run run_tracked_pipeline() first.")
        return

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.cv_f1_macro DESC"]
    )

    if not runs:
        logger.warning("No runs found.")
        return

    rows = []
    for r in runs:
        rows.append({
            "run_id":        r.info.run_id[:8],
            "run_name":      r.info.run_name,
            "cv_f1_macro":   r.data.metrics.get("cv_f1_macro", "N/A"),
            "n_jobs":        r.data.metrics.get("n_jobs_total", "N/A"),
            "sectors_pred":  r.data.metrics.get("n_sectors_predicted", "N/A"),
            "logreg_C":      r.data.params.get("logreg_C", "N/A"),
            "status":        r.info.status,
        })

    df_runs = pd.DataFrame(rows)
    print("\n" + "="*70)
    print("MLflow Run Comparison")
    print("="*70)
    print(df_runs.to_string(index=False))
    print("="*70 + "\n")
    return df_runs


# =============================================================================
# RUN:
#   python mlflow_tracking.py
#
# Then view results:
#   mlflow ui
#   → http://localhost:5000
#
# OR import:
#   from mlflow_tracking import run_tracked_pipeline
#   df, skill_freq, run_id = run_tracked_pipeline("jobs.db")
# =============================================================================

if __name__ == "__main__":
    df, skill_freq, run_id = run_tracked_pipeline("jobs.db")
    print(f"\nRun ID: {run_id}")
    print("\nRun comparison:")
    compare_runs()
