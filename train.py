"""
train.py  -  run this ONCE (or whenever your datasets change).

It cleans the data, creates embeddings (the slow part) and saves everything
into the  artifacts/  folder so predict.py and app.py can just load it.

    python train.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE = Path(__file__).parent
JOBS_CSV = BASE / "postings" / "postings.csv"
RESUME_CSV = BASE / "resume_data" / "Resume" / "Resume.csv"
ARTIFACTS = BASE / "artifacts"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_resumes() -> pd.DataFrame:
    df = pd.read_csv(RESUME_CSV)
    df = df.dropna(subset=["Resume_str"])
    df["clean_text"] = (
        df["Resume_str"]
        .str.replace("\n", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
    return df


def load_jobs() -> pd.DataFrame:
    df = pd.read_csv(JOBS_CSV)
    # make column names safe: "job level" -> "job_level"
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df = df.dropna(subset=["job_summary", "job_title"])
    df["clean_description"] = (
        df["job_summary"]
        .str.replace(r"<[^>]+>", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df.drop_duplicates(subset=["clean_description"]).reset_index(drop=True)
    df["embedding_text"] = df["job_title"] + ". " + df["clean_description"]
    return df


def main():
    ARTIFACTS.mkdir(exist_ok=True)

    print("Loading data...")
    resumes = load_resumes()
    jobs = load_jobs()
    print(f"  {len(resumes)} resumes, {len(jobs)} jobs")

    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)

    # normalize_embeddings=True -> cosine similarity becomes a simple dot product
    print("Encoding resumes...")
    resume_emb = model.encode(
        resumes["clean_text"].tolist(),
        batch_size=32, show_progress_bar=True, normalize_embeddings=True,
    )
    print("Encoding jobs...")
    job_emb = model.encode(
        jobs["embedding_text"].tolist(),
        batch_size=32, show_progress_bar=True, normalize_embeddings=True,
    )

    print("Saving to", ARTIFACTS)
    np.save(ARTIFACTS / "resume_embeddings.npy", resume_emb.astype("float32"))
    np.save(ARTIFACTS / "job_embeddings.npy", job_emb.astype("float32"))

    # only keep the columns the app needs, and trim long text
    job_cols = ["job_title", "company", "job_location", "job_link",
                "job_level", "job_type", "clean_description"]
    jobs_meta = jobs[[c for c in job_cols if c in jobs.columns]].copy()
    jobs_meta["clean_description"] = jobs_meta["clean_description"].str[:1200]
    jobs_meta.to_csv(ARTIFACTS / "jobs_meta.csv", index=False)

    resumes_meta = resumes[["ID", "Category"]].copy()
    resumes_meta["preview"] = resumes["clean_text"].str[:600]
    resumes_meta.to_csv(ARTIFACTS / "resumes_meta.csv", index=False)

    # save the model itself so it loads from disk next time (no re-download)
    model.save(str(ARTIFACTS / "model"))
    print("Done. Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
