"""
predict.py  -  loads the saved artifacts (takes seconds) and does the matching.

Used by app.py, but also works from the command line:

    python predict.py my_resume.pdf
    python predict.py my_resume.pdf --top 5
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pdfplumber

ARTIFACTS = Path(__file__).parent / "artifacts"


def extract_text_from_pdf(file) -> str:
    """file can be a path or an uploaded file object (e.g. from Streamlit)."""
    pages = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    text = " ".join(pages)
    return re.sub(r"\s+", " ", text).strip()


class Matcher:
    def __init__(self, artifacts_dir: Path = ARTIFACTS):
        from sentence_transformers import SentenceTransformer

        if not (artifacts_dir / "job_embeddings.npy").exists():
            raise FileNotFoundError("Artifacts not found. Run  python train.py  first.")

        self.model = SentenceTransformer(str(artifacts_dir / "model"))
        self.job_emb = np.load(artifacts_dir / "job_embeddings.npy")
        self.resume_emb = np.load(artifacts_dir / "resume_embeddings.npy")
        self.jobs = pd.read_csv(artifacts_dir / "jobs_meta.csv")
        self.resumes = pd.read_csv(artifacts_dir / "resumes_meta.csv")

    def _embed(self, text: str) -> np.ndarray:
        return self.model.encode([text], normalize_embeddings=True)[0]

    def jobs_for_resume(self, resume_text, top_n=10, location=None,
                        levels=None, job_types=None) -> pd.DataFrame:
        scores = self.job_emb @ self._embed(resume_text)  # cosine similarity

        mask = np.ones(len(scores), dtype=bool)
        if location and "job_location" in self.jobs:
            mask &= self.jobs["job_location"].fillna("").str.contains(
                location, case=False, regex=False).to_numpy()
        if levels and "job_level" in self.jobs:
            mask &= self.jobs["job_level"].isin(levels).to_numpy()
        if job_types and "job_type" in self.jobs:
            mask &= self.jobs["job_type"].isin(job_types).to_numpy()

        idx = np.where(mask)[0]
        idx = idx[np.argsort(-scores[idx])[:top_n]]
        out = self.jobs.iloc[idx].copy()
        out["match_score"] = scores[idx]
        return out.reset_index(drop=True)

    def candidates_for_job(self, job_text, top_n=10) -> pd.DataFrame:
        scores = self.resume_emb @ self._embed(job_text)
        idx = np.argsort(-scores)[:top_n]
        out = self.resumes.iloc[idx].copy()
        out["match_score"] = scores[idx]
        return out.reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="path to a resume PDF")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    text = extract_text_from_pdf(args.pdf)
    if not text:
        raise SystemExit("No text found in the PDF (is it a scanned image?).")
    res = Matcher().jobs_for_resume(text, top_n=args.top)
    print(res[["job_title", "company", "job_location", "match_score"]].to_string())
