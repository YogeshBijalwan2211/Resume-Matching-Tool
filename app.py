"""
app.py  -  the front end.   Run with:   streamlit run app.py
"""
import streamlit as st

from predict import Matcher, extract_text_from_pdf

st.set_page_config(page_title="Resume Matching Tool", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading model and embeddings...")
def get_matcher() -> Matcher:
    return Matcher()  # loaded once, reused on every interaction


try:
    matcher = get_matcher()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.title("📄 Resume Matching Tool")
st.caption("Upload a resume to find the best matching jobs, or paste a job description to find matching candidates.")

tab_jobs, tab_cands = st.tabs(["Find jobs for my resume", "Find candidates for a job"])

# ---------------------------------------------------------------- Tab 1
with tab_jobs:
    with st.sidebar:
        st.header("Filters")
        top_n = st.slider("Number of results", 3, 30, 10)
        location = st.text_input("Location contains", placeholder="e.g. Bloomington")
        levels = job_types = None
        if "job_level" in matcher.jobs:
            levels = st.multiselect("Job level", sorted(matcher.jobs["job_level"].dropna().unique()))
        if "job_type" in matcher.jobs:
            job_types = st.multiselect("Job type", sorted(matcher.jobs["job_type"].dropna().unique()))

    pdf = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

    if pdf is not None:
        try:
            text = extract_text_from_pdf(pdf)
        except Exception as e:
            st.error(f"Could not read this PDF: {e}")
            st.stop()

        if not text:
            st.warning("No text found. This looks like a scanned/image PDF; please upload a text-based PDF.")
        else:
            with st.expander("Text extracted from your resume"):
                st.write(text[:3000] + ("..." if len(text) > 3000 else ""))

            with st.spinner("Matching..."):
                results = matcher.jobs_for_resume(
                    text, top_n=top_n, location=location or None,
                    levels=levels or None, job_types=job_types or None,
                )

            if results.empty:
                st.info("No jobs match those filters. Try relaxing them.")
            for i, row in results.iterrows():
                with st.container(border=True):
                    left, right = st.columns([4, 1])
                    left.subheader(f"{i + 1}. {row['job_title']}")
                    left.markdown(
                        f"**{row['company']}** · {row.get('job_location', '')} · "
                        f"{row.get('job_level', '')} · {row.get('job_type', '')}"
                    )
                    right.metric("Match", f"{row['match_score'] * 100:.0f}%")
                    st.progress(float(max(0.0, min(1.0, row["match_score"]))))
                    with st.expander("Job description"):
                        st.write(row["clean_description"])
                    if isinstance(row.get("job_link"), str):
                        st.link_button("View job posting", row["job_link"])

# ---------------------------------------------------------------- Tab 2
with tab_cands:
    jd = st.text_area("Paste a job description", height=200)
    n_c = st.slider("Number of candidates", 3, 30, 10, key="n_c")
    if st.button("Find candidates") and jd.strip():
        res = matcher.candidates_for_job(jd, top_n=n_c)
        for i, row in res.iterrows():
            with st.container(border=True):
                a, b = st.columns([4, 1])
                a.markdown(f"**{i + 1}. Resume ID {row['ID']}** · {row['Category']}")
                b.metric("Match", f"{row['match_score'] * 100:.0f}%")
                with st.expander("Resume preview"):
                    st.write(row["preview"])
