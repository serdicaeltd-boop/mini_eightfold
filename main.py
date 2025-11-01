import os
import re
import nltk
import streamlit as st
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF

# --- NLTK setup ---
nltk_packages = ["punkt", "stopwords"]
for pkg in nltk_packages:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOPWORDS = set(stopwords.words("english"))
PASSWORD = "admin123"  # change this to your own

# --- Streamlit page ---
st.set_page_config(page_title="Mini-Eightfold AI", page_icon="🧠", layout="wide")

# --- Model loading (lazy + cached) ---
@st.cache_resource
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# --- Text cleaning ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)


# --- Load resumes / job descriptions ---
def load_text_files(folder):
    texts = {}
    if not os.path.exists(folder):
        os.makedirs(folder)
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                texts[filename] = f.read()
    return texts


# --- Matching ---
def match_resumes(resumes, job_desc, use_model=True):
    if not resumes or not job_desc:
        return []

    if use_model:
        model = load_model()
        cleaned_resumes = [clean_text(t) for t in resumes.values()]
        cleaned_jd = clean_text(job_desc)
        embeddings = model.encode([cleaned_jd] + cleaned_resumes)
        jd_vector = embeddings[0].reshape(1, -1)
        resume_vectors = embeddings[1:]
        similarities = cosine_similarity(jd_vector, resume_vectors)[0]
    else:
        # TF-IDF fallback
        vectorizer = TfidfVectorizer()
        texts = list(resumes.values()) + [job_desc]
        tfidf_matrix = vectorizer.fit_transform(texts)
        jd_vector = tfidf_matrix[-1]
        resume_vectors = tfidf_matrix[:-1]
        similarities = cosine_similarity(resume_vectors, jd_vector).flatten()

    results = sorted(zip(resumes.keys(), similarities), key=lambda x: x[1], reverse=True)
    return results


# --- PDF export ---
def generate_pdf(results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="📋 Resume Matching Results", ln=True, align="C")
    pdf.ln(10)
    for name, score in results:
        pdf.cell(200, 10, txt=f"{name}: {score:.3f}", ln=True, align="L")
    pdf_file = "results_summary.pdf"
    pdf.output(pdf_file)
    return pdf_file


# --- Chatbot ---
def chat_response(query, ranked_results):
    query = query.lower()
    if "best" in query:
        top_candidate = max(ranked_results, key=lambda x: x[1])[0]
        return f"The best candidate is **{top_candidate}**."
    elif "worst" in query:
        low_candidate = min(ranked_results, key=lambda x: x[1])[0]
        return f"The lowest scoring candidate is **{low_candidate}**."
    elif "average" in query:
        avg_score = sum(score for _, score in ranked_results) / len(ranked_results)
        return f"The average candidate score is **{avg_score:.3f}**."
    else:
        return "Ask things like 'Who is the best candidate?' or 'Average score?'"


# --- Login ---
def login_screen():
    st.title("🔐 Mini-Eightfold AI — Login")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if password == PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password")


# --- Main app ---
def app():
    st.title("🚀 Mini-Eightfold AI — Resume Matcher")
    st.write("Match resumes to job descriptions and chat with AI insights.")

    # Load data
    job_folder = "data/job_descriptions"
    resume_folder = "data/resumes"

    job_files = [f for f in os.listdir(job_folder) if f.endswith(".txt")] if os.path.exists(job_folder) else []
    if not job_files:
        st.warning("No job description found!")
        return

    resumes = load_text_files(resume_folder)
    if not resumes:
        st.warning("No resumes found!")
        return

    selected_jd = st.selectbox("Select Job Description", job_files)
    with open(os.path.join(job_folder, selected_jd), "r", encoding="utf-8") as f:
        job_desc = f.read()

    st.info(f"Loaded {len(resumes)} resumes for matching.")

    use_model = st.checkbox("Use AI model for matching (slower)", value=True)
    if st.button("Run Matching"):
        results = match_resumes(resumes, job_desc, use_model=use_model)
        st.subheader("📊 Ranked Candidates")
        for name, score in results:
            st.markdown(f"**{name} — Score: {score:.3f}**")

        # PDF
        if st.button("📄 Generate PDF Summary"):
            pdf_path = generate_pdf(results)
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", f, file_name="resume_results.pdf", mime="application/pdf")

        # Chatbot
        query = st.text_input("💬 Ask a question (e.g., Who is the best candidate?)")
        if query:
            st.markdown(chat_response(query, results))


# --- Entry ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login_screen()
else:
    app()
