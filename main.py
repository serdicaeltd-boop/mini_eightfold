import os
import re
import nltk
import streamlit as st
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF

# --- NLTK auto-setup (for Streamlit Cloud) ---
nltk_packages = ["punkt", "stopwords"]
for pkg in nltk_packages:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

# --- Constants ---
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
STOPWORDS = set(stopwords.words("english"))
PASSWORD = "admin123"  # change this for your app


# --- Helper Functions ---
def clean_text(text):
    """Lowercase, remove non-letters, and filter out stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)


def load_resumes(folder="data/resumes"):
    resumes = {}
    if not os.path.exists(folder):
        os.makedirs(folder)
        return resumes

    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                resumes[file] = f.read()
    return resumes


def load_job_description(file_path="data/job_descriptions/quality_engineer.txt"):
    if not os.path.exists(file_path):
        st.warning("No job description found.")
        return ""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def match_resumes(resumes, job_desc):
    """Compute similarity scores between resumes and job description."""
    if not resumes or not job_desc:
        return []

    model = SentenceTransformer(MODEL_NAME)

    cleaned_resumes = [clean_text(t) for t in resumes.values()]
    cleaned_jd = clean_text(job_desc)

    embeddings = model.encode([cleaned_jd] + cleaned_resumes)
    jd_vector = embeddings[0].reshape(1, -1)
    resume_vectors = embeddings[1:]

    similarities = cosine_similarity(jd_vector, resume_vectors)[0]
    results = sorted(zip(resumes.keys(), similarities), key=lambda x: x[1], reverse=True)
    return results


def generate_pdf(results):
    """Generate a PDF summary of ranked resumes."""
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


# --- Streamlit UI ---
def login_screen():
    st.title("🔐 Mini-Eightfold AI — Secure Login")
    st.write("Please enter the password to access the system.")

    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if password == PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")


def app():
    st.title("🚀 Mini-Eightfold AI — Resume Matcher")

    job_desc = load_job_description()
    resumes = load_resumes()

    if not resumes:
        st.warning("⚠️ No resumes found in the folder!")
        return

    st.success(f"✅ Loaded {len(resumes)} resumes and job description.")
    st.write("⏳ Computing similarity scores...")

    results = match_resumes(resumes, job_desc)

    st.subheader("📊 Results:")
    for name, score in results:
        st.markdown(f"**{name} — Score: {score:.3f}**")

    # PDF export
    if st.button("📄 Generate PDF Summary"):
        pdf_path = generate_pdf(results)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download Results PDF",
                data=f,
                file_name="resume_match_results.pdf",
                mime="application/pdf",
            )


# --- App Entry Point ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login_screen()
else:
    app()
