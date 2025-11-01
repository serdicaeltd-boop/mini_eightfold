import os
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd

# ---------------------------------
# SETUP
# ---------------------------------
nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(page_title="Mini-Eightfold AI", page_icon="🧠", layout="wide")

PASSWORD = "aihero"  # 🔒 change this to your own secret password

# ---------------------------------
# AUTHENTICATION
# ---------------------------------
def login():
    st.title("🔐 Login to Mini-Eightfold AI")
    st.markdown("Enter your access password below to continue.")
    password_input = st.text_input("Password", type="password")
    if st.button("Login"):
        if password_input == PASSWORD:
            st.session_state["authenticated"] = True
            st.success("✅ Access granted! Loading app...")
        else:
            st.error("❌ Incorrect password. Try again.")

# ---------------------------------
# CLEANING FUNCTIONS
# ---------------------------------
def clean_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    filtered = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered)

def load_text_files(folder):
    texts = {}
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                texts[filename] = f.read()
    return texts

# ---------------------------------
# MATCHING & SCORING
# ---------------------------------
def match_resumes(resumes, job_desc):
    cleaned_resumes = [clean_text(t) for t in resumes.values()]
    cleaned_jd = clean_text(job_desc)

    tfidf = TfidfVectorizer()
    all_texts = cleaned_resumes + [cleaned_jd]
    tfidf_matrix = tfidf.fit_transform(all_texts)

    jd_vector = tfidf_matrix[-1]
    resume_vectors = tfidf_matrix[:-1]

    similarities = cosine_similarity(resume_vectors, jd_vector)
    scores = similarities.flatten()

    return dict(zip(resumes.keys(), scores))

# ---------------------------------
# CHATBOT (simple rule-based)
# ---------------------------------
def chat_response(query, ranked_results):
    if "best" in query:
        top_candidate = max(ranked_results, key=ranked_results.get)
        return f"The best candidate is **{top_candidate}** with a score of {ranked_results[top_candidate]:.2f}."
    elif "worst" in query:
        low_candidate = min(ranked_results, key=ranked_results.get)
        return f"The lowest scoring candidate is **{low_candidate}** with a score of {ranked_results[low_candidate]:.2f}."
    elif "average" in query:
        avg_score = sum(ranked_results.values()) / len(ranked_results)
        return f"The average candidate score is **{avg_score:.2f}**."
    else:
        return "You can ask me things like 'Who is the best candidate?' or 'What is the average score?'"

# ---------------------------------
# MAIN APP FUNCTION
# ---------------------------------
def app():
    st.title("🧠 Mini-Eightfold AI — Resume Matcher")
    st.markdown("Match resumes to job descriptions with AI and chat with insights.")

    job_folder = "data/job_descriptions"
    resume_folder = "data/resumes"

    if not os.path.exists(job_folder) or not os.path.exists(resume_folder):
        st.error("❌ Missing data folders.")
        return

    job_files = list(os.listdir(job_folder))
    selected_jd = st.selectbox("Select a Job Description", job_files)
    jd_path = os.path.join(job_folder, selected_jd)
    with open(jd_path, "r", encoding="utf-8", errors="ignore") as f:
        job_desc = f.read()

    resumes = load_text_files(resume_folder)
    st.info(f"Loaded {len(resumes)} resumes for matching.")

    if st.button("Run Matching"):
        results = match_resumes(resumes, job_desc)
        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
        st.success("✅ Matching complete!")

        df = pd.DataFrame(ranked, columns=["Candidate", "Score"])
        st.dataframe(df)

        st.markdown("### 💬 Chat with the AI Assistant")
        query = st.text_input("Ask a question (e.g., Who is the best candidate?)")
        if query:
            st.markdown(chat_response(query.lower(), results))

# ---------------------------------
# ENTRY POINT
# ---------------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login()
else:
    app()
