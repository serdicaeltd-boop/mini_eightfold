import os
import re
import nltk
import streamlit as st
import pandas as pd
from fpdf import FPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))
SKILL_KEYWORDS = ["python", "excel", "leadership", "quality", "analysis", "manufacturing", "project", "data", "communication"]

# --- FUNCTIONS ---

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(tokens)

def load_files(folder):
    data = {}
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                data[filename] = clean_text(f.read())
    return data

def compute_similarity(resumes, job_desc):
    results = []
    for name, text in resumes.items():
        vect = TfidfVectorizer()
        tfidf = vect.fit_transform([job_desc, text])
        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        results.append((name, score))
    return sorted(results, key=lambda x: x[1], reverse=True)

def generate_explanation(resume_text, job_keywords):
    strengths = [kw for kw in job_keywords if kw in resume_text]
    weaknesses = [kw for kw in job_keywords if kw not in resume_text]
    return strengths, weaknesses

def highlight_keywords(text, keywords):
    for kw in set(keywords):
        pattern = re.compile(r'\b(' + re.escape(kw) + r')\b', re.IGNORECASE)
        text = pattern.sub(f"<span style='background-color:yellow'>{kw}</span>", text)
    return text

def export_pdf(results, resumes, job_keywords, chatbot_log):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "Mini-Eightfold AI — Report\n\n", align='C')
    pdf.multi_cell(0, 10, "Ranking Results:\n")
    for name, score in results:
        pdf.multi_cell(0, 10, f"{name}: {score:.3f}")
    pdf.multi_cell(0, 10, "\nChatbot Summary:\n")
    for entry in chatbot_log:
        pdf.multi_cell(0, 10, f"Q: {entry['question']}")
        pdf.multi_cell(0, 10, f"A: {entry['answer']}\n")
    filename = "results/report.pdf"
    os.makedirs("results", exist_ok=True)
    pdf.output(filename)
    return filename

def extract_keywords_from_jd(text):
    tokens = [word for word in text.split() if word not in STOPWORDS and len(word) > 2]
    freq = nltk.FreqDist(tokens)
    return [word for word, _ in freq.most_common(20)]

def chatbot_answer(question, results, resumes, job_keywords):
    q = question.lower()
    if "best" in q:
        top_candidate, top_score = results[0]
        return f"The best match is **{top_candidate}** with a score of {top_score:.3f}."
    elif "worst" in q or "lowest" in q:
        bottom_candidate, bottom_score = results[-1]
        return f"The lowest match is **{bottom_candidate}** with a score of {bottom_score:.3f}."
    elif "compare" in q:
        names = [r[0].split('.')[0] for r in results]
        found = [n for n in names if n.lower() in q]
        if len(found) == 2:
            r1 = next(r for r in results if found[0].lower() in r[0].lower())
            r2 = next(r for r in results if found[1].lower() in r[0].lower())
            diff = abs(r1[1] - r2[1])
            better = r1 if r1[1] > r2[1] else r2
            return f"{better[0]} performed better by {diff:.2f} points."
        else:
            return "Please mention exactly two names to compare."
    elif "weak" in q or "missing" in q:
        name = next((r[0] for r in results if r[0].split('.')[0].lower() in q), None)
        if name:
            strengths, weaknesses = generate_explanation(resumes[name], job_keywords)
            return f"{name} is missing: {', '.join(weaknesses) if weaknesses else 'No missing skills found.'}"
        else:
            return "Please specify whose weaknesses you want to know."
    else:
        return "Try asking: 'Who is the best?', 'Compare John and Jane', or 'What skills is John missing?'"


# --- MAIN APP ---

st.set_page_config(page_title="Mini-Eightfold AI", layout="wide")
st.title("🚀 Mini-Eightfold AI — Resume Matcher 2.0")
st.write("Compare multiple resumes against multiple job descriptions with smart analysis.")

# Load job descriptions
JD_FOLDER = "data/job_descriptions"
os.makedirs(JD_FOLDER, exist_ok=True)
jd_files = [f for f in os.listdir(JD_FOLDER) if f.endswith(".txt")]

if not jd_files:
    st.warning("Please upload at least one job description in `data/job_descriptions/`.")
else:
    selected_jd = st.selectbox("Select a job description:", jd_files)
    job_desc_path = os.path.join(JD_FOLDER, selected_jd)
    with open(job_desc_path, "r", encoding="utf-8") as f:
        job_desc = clean_text(f.read())

    job_keywords = extract_keywords_from_jd(job_desc)
    resumes = load_files("data/resumes")
    results = compute_similarity(resumes, job_desc)
    chatbot_log = []

    st.subheader(f"Results for: {selected_jd}")
    df = pd.DataFrame(results, columns=["Resume", "Score"])
    st.dataframe(df)

    st.subheader("Resumes with Highlights")
    for name, score in results:
        strengths, weaknesses = generate_explanation(resumes[name], job_keywords)
        highlighted_text = highlight_keywords(resumes[name], strengths + SKILL_KEYWORDS)
        st.markdown(f"**{name} — Score: {score:.3f}**", unsafe_allow_html=True)
        st.markdown(highlighted_text, unsafe_allow_html=True)
        if weaknesses:
            st.markdown(f"<span style='color:red'>Missing keywords: {', '.join(weaknesses)}</span>", unsafe_allow_html=True)
        st.write("---")

    st.subheader("💬 Chatbot Assistant")
    user_input = st.text_input("Ask something about your results:")
    if user_input:
        answer = chatbot_answer(user_input, results, resumes, job_keywords)
        st.markdown(answer, unsafe_allow_html=True)
        chatbot_log.append({"question": user_input, "answer": answer})

    if st.button("Generate PDF Report"):
        filename = export_pdf(results, resumes, job_keywords, chatbot_log)
        st.success(f"PDF exported: {filename}")
