import os
import re
from collections import Counter
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from fpdf import FPDF

# ================= CONFIG =================
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RESUME_FOLDER = "resumes"
# ==========================================

# ================= FUNCTIONS =================
def extract_keywords(text):
    stopwords = set([
        "and", "or", "the", "a", "an", "in", "on", "for", "of", "to", "with",
        "is", "are", "at", "by", "as", "from", "this", "that", "it", "be"
    ])
    words = re.findall(r"\b\w+\b", text.lower())
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    return Counter(keywords)


def rank_resumes(job_text, resumes_dict):
    model = SentenceTransformer(MODEL_NAME)
    job_emb = model.encode(job_text, convert_to_tensor=True)
    resume_texts = list(resumes_dict.values())
    resume_emb = model.encode(resume_texts, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(job_emb, resume_emb)[0]
    results = sorted(
        zip(resumes_dict.keys(), sims.tolist()), key=lambda x: x[1], reverse=True
    )
    return results


def generate_explanation(candidate_text, job_keywords):
    candidate_keywords = extract_keywords(candidate_text)
    overlap = set(job_keywords.keys()) & set(candidate_keywords.keys())
    
    strengths = sorted(overlap, key=lambda w: candidate_keywords[w], reverse=True)[:5]
    weaknesses = [w for w in job_keywords.keys() if w not in candidate_keywords][:5]
    
    return strengths, weaknesses


def compare_candidates(candidate1, candidate2, resumes, job_keywords):
    str1, weak1 = generate_explanation(resumes[candidate1], job_keywords)
    str2, weak2 = generate_explanation(resumes[candidate2], job_keywords)

    comparison = f"{candidate1}\n  Strengths: {', '.join(str1) if str1 else 'None'}\n  Weaknesses: {', '.join(weak1) if weak1 else 'None'}\n\n"
    comparison += f"{candidate2}\n  Strengths: {', '.join(str2) if str2 else 'None'}\n  Weaknesses: {', '.join(weak2) if weak2 else 'None'}"
    return comparison


def answer_free_text(question, resumes, results, job_keywords, top_n=2):
    question_words = set(re.findall(r"\b\w+\b", question.lower()))
    candidate_scores = {}

    for name, _ in results:
        candidate_text = resumes[name].lower()
        matched = [word for word in question_words if word in candidate_text]
        candidate_scores[name] = len(matched)

    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)

    answers = []
    for name, matches in sorted_candidates[:top_n]:
        if matches == 0:
            continue
        confidence = matches / len(question_words)
        matched_words = [w for w in question_words if w in resumes[name].lower()]
        answers.append({
            "candidate": name,
            "confidence": confidence,
            "matched_words": matched_words
        })

    if not answers:
        return "I couldn’t find any candidate that matches your question."
    else:
        output = ""
        for ans in answers:
            output += f"{ans['candidate']} — Confidence: {ans['confidence']:.2f}\n"
            output += f"Matched words: {', '.join(ans['matched_words'])}\n\n"
        return output


def clean_text(text):
    """Ensure Latin-1 safe text for FPDF."""
    return text.encode('latin-1', 'ignore').decode('latin-1')


def export_pdf(results, resumes, job_keywords, chatbot_log=None, filename="Mini_Eightfold_Report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Mini-Eightfold AI Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", "B", 12)
    pdf.multi_cell(0, 8, clean_text("Ranked Candidates:"))
    pdf.set_font("Arial", "", 12)

    for i, (name, score) in enumerate(results, 1):
        strengths, weaknesses = generate_explanation(resumes[name], job_keywords)
        pdf.multi_cell(
            0, 8, clean_text(f"{i}. {name} — Score: {score:.3f}\n  Strengths: {', '.join(strengths) if strengths else 'None'}\n  Weaknesses: {', '.join(weaknesses) if weaknesses else 'None'}\n")
        )

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.multi_cell(0, 8, clean_text("Resumes Summary:"))
    pdf.set_font("Arial", "", 12)

    for name, _ in results:
        pdf.multi_cell(0, 8, clean_text(f"{name}:"))
        pdf.multi_cell(0, 8, clean_text(resumes[name]))
        pdf.ln(5)

    # Include Chatbot log
    if chatbot_log:
        pdf.set_font("Arial", "B", 12)
        pdf.multi_cell(0, 8, clean_text("Chatbot Q&A Log:"))
        pdf.set_font("Arial", "", 12)
        for entry in chatbot_log:
            pdf.multi_cell(0, 8, clean_text(f"Q: {entry['question']}\nA: {entry['answer']}\n"))
            pdf.ln(2)

    pdf.output(filename)
    return filename


# ================= STREAMLIT UI =================
st.set_page_config(page_title="Mini-Eightfold AI", layout="wide")
st.title("Mini-Eightfold AI — Resume Matcher (Auto-Load Folder)")

chatbot_log = []

# --- Auto-load resumes ---
if not os.path.exists(RESUME_FOLDER):
    os.makedirs(RESUME_FOLDER)
    st.info(f"Created '{RESUME_FOLDER}' folder. Place your resume .txt files here.")

resume_files = [f for f in os.listdir(RESUME_FOLDER) if f.endswith(".txt")]
resumes = {}
for f in resume_files:
    with open(os.path.join(RESUME_FOLDER, f), "r", encoding="utf-8") as file:
        resumes[f] = file.read()

if not resumes:
    st.warning(f"No resume files found in '{RESUME_FOLDER}' folder. Please add .txt resumes.")

# --- Job description upload ---
job_file = st.file_uploader("Upload Job Description (.txt)", type="txt")
if job_file:
    job_text = job_file.read().decode("utf-8")

    if resumes:
        st.info("Computing similarity scores...")
        results = rank_resumes(job_text, resumes)
        job_keywords = extract_keywords(job_text)

        st.subheader("Ranked Candidates")
        for i, (name, score) in enumerate(results, 1):
            strengths, weaknesses = generate_explanation(resumes[name], job_keywords)
            st.write(f"{i}. {name} — Score: {score:.3f}")
            st.write(f"   Strengths: {', '.join(strengths) if strengths else 'None'}; Weaknesses: {', '.join(weaknesses) if weaknesses else 'None'}")

        st.subheader("Resumes")
        for name, score in results:
            st.write(f"{name} — Score: {score:.3f}")
            st.text(resumes[name])
            st.write("---")

        st.subheader("Chatbot")
        user_input = st.text_input("Ask me something about the results:")

        if user_input:
            answer = ""
            input_lower = user_input.lower()
            
            if "best" in input_lower:
                top_candidate = results[0][0]
                top_score = results[0][1]
                answer = f"The best matching resume is '{top_candidate}' with a score of {top_score:.3f}."
            
            elif "worst" in input_lower or "lowest" in input_lower:
                bottom_candidate = results[-1][0]
                bottom_score = results[-1][1]
                answer = f"The lowest matching resume is '{bottom_candidate}' with a score of {bottom_score:.3f}."
            
            elif "why" in input_lower or "reason" in input_lower:
                top_candidate = results[0][0]
                strengths, weaknesses = generate_explanation(resumes[top_candidate], job_keywords)
                answer = f"'{top_candidate}' ranked higher because it shares more key terms with the job description.\n"
                answer += f"Strengths: {', '.join(strengths) if strengths else 'None'}; Weaknesses: {', '.join(weaknesses) if weaknesses else 'None'}"

            elif "compare" in input_lower:
                names = re.findall(r"\b\w+\.txt\b", user_input)
                if len(names) >= 2:
                    answer = compare_candidates(names[0], names[1], resumes, job_keywords)
                else:
                    answer = "Please provide two candidate filenames to compare, e.g., 'Compare resume_john.txt and resume_jane.txt'."

            elif "summary" in input_lower or "overview" in input_lower:
                answer = "Summary of all candidates:\n"
                for name, score in results:
                    strengths, weaknesses = generate_explanation(resumes[name], job_keywords)
                    answer += f"- {name} ({score:.3f}) — Strengths: {', '.join(strengths) if strengths else 'None'}; Weaknesses: {', '.join(weaknesses) if weaknesses else 'None'}\n"
            else:
                answer = answer_free_text(user_input, resumes, results, job_keywords, top_n=2)

            st.markdown(answer)
            chatbot_log.append({"question": user_input, "answer": answer})

        st.subheader("Export Report")
        if st.button("Generate PDF Report"):
            pdf_file = export_pdf(results, resumes, job_keywords, chatbot_log)
            with open(pdf_file, "rb") as f:
                st.download_button("Download PDF", f, file_name=pdf_file)
