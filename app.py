"""CYGNUSA Elite-Hire: Explainable AI-Driven Hiring Evaluation."""

from __future__ import annotations

import re
from typing import List, Tuple

import pdfplumber
import streamlit as st

DEFAULT_JD_SKILLS = ["python", "sql", "data structures", "problem solving"]
DEFAULT_REQUIRED_EXPERIENCE = 3  # years
DEFAULT_REQUIRED_EDUCATION = "bachelor"


def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from a PDF file-like object and normalize to lowercase."""
    if pdf_file is None:
        return ""

    try:
        with pdfplumber.open(pdf_file) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages).lower()
    except Exception:
        return ""


def count_years_of_experience(text: str) -> int:
    """Find the largest numeric years-of-experience mention in the text."""
    if not text:
        return 0

    matches = re.findall(r"(\d+)\s*\+?\s*years?", text)
    years = [int(match) for match in matches] if matches else [0]
    return max(years)


def extract_required_skills(job_description: str, fallback_skills: List[str]) -> List[str]:
    """Extract required skills from a job description (comma or newline-separated)."""
    if not job_description.strip():
        return fallback_skills

    split_candidates = re.split(r"[,\n;/]+", job_description.lower())
    cleaned = [skill.strip() for skill in split_candidates if skill.strip()]
    return cleaned or fallback_skills


def evaluate_skills(text: str, skills: List[str]) -> Tuple[List[str], float]:
    """Return matched skills and the skills score out of 50."""
    matched = [skill for skill in skills if skill in text]
    score = (len(matched) / len(skills)) * 50 if skills else 0
    return matched, score


def evaluate_experience(text: str, required_years: int) -> Tuple[int, float]:
    """Return years found and experience score out of 30."""
    years_found = count_years_of_experience(text)
    score = 30 if years_found >= required_years else 15
    return years_found, score


def evaluate_education(text: str, required_education: str) -> Tuple[bool, float]:
    """Return education match and education score out of 20."""
    matches = required_education.lower() in text
    score = 20 if matches else 10
    return matches, score


def decide(score: float) -> str:
    """Determine hiring decision based on total score."""
    if score >= 75:
        return "Hire"
    if score >= 50:
        return "Potential"
    return "Reject"


def build_explanation(
    matched_skills: List[str],
    total_skills: int,
    years_found: int,
    experience_required: int,
    education_match: bool,
    total_score: float,
    decision: str,
    job_role: str,
) -> str:
    """Create a human-readable explanation for the evaluation."""
    skills_list = ", ".join(matched_skills) if matched_skills else "no listed skills"
    skills_sentence = (
        f"Candidate matches {len(matched_skills)} out of {total_skills} required "
        f"skills including {skills_list}."
    )

    if years_found >= experience_required:
        experience_sentence = (
            "Experience meets the required threshold "
            f"with {years_found} years mentioned."
        )
    else:
        experience_sentence = (
            "Experience is slightly below the required threshold "
            f"with {years_found} years mentioned."
        )

    education_sentence = (
        "Education aligns with role expectations."
        if education_match
        else "Education is below the stated requirement."
    )

    return (
        f"Role evaluated: {job_role or 'Role not specified'}.\n"
        f"{skills_sentence}\n"
        f"{experience_sentence}\n"
        f"{education_sentence}\n"
        f"Final score: {round(total_score)}/100 — Decision: {decision}."
    )


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(page_title="CYGNUSA Elite-Hire", layout="centered")
    st.markdown(
        """
        <style>
            .stApp {
                background-color: #ffffff;
                color: #0b1f3a;
            }
            h1, h2, h3, h4, h5, h6, p, label, span, div {
                color: #0b1f3a;
            }
            .stButton > button {
                background-color: #0b1f3a;
                color: #ffffff;
                border-radius: 8px;
                border: none;
            }
            .stTextInput input, .stTextArea textarea, .stFileUploader {
                background-color: #ffffff;
                color: #0b1f3a;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
            }
            .st-emotion-cache-d8lm1x {
                font-family: "Source Sans", sans-serif;
                color: #000 !important;
                white-space: pre-wrap;
                word-break: break-word;
                display: inline-block;
                vertical-align: middle;
                width: 100%;
            }
                .st-emotion-cache-zh4rd8 {
                
                    color: #000 !important;
                    
                    background-color: #ffff !important;
                }

        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("CYGNUSA Elite-Hire")
    st.subheader("Explainable AI-Driven Hiring Evaluation")

    st.markdown(
        "Provide the job role and job description, then upload a resume in PDF format. "
        "The system uses transparent, rule-based scoring to evaluate skills, experience, "
        "and education alignment."
    )

    job_role = st.text_input("Job Role", placeholder="e.g., Data Analyst")
    job_description = st.text_area(
        "Job Description (list skills and requirements)",
        placeholder="Example: Python, SQL, problem solving, 3 years experience, bachelor",
        height=140,
    )

    st.caption(
        "Tip: Separate required skills with commas or new lines for best matching."
    )

    uploaded_file = st.file_uploader("Upload PDF Resume", type=["pdf"])

    if not job_description.strip():
        st.info("Please enter a job description to begin the evaluation.")
        return

    if uploaded_file is None:
        st.info("Please upload a PDF resume to begin the evaluation.")
        return

    resume_text = extract_text_from_pdf(uploaded_file)

    if not resume_text:
        st.warning("We couldn't extract text from that PDF. Try another file.")
        return

    required_skills = extract_required_skills(job_description, DEFAULT_JD_SKILLS)
    matched_skills, skills_score = evaluate_skills(resume_text, required_skills)
    years_found, experience_score = evaluate_experience(
        resume_text, DEFAULT_REQUIRED_EXPERIENCE
    )
    education_match, education_score = evaluate_education(
        resume_text, DEFAULT_REQUIRED_EDUCATION
    )

    total_score = skills_score + experience_score + education_score
    decision = decide(total_score)

    explanation = build_explanation(
        matched_skills,
        len(required_skills),
        years_found,
        DEFAULT_REQUIRED_EXPERIENCE,
        education_match,
        total_score,
        decision,
        job_role,
    )

    st.divider()
    st.header("Results")

    st.metric("Match Score", f"{round(total_score)} / 100")
    st.metric("Decision", decision)
    st.write(f"Role: {job_role or 'Not specified'}")

    st.subheader("Score Breakdown")
    st.write(f"Skills Score: {round(skills_score)} / 50")
    st.write(f"Experience Score: {round(experience_score)} / 30")
    st.write(f"Education Score: {round(education_score)} / 20")

    st.subheader("Explanation")
    st.text(explanation)


if __name__ == "__main__":
    main()
