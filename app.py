"""CYGNUSA Elite-Hire: Explainable AI-Driven Hiring Evaluation."""

from __future__ import annotations

import re
from typing import List, Tuple

import pdfplumber
import streamlit as st

JD_SKILLS = ["python", "sql", "data structures", "problem solving"]
REQUIRED_EXPERIENCE = 3  # years
REQUIRED_EDUCATION = "bachelor"


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
        f"{skills_sentence}\n"
        f"{experience_sentence}\n"
        f"{education_sentence}\n"
        f"Final score: {round(total_score)}/100 — Decision: {decision}."
    )


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(page_title="CYGNUSA Elite-Hire", layout="centered")

    st.title("CYGNUSA Elite-Hire")
    st.subheader("Explainable AI-Driven Hiring Evaluation")

    st.markdown(
        "Upload a resume in PDF format. The system uses transparent, rule-based scoring "
        "to evaluate skills, experience, and education alignment."
    )

    uploaded_file = st.file_uploader("Upload PDF Resume", type=["pdf"])

    if uploaded_file is None:
        st.info("Please upload a PDF resume to begin the evaluation.")
        return

    resume_text = extract_text_from_pdf(uploaded_file)

    if not resume_text:
        st.warning("We couldn't extract text from that PDF. Try another file.")
        return

    matched_skills, skills_score = evaluate_skills(resume_text, JD_SKILLS)
    years_found, experience_score = evaluate_experience(
        resume_text, REQUIRED_EXPERIENCE
    )
    education_match, education_score = evaluate_education(
        resume_text, REQUIRED_EDUCATION
    )

    total_score = skills_score + experience_score + education_score
    decision = decide(total_score)

    explanation = build_explanation(
        matched_skills,
        len(JD_SKILLS),
        years_found,
        REQUIRED_EXPERIENCE,
        education_match,
        total_score,
        decision,
    )

    st.divider()
    st.header("Results")

    st.metric("Match Score", f"{round(total_score)} / 100")
    st.metric("Decision", decision)

    st.subheader("Score Breakdown")
    st.write(f"Skills Score: {round(skills_score)} / 50")
    st.write(f"Experience Score: {round(experience_score)} / 30")
    st.write(f"Education Score: {round(education_score)} / 20")

    st.subheader("Explanation")
    st.text(explanation)


if __name__ == "__main__":
    main()
