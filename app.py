"""CYGNUSA Elite-Hire: Explainable AI-Driven Hiring Evaluation (Flask)."""

from __future__ import annotations

import re
from typing import List, Tuple

import pdfplumber
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

DEFAULT_JD_SKILLS = ["python", "sql", "data structures", "problem solving"]
DEFAULT_REQUIRED_EXPERIENCE = 3  # years
DEFAULT_REQUIRED_EDUCATION = "bachelor"

# Local job role dataset (technology roles only).
JOB_ROLES = {
    "Full Stack Developer": {
        "skills": [
            "react.js",
            "node.js",
            "express.js",
            "postgresql",
            "rest api development",
            "aws basics",
            "docker",
            "git",
        ],
        "experience": 3,
        "education": "Bachelor in Computer Science or related field",
    },
    "Frontend Developer": {
        "skills": [
            "react.js",
            "typescript",
            "html5",
            "css3",
            "tailwind / bootstrap",
            "state management (redux / context api)",
            "web performance optimization",
        ],
        "experience": 2,
        "education": "Bachelor in Computer Science or related field",
    },
    "Backend Developer": {
        "skills": [
            "node.js",
            "python (django / fastapi)",
            "postgresql / mongodb",
            "microservices architecture",
            "authentication systems",
            "api security",
        ],
        "experience": 3,
        "education": "Bachelor in Computer Science or related field",
    },
    "DevOps Engineer": {
        "skills": [
            "docker",
            "kubernetes",
            "ci/cd pipelines",
            "aws / azure",
            "terraform",
            "monitoring tools (prometheus / grafana)",
            "linux administration",
        ],
        "experience": 3,
        "education": "Bachelor in Computer Science or related field",
    },
    "Data Analyst": {
        "skills": [
            "sql",
            "python (pandas / numpy)",
            "power bi / tableau",
            "data cleaning",
            "statistical analysis",
            "excel advanced functions",
        ],
        "experience": 2,
        "education": "Bachelor in Computer Science, Statistics, or related field",
    },
    "Data Scientist": {
        "skills": [
            "python (scikit-learn, tensorflow, pytorch)",
            "machine learning algorithms",
            "data visualization",
            "feature engineering",
            "model evaluation",
            "statistical modeling",
        ],
        "experience": 3,
        "education": "Bachelor or Master in Data Science, AI, or Computer Science",
    },
    "Cloud Engineer": {
        "skills": [
            "aws / azure / gcp",
            "infrastructure as code",
            "networking fundamentals",
            "cloud security",
            "load balancing",
            "serverless architecture",
        ],
        "experience": 3,
        "education": "Bachelor in Computer Science or related field",
    },
    "Mobile App Developer": {
        "skills": [
            "flutter / react native",
            "android (kotlin / java)",
            "ios development",
            "rest api integration",
            "mobile ui/ux principles",
        ],
        "experience": 2,
        "education": "Bachelor in Computer Science or related field",
    },
}

app = Flask(__name__)
app.secret_key = "cygnusa-elite-hire"


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


# Core logic function wrappers (required names)

def extract_text(pdf_file) -> str:
    """Wrapper for PDF text extraction."""
    return extract_text_from_pdf(pdf_file)


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


# Core logic function wrappers (required names)

def decision(score: float, skill_match_ratio: float) -> str:
    """Wrapper for decision logic with skill protection rule."""
    base_decision = decide(score)
    # Skill protection: strong skill match cannot be rejected due to experience gap.
    if skill_match_ratio >= 0.75 and base_decision == "Reject":
        return "Potential"
    return base_decision


# Core logic function wrappers (required names)

def calculate_score(
    resume_text: str,
    job_description: str,
    role_requirements: dict,
) -> dict:
    """Calculate score and breakdown for a resume based on role requirements."""
    required_skills = role_requirements.get("skills") or extract_required_skills(
        job_description, DEFAULT_JD_SKILLS
    )
    required_experience = role_requirements.get(
        "experience", DEFAULT_REQUIRED_EXPERIENCE
    )
    required_education = role_requirements.get(
        "education", DEFAULT_REQUIRED_EDUCATION
    )

    matched_skills, _ = evaluate_skills(resume_text, required_skills)
    skill_match_ratio = len(matched_skills) / len(required_skills) if required_skills else 0

    # Updated weighting: skills 60, experience 25, education 15.
    skills_score = skill_match_ratio * 60

    years_found = count_years_of_experience(resume_text)
    # Experience scoring with partial credit (not a strict cutoff).
    if years_found >= required_experience:
        experience_score = 25
    elif years_found >= max(required_experience - 1, 0):
        experience_score = 15
    else:
        experience_score = 5

    education_match, _ = evaluate_education(resume_text, required_education)
    education_score = 15 if education_match else 7

    total_score = skills_score + experience_score + education_score

    return {
        "required_skills": required_skills,
        "matched_skills": matched_skills,
        "skills_score": skills_score,
        "skill_match_ratio": skill_match_ratio,
        "years_found": years_found,
        "required_experience": required_experience,
        "experience_score": experience_score,
        "education_score": education_score,
        "education_match": education_match,
        "required_education": required_education,
        "total_score": total_score,
    }


def build_explanation(
    matched_skills: List[str],
    total_skills: int,
    years_found: int,
    experience_required: int,
    education_match: bool,
    total_score: float,
    decision_text: str,
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
        "Skill alignment is the primary factor in this evaluation.\n"
        f"{skills_sentence}\n"
        "Experience is considered but does not automatically disqualify strong skills.\n"
        f"{experience_sentence}\n"
        f"{education_sentence}\n"
        f"Final score: {round(total_score)}/100 — Decision: {decision_text}."
    )


# Core logic function wrappers (required names)

def explain(
    matched_skills: List[str],
    total_skills: int,
    years_found: int,
    experience_required: int,
    education_match: bool,
    total_score: float,
    decision_text: str,
    job_role: str,
) -> str:
    """Wrapper for explanation generation."""
    return build_explanation(
        matched_skills,
        total_skills,
        years_found,
        experience_required,
        education_match,
        total_score,
        decision_text,
        job_role,
    )


@app.route("/")
def index() -> str:
    """Render the upload form."""
    return render_template(
        "index.html",
        roles=sorted(JOB_ROLES.keys()),
        roles_data=JOB_ROLES,
        selected_role=session.get("selected_role", ""),
    )


@app.route("/dashboard")
def dashboard() -> str:
    """Render the employer dashboard overview."""
    selected_role = session.get("selected_role", "")
    role_requirements = JOB_ROLES.get(selected_role, {})
    return render_template(
        "dashboard.html",
        roles=sorted(JOB_ROLES.keys()),
        roles_data=JOB_ROLES,
        selected_role=selected_role,
        role_requirements=role_requirements,
    )


@app.route("/shortlist")
def shortlist() -> str:
    """Render the shortlist page."""
    selected_role = session.get("selected_role", "")
    role_requirements = JOB_ROLES.get(selected_role, {})
    return render_template(
        "shortlist.html",
        roles=sorted(JOB_ROLES.keys()),
        roles_data=JOB_ROLES,
        selected_role=selected_role,
        role_requirements=role_requirements,
    )


@app.route("/evaluation")
def evaluation() -> str:
    """Render the candidate evaluation page."""
    selected_role = session.get("selected_role", "")
    role_requirements = JOB_ROLES.get(selected_role, {})
    return render_template(
        "evaluation.html",
        roles=sorted(JOB_ROLES.keys()),
        roles_data=JOB_ROLES,
        selected_role=selected_role,
        role_requirements=role_requirements,
    )


@app.route("/upload", methods=["POST"])
def upload() -> str:
    """Handle resume uploads and render ranking results."""
    job_role = request.form.get("job_role", "").strip()
    job_description = ""
    if job_role:
        session["selected_role"] = job_role
    role_requirements = JOB_ROLES.get(job_role, {})
    if not role_requirements:
        role_requirements = {
            "skills": DEFAULT_JD_SKILLS,
            "experience": DEFAULT_REQUIRED_EXPERIENCE,
            "education": DEFAULT_REQUIRED_EDUCATION,
        }

    uploaded_files = request.files.getlist("resumes")
    if not uploaded_files:
        return redirect(url_for("index"))

    results = []
    warnings = []

    for uploaded_file in uploaded_files:
        resume_text = extract_text(uploaded_file)
        if not resume_text:
            warnings.append(
                f"We couldn't extract text from {uploaded_file.filename}."
            )
            continue

        score_data = calculate_score(resume_text, job_description, role_requirements)
        decision_text = decision(
            score_data["total_score"], score_data["skill_match_ratio"]
        )
        explanation = explain(
            score_data["matched_skills"],
            len(score_data["required_skills"]),
            score_data["years_found"],
            score_data["required_experience"],
            score_data["education_match"],
            score_data["total_score"],
            decision_text,
            job_role,
        )

        results.append(
            {
                "name": uploaded_file.filename,
                "score": round(score_data["total_score"]),
                "decision": decision_text,
                "explanation": explanation,
                "breakdown": {
                    "skills": round(score_data["skills_score"]),
                    "experience": round(score_data["experience_score"]),
                    "education": round(score_data["education_score"]),
                },
            }
        )

    ranked_results = sorted(results, key=lambda item: item["score"], reverse=True)

    return render_template(
        "results.html",
        results=ranked_results,
        warnings=warnings,
        job_role=job_role,
        roles=sorted(JOB_ROLES.keys()),
        roles_data=JOB_ROLES,
        selected_role=job_role,
        role_requirements=role_requirements,
        total_candidates=len(ranked_results),
    )


@app.route("/set-role", methods=["POST"])
def set_role() -> str:
    """Persist selected role in session for dashboard pages."""
    payload = request.get_json(silent=True) or {}
    selected_role = payload.get("role", "")
    if selected_role in JOB_ROLES:
        session["selected_role"] = selected_role
        return jsonify({"status": "ok"})
    return jsonify({"status": "invalid role"}), 400


if __name__ == "__main__":
    app.run(debug=True)
