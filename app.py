"""CYGNUSA Elite-Hire: Explainable AI-Driven Hiring Evaluation (Flask)."""

from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

import pdfplumber
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

DEFAULT_JD_SKILLS = ["python", "sql", "data structures", "problem solving"]
DEFAULT_REQUIRED_EXPERIENCE = 3  # years
DEFAULT_REQUIRED_EDUCATION = "bachelor"
DEFAULT_FUNDAMENTALS = [
    "Role fundamentals not specified.",
]

# Local job role dataset (technology roles only).
JOB_ROLES = {
    "Full Stack Developer": {
        "fundamentals": [
            "Frontend + Backend integration understanding",
            "API development and integration",
            "Database interaction fundamentals",
            "Version control knowledge",
            "Basic cloud deployment knowledge",
        ],
        "skills": ["react.js", "node.js", "sql/nosql", "rest apis", "git"],
        "experience": 3,
        "education": "Computer Science / Software Engineering",
    },
    "Frontend Developer": {
        "fundamentals": [
            "UI/UX implementation",
            "Responsive design principles",
            "Browser rendering optimization",
            "Client-side performance understanding",
        ],
        "skills": ["react.js", "typescript", "html5", "css3", "ui frameworks"],
        "experience": 2,
        "education": "Computer Science / IT",
    },
    "Backend Developer": {
        "fundamentals": [
            "Server-side architecture",
            "Authentication and authorization",
            "Database design principles",
            "API security fundamentals",
        ],
        "skills": ["node.js", "python", "sql/nosql", "microservices"],
        "experience": 3,
        "education": "Computer Science",
    },
    "DevOps Engineer": {
        "fundamentals": [
            "CI/CD lifecycle understanding",
            "Infrastructure automation",
            "Monitoring and logging principles",
            "Containerization fundamentals",
        ],
        "skills": ["docker", "kubernetes", "cloud platforms", "terraform"],
        "experience": 3,
        "education": "Computer Science / IT",
    },
    "Data Analyst": {
        "fundamentals": [
            "Data interpretation",
            "Statistical reasoning",
            "Business data visualization",
            "Data cleaning techniques",
        ],
        "skills": ["sql", "python", "tableau/power bi", "excel"],
        "experience": 2,
        "education": "Statistics / Computer Science",
    },
    "Cloud Engineer": {
        "fundamentals": [
            "Cloud infrastructure design",
            "Networking fundamentals",
            "Security and access management",
            "Load balancing and scalability",
        ],
        "skills": ["aws/azure/gcp", "infrastructure as code", "networking"],
        "experience": 3,
        "education": "Computer Science",
    },
    "Mobile App Developer": {
        "fundamentals": [
            "Mobile architecture principles",
            "API integration",
            "Platform-specific UI optimization",
            "Performance tuning",
        ],
        "skills": ["flutter", "react native", "kotlin/swift"],
        "experience": 2,
        "education": "Computer Science",
    },
    "Software Engineer": {
        "fundamentals": [
            "Problem solving and debugging",
            "Data structures and algorithms",
            "System design exposure",
            "Code quality and testing fundamentals",
        ],
        "skills": ["python", "java", "data structures", "algorithms", "git"],
        "experience": 2,
        "education": "Computer Science / Software Engineering",
    },
    "Backend Microservices Engineer": {
        "fundamentals": [
            "API scalability",
            "Distributed systems thinking",
            "Event driven architecture",
            "Service observability",
        ],
        "skills": ["microservices", "kafka", "node.js", "python", "sql/nosql"],
        "experience": 3,
        "education": "Computer Science",
    },
    "Platform Engineer": {
        "fundamentals": [
            "Infrastructure automation",
            "Cloud reliability",
            "Observability systems",
            "Service resilience practices",
        ],
        "skills": ["terraform", "kubernetes", "cloud platforms", "monitoring", "ci/cd"],
        "experience": 3,
        "education": "Computer Science / IT",
    },
    "AI/ML Engineer": {
        "fundamentals": [
            "Model development lifecycle",
            "Data pipelines",
            "ML deployment practices",
            "Experiment tracking",
        ],
        "skills": ["python", "machine learning", "model deployment", "feature engineering"],
        "experience": 3,
        "education": "Computer Science / Data Science",
    },
    "QA Automation Engineer": {
        "fundamentals": [
            "Test automation strategy",
            "CI testing pipelines",
            "Performance testing",
            "Defect triage",
        ],
        "skills": ["selenium", "cypress", "api testing", "ci/cd", "python"],
        "experience": 2,
        "education": "Computer Science / IT",
    },
    "Site Reliability Engineer": {
        "fundamentals": [
            "System uptime focus",
            "Monitoring and alerting",
            "Failure recovery",
            "Incident response",
        ],
        "skills": ["monitoring", "linux", "cloud platforms", "sre", "automation"],
        "experience": 3,
        "education": "Computer Science / IT",
    },
}

# Skill aliases for stronger matching.
SKILL_ALIASES = {
    "react.js": ["react", "reactjs"],
    "node.js": ["node", "nodejs"],
    "postgresql": ["postgres", "psql"],
    "aws": ["amazon web services", "aws cloud"],
    "aws / azure / gcp": ["amazon web services", "azure", "gcp", "google cloud"],
    "aws / azure": ["amazon web services", "azure"],
    "rest api development": ["rest api", "restful api"],
    "ci/cd pipelines": ["ci cd", "continuous integration", "continuous delivery"],
    "docker": ["docker container", "containerization"],
    "kubernetes": ["k8s"],
    "python (django / fastapi)": ["python", "django", "fastapi"],
    "postgresql / mongodb": ["postgresql", "postgres", "mongodb", "mongo"],
    "power bi / tableau": ["power bi", "tableau"],
    "python (pandas / numpy)": ["python", "pandas", "numpy"],
    "python (scikit-learn, tensorflow, pytorch)": [
        "python",
        "scikit-learn",
        "tensorflow",
        "pytorch",
    ],
    "flutter / react native": ["flutter", "react native"],
    "android (kotlin / java)": ["android", "kotlin", "java"],
    "sql/nosql": ["sql", "nosql", "mongodb", "postgresql", "mysql"],
    "rest apis": ["rest api", "restful api", "api integration"],
    "ui frameworks": ["bootstrap", "tailwind", "material ui"],
    "tableau/power bi": ["tableau", "power bi"],
    "cloud platforms": ["aws", "azure", "gcp", "amazon web services", "google cloud"],
    "aws/azure/gcp": ["aws", "azure", "gcp", "amazon web services", "google cloud"],
    "kotlin/swift": ["kotlin", "swift", "ios", "android"],
    "react native": ["reactnative", "react-native"],
    "machine learning": ["ml", "machine learning", "deep learning"],
    "model deployment": ["mlops", "deployment", "model serving"],
    "monitoring": ["prometheus", "grafana", "monitoring", "observability"],
    "ci/cd": ["ci/cd", "continuous integration", "continuous delivery"],
    "api testing": ["postman", "api testing"],
}

SKILL_CLUSTERS = {
    "frontend": ["react", "typescript", "html", "css", "ui", "frontend"],
    "backend": ["node", "python", "java", "microservices", "api", "backend"],
    "cloud/devops": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd"],
    "database": ["sql", "nosql", "postgres", "mongodb", "mysql"],
    "programming fundamentals": ["data structures", "algorithms", "problem solving"],
}

DEGREE_KEYWORDS = [
    "bachelor",
    "master",
    "phd",
    "b.sc",
    "m.sc",
    "bachelor of science",
    "master of science",
    "computer science",
    "software engineering",
    "information technology",
]
CERT_KEYWORDS = [
    "aws certified",
    "azure certified",
    "google cloud certified",
    "ccna",
    "cka",
    "ckad",
    "pmp",
    "scrum master",
    "oracle certified",
]
COURSE_KEYWORDS = [
    "coursera",
    "udemy",
    "edx",
    "pluralsight",
    "linkedin learning",
    "online course",
    "certificate",
]
PROJECT_EVIDENCE_KEYWORDS = [
    "github",
    "portfolio",
    "hackathon",
    "open source",
    "competitive coding",
    "codeforces",
    "leetcode",
    "deployed",
    "deployment",
    "production",
    "ci/cd",
    "pipeline",
    "iot",
]
PROJECT_TECH_KEYWORDS = [
    "react",
    "node",
    "django",
    "fastapi",
    "kubernetes",
    "docker",
    "aws",
    "azure",
    "gcp",
    "terraform",
    "rest api",
]
BUZZWORDS = [
    "results-driven",
    "dynamic professional",
    "highly motivated",
    "leveraged cutting edge",
    "innovative solutions",
    "synergy",
]

app = Flask(__name__)
app.secret_key = "cygnusa-elite-hire"


@app.context_processor
def inject_counts() -> dict:
    """Provide candidate counts for sidebar badges."""
    role_results = session.get("role_results", {})
    selected_role = session.get("selected_role", "")
    candidates = role_results.get(selected_role, session.get("last_results", []))
    return {
        "candidate_counts": {
            "hire": sum(1 for candidate in candidates if candidate.get("decision") == "Hire"),
            "potential": sum(
                1 for candidate in candidates if candidate.get("decision") == "Potential"
            ),
            "reject": sum(
                1 for candidate in candidates if candidate.get("decision") == "Reject"
            ),
        }
    }

def normalize_text(text: str) -> str:
    """Normalize text for consistent matching."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    normalized = normalized.replace("\r", " ").replace("\t", " ")
    normalized = re.sub(r"-\s*\n", "", normalized)
    normalized = normalized.replace("\n", " ")
    normalized = normalized.replace("•", " ").replace("·", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip().lower()


def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from a PDF file-like object and normalize to lowercase."""
    if pdf_file is None:
        return ""

    try:
        with pdfplumber.open(pdf_file) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return normalize_text("\n".join(pages))
    except Exception:
        return ""


def get_role_requirements(role: str) -> dict:
    """Return role requirements with employer overrides applied."""
    base_requirements = JOB_ROLES.get(role, {})
    custom_roles = session.get("custom_roles", {})
    custom = custom_roles.get(role, {})
    return {
        "fundamentals": custom.get("fundamentals")
        or base_requirements.get("fundamentals", DEFAULT_FUNDAMENTALS),
        "skills": custom.get("skills")
        or base_requirements.get("skills", DEFAULT_JD_SKILLS),
        "experience": custom.get(
            "experience", base_requirements.get("experience", DEFAULT_REQUIRED_EXPERIENCE)
        ),
        "education": custom.get(
            "education", base_requirements.get("education", DEFAULT_REQUIRED_EDUCATION)
        ),
    }


def get_default_role_requirements(role: str) -> dict:
    """Return role requirements without applying overrides (used for upload form)."""
    base_requirements = JOB_ROLES.get(role, {})
    return {
        "fundamentals": base_requirements.get("fundamentals", DEFAULT_FUNDAMENTALS),
        "skills": base_requirements.get("skills", DEFAULT_JD_SKILLS),
        "experience": base_requirements.get(
            "experience", DEFAULT_REQUIRED_EXPERIENCE
        ),
        "education": base_requirements.get(
            "education", DEFAULT_REQUIRED_EDUCATION
        ),
    }


def extract_education_and_certifications(text: str) -> dict:
    """Extract degree mentions, certifications, and course references."""
    normalized_text = normalize_text(text)
    degrees = [keyword for keyword in DEGREE_KEYWORDS if keyword in normalized_text]
    certifications = [keyword for keyword in CERT_KEYWORDS if keyword in normalized_text]
    courses = [keyword for keyword in COURSE_KEYWORDS if keyword in normalized_text]
    return {
        "degrees": sorted(set(degrees)),
        "certifications": sorted(set(certifications)),
        "courses": sorted(set(courses)),
    }


def extract_resume_evidence(text: str) -> list[str]:
    """Extract notable resume evidence signals."""
    normalized_text = normalize_text(text)
    evidence = [keyword for keyword in PROJECT_EVIDENCE_KEYWORDS if keyword in normalized_text]
    return sorted(set(evidence))


def extract_project_evidence(text: str) -> dict:
    """Extract project technology and deployment evidence."""
    normalized_text = normalize_text(text)
    technologies = [
        keyword for keyword in PROJECT_TECH_KEYWORDS if keyword in normalized_text
    ]
    deployments = [
        keyword
        for keyword in ["deployed", "deployment", "production", "ci/cd", "pipeline"]
        if keyword in normalized_text
    ]
    return {
        "technologies": sorted(set(technologies)),
        "deployments": sorted(set(deployments)),
    }


def detect_skill_alias(skill: str, text: str) -> bool:
    """Detect a skill via aliases and exact matches."""
    normalized_text = normalize_text(text)
    tokens = set(normalized_text.split())
    normalized_skill = skill.lower()
    aliases = SKILL_ALIASES.get(normalized_skill, [])
    candidates = [normalized_skill, *aliases]
    return any(
        candidate in normalized_text
        if any(char in candidate for char in [" ", ".", "/"])
        else candidate in tokens
        for candidate in candidates
    )


def detect_skill_clusters(text: str) -> tuple[list[str], list[str]]:
    """Detect major skill clusters to prevent false rejection."""
    normalized_text = normalize_text(text)
    tokens = set(normalized_text.split())
    matched = []
    missing = []
    for cluster, keywords in SKILL_CLUSTERS.items():
        if any(
            keyword in normalized_text
            if any(char in keyword for char in [" ", ".", "/"])
            else keyword in tokens
            for keyword in keywords
        ):
            matched.append(cluster)
        else:
            missing.append(cluster)
    return matched, missing


def detect_ai_likelihood(text: str) -> dict:
    """Probabilistic AI likelihood analysis for resume text."""
    normalized_text = normalize_text(text)
    sentences = [s.strip() for s in re.split(r"[.!?]+", normalized_text) if s.strip()]
    words = normalized_text.split()
    if not words:
        return {
            "score": 50,
            "label": "Possibly AI Assisted",
            "reasons": ["Limited content available for analysis."],
            "signals": {
                "uniformity": 0,
                "buzzword_density": 0,
                "project_detail": 0,
                "structure_repetition": 0,
                "skill_context": 0,
            },
        }

    sentence_lengths = [len(sentence.split()) for sentence in sentences] or [0]
    avg_length = sum(sentence_lengths) / max(len(sentence_lengths), 1)
    variance = sum((length - avg_length) ** 2 for length in sentence_lengths) / max(
        len(sentence_lengths), 1
    )
    uniformity = 1 / (1 + variance)

    buzzword_hits = sum(1 for phrase in BUZZWORDS if phrase in normalized_text)
    buzzword_density = buzzword_hits / max(len(sentences), 1)

    project_signals = [
        "deployed",
        "deployment",
        "latency",
        "throughput",
        "metrics",
        "version",
        "implemented",
        "built",
    ]
    project_detail = sum(1 for phrase in project_signals if phrase in normalized_text)

    structure_patterns = ["developed", "built", "led", "designed", "implemented"]
    structure_repetition = sum(
        1 for sentence in sentences if any(sentence.startswith(pattern) for pattern in structure_patterns)
    )
    structure_repetition_ratio = structure_repetition / max(len(sentences), 1)

    skill_context_indicators = ["using", "with", "by", "to", "for"]
    context_hits = sum(1 for word in skill_context_indicators if word in words)
    skill_context_score = context_hits / max(len(words), 1)

    score = 0
    reasons = []
    if uniformity > 0.25:
        score += 25
        reasons.append("Highly uniform sentence structure detected.")
    if buzzword_density > 0.2:
        score += 20
        reasons.append("High buzzword density detected.")
    if project_detail < 2:
        score += 20
        reasons.append("Limited project detail evidence.")
    if structure_repetition_ratio > 0.4:
        score += 15
        reasons.append("Repeating achievement sentence structures detected.")
    if skill_context_score < 0.03:
        score += 20
        reasons.append("Skills listed with limited usage context.")

    score = min(100, score)
    if score >= 70:
        label = "Highly AI Generated"
    elif score >= 40:
        label = "Possibly AI Assisted"
    else:
        label = "Likely Human"

    return {
        "score": score,
        "label": label,
        "reasons": reasons or ["No strong AI signals detected."],
        "signals": {
            "uniformity": round(uniformity, 2),
            "buzzword_density": round(buzzword_density, 2),
            "project_detail": project_detail,
            "structure_repetition": round(structure_repetition_ratio, 2),
            "skill_context": round(skill_context_score, 2),
        },
    }


def evaluate_skill_clusters(text: str) -> tuple[list[str], list[str]]:
    """Evaluate skill clusters to reduce over-rejection."""
    normalized_text = normalize_text(text)
    tokens = set(normalized_text.split())
    matched = []
    missing = []
    for cluster, keywords in SKILL_CLUSTERS.items():
        if any(
            keyword in normalized_text
            if any(char in keyword for char in [" ", ".", "/"])
            else keyword in tokens
            for keyword in keywords
        ):
            matched.append(cluster)
        else:
            missing.append(cluster)
    return matched, missing


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


def evaluate_required_skills(text: str, required_skills: List[str]) -> Tuple[List[str], List[str], float]:
    """Evaluate required skills using exact and alias matching."""
    matched = []
    missing = []

    for skill in required_skills:
        if detect_skill_alias(skill, text):
            matched.append(skill)
        else:
            missing.append(skill)

    skill_match_ratio = len(matched) / len(required_skills) if required_skills else 0
    return matched, missing, skill_match_ratio


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


def enforce_cluster_protection(
    decision_text: str,
    matched_clusters: list[str],
    project_evidence: list[str],
    score: float,
) -> str:
    """Ensure strong cluster alignment prevents outright rejection."""
    if len(matched_clusters) >= 2 and decision_text == "Reject":
        return "Potential"
    if project_evidence and matched_clusters and decision_text == "Reject" and score >= 40:
        return "Potential"
    return decision_text


def calculate_adaptive_score(
    skill_match_ratio: float,
    years_found: int,
    required_experience: int,
    education_match: bool,
    certification_bonus: int,
    course_bonus: int,
    project_evidence: dict,
) -> tuple[float, float, float, float]:
    """Calculate weighted adaptive score and return component scores."""
    skills_score = skill_match_ratio * 60
    if years_found >= required_experience:
        experience_score = 25
    elif skill_match_ratio > 0.7:
        experience_score = 20
    elif years_found >= max(required_experience - 1, 0):
        experience_score = 15
    else:
        experience_score = 5

    education_score = 15 if education_match else 7
    evidence_bonus = 2 if project_evidence["deployments"] else 0
    total_score = skills_score + experience_score + education_score
    total_score = min(
        100, total_score + certification_bonus + course_bonus + evidence_bonus
    )
    return total_score, skills_score, experience_score, education_score


def generate_analytics_summary(candidates: list[dict]) -> dict:
    """Generate recruiter analytics for dashboards."""
    total = len(candidates)
    average_score = (
        round(sum(candidate["score"] for candidate in candidates) / total) if total else 0
    )
    missing_skills: dict[str, int] = {}
    missing_clusters: dict[str, int] = {}
    skill_match_total = 0
    for candidate in candidates:
        skill_match_total += candidate.get("skill_match_ratio", 0)
        for skill in candidate.get("missing_skills", []):
            missing_skills[skill] = missing_skills.get(skill, 0) + 1
        for cluster in candidate.get("missing_clusters", []):
            missing_clusters[cluster] = missing_clusters.get(cluster, 0) + 1

    readiness = (
        round((skill_match_total / total) * 100) if total else 0
    )
    return {
        "average_score": average_score,
        "missing_skills": sorted(missing_skills.items(), key=lambda item: item[1], reverse=True),
        "missing_clusters": sorted(
            missing_clusters.items(), key=lambda item: item[1], reverse=True
        ),
        "readiness": readiness,
    }


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

    matched_skills, missing_skills, skill_match_ratio = evaluate_required_skills(
        resume_text, required_skills
    )

    education_evidence = extract_education_and_certifications(resume_text)
    certification_bonus = 5 * len(education_evidence["certifications"])
    course_bonus = 2 * len(education_evidence["courses"])
    project_evidence = extract_project_evidence(resume_text)
    years_found = count_years_of_experience(resume_text)
    education_match, _ = evaluate_education(resume_text, required_education)
    total_score, skills_score, experience_score, education_score = calculate_adaptive_score(
        skill_match_ratio,
        years_found,
        required_experience,
        education_match,
        certification_bonus,
        course_bonus,
        project_evidence,
    )

    return {
        "required_skills": required_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skills_score": skills_score,
        "skill_match_ratio": skill_match_ratio,
        "years_found": years_found,
        "required_experience": required_experience,
        "experience_score": experience_score,
        "education_score": education_score,
        "education_match": education_match,
        "required_education": required_education,
        "education_evidence": education_evidence,
        "certification_bonus": certification_bonus,
        "course_bonus": course_bonus,
        "project_evidence": project_evidence,
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
    matched_clusters: list[str],
    project_evidence: list[str],
    certification_count: int,
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
    clusters_sentence = (
        f"Matched skill clusters: {', '.join(matched_clusters)}."
        if matched_clusters
        else "Skill clusters need reinforcement."
    )
    project_sentence = (
        f"Project evidence detected: {', '.join(project_evidence)}."
        if project_evidence
        else "No project/deployment evidence detected."
    )
    certification_sentence = (
        "Certifications present in resume."
        if certification_count
        else "No certifications detected."
    )
    flexibility_sentence = (
        "Experience gap impact reduced due to strong skill alignment."
        if total_skills and len(matched_skills) / total_skills > 0.7
        else "Experience alignment considered with role requirements."
    )

    return (
        f"Role evaluated: {job_role or 'Role not specified'}.\n"
        "Skill alignment is the primary factor in this evaluation.\n"
        f"{skills_sentence}\n"
        f"{clusters_sentence}\n"
        f"{flexibility_sentence}\n"
        f"{experience_sentence}\n"
        f"{education_sentence}\n"
        f"{certification_sentence}\n"
        f"{project_sentence}\n"
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
    matched_clusters: list[str],
    project_evidence: list[str],
    certification_count: int,
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
        matched_clusters,
        project_evidence,
        certification_count,
    )


@app.route("/")
def index() -> str:
    """Render the upload form."""
    selected_role = session.get("selected_role", "")
    return render_template(
        "index.html",
        roles=sorted(JOB_ROLES.keys()),
        roles_data=JOB_ROLES,
        selected_role=selected_role,
        role_requirements=get_default_role_requirements(selected_role),
        active_page="upload",
    )


@app.route("/dashboard")
def dashboard() -> str:
    """Render the employer dashboard overview."""
    selected_role = session.get("selected_role", "")
    role_requirements = get_role_requirements(selected_role)
    role_results = session.get("role_results", {})
    candidates = role_results.get(selected_role, session.get("last_results", []))
    analytics = generate_analytics_summary(candidates)
    counts = {
        "total": len(candidates),
        "hire": sum(1 for candidate in candidates if candidate["decision"] == "Hire"),
        "potential": sum(
            1 for candidate in candidates if candidate["decision"] == "Potential"
        ),
        "reject": sum(
            1 for candidate in candidates if candidate["decision"] == "Reject"
        ),
    }
    return render_template(
        "dashboard.html",
        roles=sorted(JOB_ROLES.keys()),
        roles_data=JOB_ROLES,
        selected_role=selected_role,
        role_requirements=role_requirements,
        counts=counts,
        analytics=analytics,
        active_page="dashboard",
    )


@app.route("/shortlist")
def shortlist() -> str:
    """Render the shortlist page."""
    selected_role = session.get("selected_role", "")
    role_requirements = get_role_requirements(selected_role)
    role_results = session.get("role_results", {})
    candidates = role_results.get(selected_role, session.get("last_results", []))
    sort_key = request.args.get("sort", "score")
    sort_order = request.args.get("order", "desc")
    reverse = sort_order == "desc"
    if sort_key == "decision":
        decision_rank = {"Hire": 3, "Potential": 2, "Reject": 1}
        candidates = sorted(
            candidates,
            key=lambda item: decision_rank.get(item["decision"], 0),
            reverse=reverse,
        )
    else:
        candidates = sorted(
            candidates, key=lambda item: item.get("score", 0), reverse=reverse
        )
    return render_template(
        "shortlist.html",
        roles=sorted(JOB_ROLES.keys()),
        roles_data=JOB_ROLES,
        selected_role=selected_role,
        role_requirements=role_requirements,
        candidates=candidates,
        sort_key=sort_key,
        sort_order=sort_order,
        active_page="shortlist",
    )


@app.route("/evaluation")
def evaluation() -> str:
    """Render the candidate evaluation page."""
    selected_role = session.get("selected_role", "")
    role_requirements = get_role_requirements(selected_role)
    return render_template(
        "evaluation.html",
        roles=sorted(JOB_ROLES.keys()),
        roles_data=JOB_ROLES,
        selected_role=selected_role,
        role_requirements=role_requirements,
        candidates=session.get("last_results", []),
        active_page="evaluation",
    )


@app.route("/candidates")
def candidates() -> str:
    """Render candidate detail page."""
    selected_role = session.get("selected_role", "")
    role_requirements = get_role_requirements(selected_role)
    return render_template(
        "candidates.html",
        roles=sorted(JOB_ROLES.keys()),
        roles_data=JOB_ROLES,
        selected_role=selected_role,
        role_requirements=role_requirements,
        candidates=session.get("last_results", []),
        active_page="candidates",
    )


@app.route("/roles")
def roles() -> str:
    """Render the job roles page."""
    selected_role = session.get("selected_role", "")
    role_requirements = get_default_role_requirements(selected_role)
    return render_template(
        "roles.html",
        roles=sorted(JOB_ROLES.keys()),
        roles_data=JOB_ROLES,
        selected_role=selected_role,
        role_requirements=role_requirements,
        active_page="roles",
    )


@app.route("/roles/update", methods=["POST"])
def update_role() -> str:
    """Backward-compatible route (editing moved to upload page)."""
    return redirect(url_for("index"))


@app.route("/insights")
def insights() -> str:
    """Render the insights page."""
    selected_role = session.get("selected_role", "")
    role_results = session.get("role_results", {})
    candidates = role_results.get(selected_role, session.get("last_results", []))
    total_candidates = len(candidates)
    average_score = (
        round(sum(candidate["score"] for candidate in candidates) / total_candidates)
        if total_candidates
        else 0
    )
    missing_skill_counts: dict[str, int] = {}
    education_matches = 0
    cert_matches = 0
    for candidate in candidates:
        for skill in candidate.get("missing_skills", []):
            missing_skill_counts[skill] = missing_skill_counts.get(skill, 0) + 1
        if candidate.get("education_match"):
            education_matches += 1
        if candidate.get("education_evidence", {}).get("certifications"):
            cert_matches += 1

    top_missing_skills = sorted(
        missing_skill_counts.items(), key=lambda item: item[1], reverse=True
    )
    role_requirements = get_role_requirements(selected_role)
    education_alignment = (
        round((education_matches / total_candidates) * 100) if total_candidates else 0
    )
    certification_presence = (
        round((cert_matches / total_candidates) * 100) if total_candidates else 0
    )
    return render_template(
        "insights.html",
        roles=sorted(JOB_ROLES.keys()),
        roles_data=JOB_ROLES,
        selected_role=selected_role,
        role_requirements=role_requirements,
        total_candidates=total_candidates,
        average_score=average_score,
        top_missing_skills=top_missing_skills,
        education_alignment=education_alignment,
        certification_presence=certification_presence,
        active_page="insights",
    )


@app.route("/upload", methods=["POST"])
def upload() -> str:
    """Handle resume uploads and render ranking results."""
    job_role = request.form.get("job_role", "").strip()
    job_description = ""
    if job_role:
        session["selected_role"] = job_role
    default_requirements = get_default_role_requirements(job_role)
    # JD override workflow: accept employer edits from the upload form only.
    skills_raw = request.form.get("skills", "")
    fundamentals_raw = request.form.get("fundamentals", "")
    experience_raw = request.form.get("experience", "").strip()
    education = request.form.get("education", "").strip()

    skills = [skill.strip().lower() for skill in skills_raw.split(",") if skill.strip()]
    fundamentals = [
        item.strip() for item in fundamentals_raw.split("\n") if item.strip()
    ]
    experience = int(experience_raw) if experience_raw.isdigit() else default_requirements["experience"]
    education_value = education or default_requirements["education"]

    role_requirements = {
        "skills": skills or default_requirements["skills"],
        "fundamentals": fundamentals or default_requirements["fundamentals"],
        "experience": experience,
        "education": education_value,
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

        matched_clusters, missing_clusters = detect_skill_clusters(resume_text)
        project_evidence = extract_resume_evidence(resume_text)
        authenticity = detect_ai_likelihood(resume_text)
        score_data = calculate_score(resume_text, job_description, role_requirements)
        decision_text = decision(
            score_data["total_score"], score_data["skill_match_ratio"]
        )
        decision_text = enforce_cluster_protection(
            decision_text,
            matched_clusters,
            project_evidence,
            score_data["total_score"],
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
            matched_clusters,
            project_evidence,
            len(score_data["education_evidence"]["certifications"]),
        )

        results.append(
            {
                "name": uploaded_file.filename,
                "score": round(score_data["total_score"]),
                "decision": decision_text,
                "explanation": explanation,
                "skill_match_ratio": round(score_data["skill_match_ratio"], 2),
                "matched_skills": score_data["matched_skills"],
                "missing_skills": score_data["missing_skills"],
                "matched_clusters": matched_clusters,
                "missing_clusters": missing_clusters,
                "project_evidence": project_evidence,
                "project_detail": score_data["project_evidence"],
                "authenticity": authenticity,
                "education_match": score_data["education_match"],
                "education_evidence": score_data["education_evidence"],
                "certification_bonus": score_data["certification_bonus"],
                "course_bonus": score_data["course_bonus"],
                "breakdown": {
                    "skills": round(score_data["skills_score"]),
                    "experience": round(score_data["experience_score"]),
                    "education": round(score_data["education_score"]),
                },
                "role": job_role or "Unspecified",
            }
        )

    ranked_results = sorted(results, key=lambda item: item["score"], reverse=True)
    session["last_results"] = ranked_results
    role_results = session.get("role_results", {})
    role_key = job_role or "Unspecified"
    role_results[role_key] = ranked_results
    session["role_results"] = role_results

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
        active_page="candidates",
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
