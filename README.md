# CYGNUSA Elite-Hire

An explainable AI-driven hiring evaluation prototype built for hackathons.

This repository contains a **Streamlit prototype** that analyzes resumes and provides a transparent scoring and decision system for recruiters. It demonstrates how explainability and fairness can be integrated into hiring automation.

---

## 🚀 Live Demo (Hackathon Prototype)

This project was built under time constraints and focuses on **clarity and explainability** rather than complex machine learning models. It provides an end-to-end pipeline from resume upload to decision explanation in a simple UI.

---

## 📌 Features

✔ Upload candidate resume (PDF)  
✔ Extract resume text  
✔ Score candidate based on:
- Skill match  
- Experience match  
- Education match  
✔ Provide a clear decision:
- **Hire**
- **Potential**
- **Reject**  
✔ Generate a human-readable explanation for the decision

This prototype highlights **transparent criteria** rather than black-box AI.

---

## 🧠 How It Works

The system evaluates resumes using a simple, explainable scoring algorithm:

1. **Skill Match (50 pts)** – How many required skills are present in the resume  
2. **Experience Match (30 pts)** – Whether the candidate meets the experience threshold  
3. **Education Match (20 pts)** – Whether the required education level is found  
4. Total score is used to categorize the candidate into one of three decisions.  
5. A breakdown and explanation of the decision is shown on the UI.

---

## 📁 Files

| Filename | Description |
|----------|-------------|
| `app.py` | Main Streamlit app |
| `README.md` | Project documentation |

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- pdfplumber (for PDF text extraction)

---

## 🚧 Installation

Make sure you’re in the project root folder, then run:

```bash
pip install streamlit pdfplumber
