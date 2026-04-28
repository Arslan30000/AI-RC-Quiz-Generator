# Intelligent Reading Comprehension and Quiz Generation System

## Overview
This project is an AI-powered Reading Comprehension and Quiz Generation System built using the RACE (ReAding Comprehension from Examinations) dataset. The system automatically generates comprehension questions, predicts correct answers, creates distractor options, and provides hints.

This project is divided into three main components:
* **Model A (Q&A Generator / Verifier):** Uses traditional ML models to verify answers and apply template-based question generation.
* **Model B (Distractor & Hint Generator):** Generates plausible but incorrect multiple-choice options and extracts graduated hints to assist the user.
* **UI Layer:** An interactive Streamlit application that wires both pipelines together.

## Authors
* **Muhammad Arslan** (23i-0572)
* **Masab Tahir** (23i-0006)
* FAST NUCES, Islamabad - BS Computer Science

---

## Getting Started (Windows PowerShell Setup)

Follow these instructions to set up the project from scratch in a terminal environment. 

### 1. Clone the Repository
```powershell
git clone [https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git](https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git)
cd YOUR-REPO-NAME
2. Create and Activate the Virtual Environment
Create an isolated Python environment to manage dependencies:
python -m venv venv

Activate the virtual environment:
.\venv\Scripts\Activate.ps1

3. Install Dependencies
Install all required machine learning and UI libraries:
pip install -r requirements.txt

4. Run the Application
Once the models are trained and saved, launch the interactive user interface using Streamlit:
streamlit run ui/app.py