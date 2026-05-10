# How to Run Each File — Step by Step

All commands are run from the project root:
  cd "d:\SEM 6\AI\AI PROJ\AI-RC-Quiz-Generator"

Run them IN ORDER for a fresh setup.

## STEP 1 — Preprocessing (creates data/processed/ matrices)
venv\Scripts\python.exe src\preprocessing.py

## STEP 2 — Train Model A (trains LR, SVM, NB, KMeans, Ensemble)
venv\Scripts\python.exe src\model_a_train.py

## STEP 3 — Run the Streamlit UI
venv\Scripts\python.exe -m streamlit run ui\app.py

---

## Optional / Individual Files

# Test the Model A Question Generator only
venv\Scripts\python.exe src\model_a_gen.py

# Test Model B distractor + hint generation only
venv\Scripts\python.exe src\model_b.py

# Run main pipeline (if exists)
venv\Scripts\python.exe src\main.py
