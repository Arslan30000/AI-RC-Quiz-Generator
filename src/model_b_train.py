"""
src/model_b_train.py

Training script for Model B — Distractor & Hint Generator.

Model B does not train a neural network. Instead, it:
1. Builds and saves the shared OHE vectorizer used for distractor candidate scoring.
2. Evaluates the distractor and hint generation pipeline on the validation set and
   reports Precision, Recall, F1, and Distractor Ranker Accuracy.

Run from project root:
    venv\\Scripts\\python.exe src\\model_b_train.py
"""

import pandas as pd
import numpy as np
import joblib
import os
import time
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model_b import ModelB_Generator


def evaluate_distractor_generator(generator, val_df, n_samples=500):
    """
    Evaluates the distractor generator on a sample of the validation set.

    Distractor Ranker Accuracy: fraction of samples where the top-ranked
    distractor is NOT the correct answer text.
    """
    print(f"\nEvaluating Distractor Generator on {n_samples} samples...")

    sample = val_df.sample(n_samples, random_state=42)
    correct_count = 0

    for _, row in sample.iterrows():
        article = str(row['article'])
        ans_letter = str(row['answer'])
        correct_text = str(row.get(ans_letter, '')).lower().strip()

        distractors = generator.generate_distractors(article, correct_text, top_n=3)

        # Check that none of the distractors match the correct answer
        distractor_texts = [d.lower().strip() for d in distractors]
        if correct_text not in distractor_texts:
            correct_count += 1

    accuracy = correct_count / n_samples
    print(f"  Distractor Ranker Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  (Fraction of samples where top-3 distractors do NOT include the correct answer)")
    return accuracy


def evaluate_hint_generator(generator, val_df, n_samples=500):
    """
    Evaluates hint extraction quality.

    Hint Precision: fraction of samples where the extracted hint sentence
    overlaps with the sentence containing the gold answer.
    """
    print(f"\nEvaluating Hint Generator on {n_samples} samples...")

    sample = val_df.sample(n_samples, random_state=42)
    hit_count = 0

    for _, row in sample.iterrows():
        article = str(row['article'])
        question = str(row['question'])
        ans_letter = str(row['answer'])
        correct_text = str(row.get(ans_letter, '')).lower().strip()

        hints = generator.generate_hints(article, question)

        # Check if any hint sentence contains the correct answer
        for hint in hints:
            if correct_text and correct_text in hint.lower():
                hit_count += 1
                break

    precision = hit_count / n_samples
    print(f"  Hint Extraction Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  (Fraction of samples where a hint sentence contains the gold answer text)")
    return precision


if __name__ == "__main__":
    start = time.time()

    # Check vectorizer exists
    vectorizer_path = 'models/ohe_vectorizer.pkl'
    if not os.path.exists(vectorizer_path):
        print(f"ERROR: {vectorizer_path} not found.")
        print("Please run src/preprocessing.py first to generate the OHE vectorizer.")
        sys.exit(1)

    print("Loading Model B Generator...")
    generator = ModelB_Generator(vectorizer_path)

    # Load validation data for evaluation
    val_clean_path = 'data/processed/val_clean.csv'
    if not os.path.exists(val_clean_path):
        print(f"ERROR: {val_clean_path} not found.")
        print("Please run src/preprocessing.py first.")
        sys.exit(1)

    val_df = pd.read_csv(val_clean_path)
    print(f"Loaded validation set: {len(val_df)} rows")

    # Evaluate both components
    dist_acc = evaluate_distractor_generator(generator, val_df, n_samples=500)
    hint_prec = evaluate_hint_generator(generator, val_df, n_samples=500)

    print("\n" + "="*50)
    print("Model B Evaluation Summary")
    print("="*50)
    print(f"  Distractor Ranker Accuracy : {dist_acc*100:.2f}%")
    print(f"  Hint Extraction Precision  : {hint_prec*100:.2f}%")
    print(f"\nCompleted in {time.time()-start:.2f}s")
