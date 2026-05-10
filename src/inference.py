"""
src/inference.py

Unified Inference API for the Intelligent RC Quiz Generation System.

This module provides a single, clean interface to run both Model A and Model B
inference given a reading passage. It is the backend used by the Streamlit UI
and can also be called from the command line for quick testing.

Usage from project root:
    venv\\Scripts\\python.exe src\\inference.py

Or import in Python:
    from src.inference import RCInferencePipeline
    pipeline = RCInferencePipeline()
    result = pipeline.run(article="Your passage here...")
"""

import joblib
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model_a_gen import ModelA_QG
from src.model_b import ModelB_Generator


class RCInferencePipeline:
    """
    Unified inference pipeline that wires Model A (Q&A generation) and
    Model B (distractor + hint generation) together.
    """

    def __init__(
        self,
        tfidf_path='models/tfidf_vectorizer.pkl',
        ohe_path='models/ohe_vectorizer.pkl',
    ):
        print("Loading Model A (TF-IDF Q&A Generator)...")
        self.model_a_qg = ModelA_QG(tfidf_path)

        print("Loading Model B (Distractor & Hint Generator)...")
        self.model_b = ModelB_Generator(ohe_path)

        print("Pipeline ready.\n")

    def run(self, article: str, question: str = None, correct_answer: str = None) -> dict:
        """
        Runs the full inference pipeline on a given article.

        If question and correct_answer are not provided, Model A is used
        to auto-generate them via TF-IDF keyword extraction.

        Args:
            article: The reading passage as a string.
            question: (optional) A manually provided question.
            correct_answer: (optional) The manually provided correct answer.

        Returns:
            A dictionary containing the generated question, answer, distractors,
            full MCQ options, hints, and latency measurements.
        """
        t_start = time.time()
        result = {}

        # --- Model A: Q&A Generation ---
        t0 = time.time()
        if not question or not correct_answer:
            question, correct_answer = self.model_a_qg.generate_qa(article)
            result['qa_source'] = 'Model A (Auto-Generated)'
        else:
            result['qa_source'] = 'Manual Input'

        result['question'] = question
        result['correct_answer'] = correct_answer
        result['model_a_latency_s'] = round(time.time() - t0, 4)

        # --- Model B: Distractor + Hint Generation ---
        t0 = time.time()
        distractors = self.model_b.generate_distractors(article, correct_answer, top_n=3)
        hints = self.model_b.generate_hints(article, question)
        result['model_b_latency_s'] = round(time.time() - t0, 4)

        result['distractors'] = distractors
        result['hints'] = hints

        # Assemble final MCQ options (shuffled)
        import random
        all_options = [correct_answer] + distractors[:3]
        random.shuffle(all_options)
        labels = ['A', 'B', 'C', 'D']
        result['options'] = {labels[i]: all_options[i] for i in range(4)}

        result['total_latency_s'] = round(time.time() - t_start, 4)
        return result


def _print_result(result: dict):
    """Pretty-print inference results to terminal."""
    print("\n" + "="*60)
    print("INFERENCE RESULT")
    print("="*60)
    print(f"Source          : {result['qa_source']}")
    print(f"Question        : {result['question']}")
    print(f"Correct Answer  : {result['correct_answer']}")
    print("\nMCQ Options:")
    for k, v in result['options'].items():
        marker = " ✓" if v.lower() == result['correct_answer'].lower() else ""
        print(f"  {k}) {v}{marker}")
    print("\nGraduated Hints:")
    for i, hint in enumerate(result['hints'], 1):
        print(f"  Hint {i}: {hint}")
    print(f"\nLatency — Model A: {result['model_a_latency_s']}s | "
          f"Model B: {result['model_b_latency_s']}s | "
          f"Total: {result['total_latency_s']}s")
    print("="*60)


if __name__ == "__main__":
    pipeline = RCInferencePipeline()

    # Example passage for quick testing
    test_article = (
        "The Apollo 11 mission was the spaceflight that first landed humans on the Moon. "
        "Commander Neil Armstrong and lunar module pilot Buzz Aldrin formed the American crew "
        "that landed the Apollo Lunar Module Eagle on July 20, 1969. Armstrong became the first "
        "person to step onto the lunar surface six hours and 39 minutes later on July 21. "
        "Aldrin joined him 19 minutes later. They spent about two and a quarter hours together "
        "outside the spacecraft, and collected 47.5 pounds of lunar material to bring back to Earth."
    )

    result = pipeline.run(article=test_article)
    _print_result(result)
