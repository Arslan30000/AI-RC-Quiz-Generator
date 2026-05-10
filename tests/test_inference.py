"""
tests/test_inference.py

Unit tests for the Unified Inference Pipeline.

Tests cover:
  - Model A Q&A generation (keyword extraction + Cloze question)
  - Model B distractor generation (correct answer not in distractors)
  - Model B hint generation (3 tiers, correct types)
  - Full pipeline integration test

Run from project root:
    venv\\Scripts\\python.exe tests\\test_inference.py
"""

import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_a_gen import ModelA_QG
from src.model_b import ModelB_Generator

# Sample passages for testing
SAMPLE_ARTICLE = (
    "The Apollo 11 mission was the spaceflight that first landed humans on the Moon. "
    "Commander Neil Armstrong and lunar module pilot Buzz Aldrin formed the American crew. "
    "Armstrong became the first person to step onto the lunar surface on July 20, 1969. "
    "They spent two hours outside the spacecraft and collected lunar material."
)

SAMPLE_QUESTION = "Who became the first person to step onto the lunar surface?"
SAMPLE_ANSWER = "Neil Armstrong"


class TestModelAQG(unittest.TestCase):
    """Unit tests for Model A Template-Based Question Generator."""

    @classmethod
    def setUpClass(cls):
        cls.model = ModelA_QG('models/tfidf_vectorizer.pkl')

    def test_generate_qa_returns_tuple(self):
        """generate_qa should return a (question, answer) tuple."""
        result = self.model.generate_qa(SAMPLE_ARTICLE)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_question_contains_blank(self):
        """Generated question should contain a blank (______) for Cloze format."""
        question, answer = self.model.generate_qa(SAMPLE_ARTICLE)
        self.assertIn("______", question,
                      "Generated question should contain '______' placeholder.")

    def test_answer_is_nonempty_string(self):
        """Generated answer (keyword) should be a non-empty string."""
        question, answer = self.model.generate_qa(SAMPLE_ARTICLE)
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer.strip()), 0)

    def test_short_article_fallback(self):
        """Very short input should not crash the generator."""
        question, answer = self.model.generate_qa("Short text here")
        self.assertIsInstance(question, str)
        self.assertIsInstance(answer, str)

    def test_empty_article_fallback(self):
        """Empty string input should not raise an exception."""
        try:
            question, answer = self.model.generate_qa("")
        except Exception as e:
            self.fail(f"generate_qa raised an exception on empty input: {e}")


class TestModelBDistracters(unittest.TestCase):
    """Unit tests for Model B Distractor Generator."""

    @classmethod
    def setUpClass(cls):
        cls.model = ModelB_Generator('models/ohe_vectorizer.pkl')

    def test_generate_distractors_returns_list(self):
        """generate_distractors should return a list."""
        distractors = self.model.generate_distractors(SAMPLE_ARTICLE, SAMPLE_ANSWER)
        self.assertIsInstance(distractors, list)

    def test_generate_distractors_count(self):
        """Should return exactly 3 distractors."""
        distractors = self.model.generate_distractors(SAMPLE_ARTICLE, SAMPLE_ANSWER, top_n=3)
        self.assertEqual(len(distractors), 3)

    def test_correct_answer_not_in_distractors(self):
        """The correct answer should NOT appear in the distractors."""
        distractors = self.model.generate_distractors(SAMPLE_ARTICLE, SAMPLE_ANSWER)
        ans_lower = SAMPLE_ANSWER.lower()
        for d in distractors:
            self.assertNotEqual(d.lower(), ans_lower,
                                f"Distractor '{d}' matches the correct answer '{SAMPLE_ANSWER}'.")

    def test_distractors_are_strings(self):
        """All distractors should be non-empty strings."""
        distractors = self.model.generate_distractors(SAMPLE_ARTICLE, SAMPLE_ANSWER)
        for d in distractors:
            self.assertIsInstance(d, str)
            self.assertGreater(len(d.strip()), 0)


class TestModelBHints(unittest.TestCase):
    """Unit tests for Model B Hint Generator."""

    @classmethod
    def setUpClass(cls):
        cls.model = ModelB_Generator('models/ohe_vectorizer.pkl')

    def test_generate_hints_returns_list(self):
        """generate_hints should return a list."""
        hints = self.model.generate_hints(SAMPLE_ARTICLE, SAMPLE_QUESTION)
        self.assertIsInstance(hints, list)

    def test_generate_hints_count(self):
        """Should return exactly 3 hints."""
        hints = self.model.generate_hints(SAMPLE_ARTICLE, SAMPLE_QUESTION)
        self.assertEqual(len(hints), 3)

    def test_hint1_is_general(self):
        """Hint 1 should contain 'General Hint'."""
        hints = self.model.generate_hints(SAMPLE_ARTICLE, SAMPLE_QUESTION)
        self.assertIn("General Hint", hints[0])

    def test_hint2_is_specific(self):
        """Hint 2 should contain 'Specific Hint'."""
        hints = self.model.generate_hints(SAMPLE_ARTICLE, SAMPLE_QUESTION)
        self.assertIn("Specific Hint", hints[1])

    def test_hint3_is_most_specific(self):
        """Hint 3 should contain 'Most Specific Hint'."""
        hints = self.model.generate_hints(SAMPLE_ARTICLE, SAMPLE_QUESTION)
        self.assertIn("Most Specific Hint", hints[2])

    def test_hints_with_empty_inputs(self):
        """Hints should not crash on empty string inputs."""
        try:
            hints = self.model.generate_hints("", "")
        except Exception as e:
            self.fail(f"generate_hints raised an exception on empty inputs: {e}")


if __name__ == "__main__":
    print("Running Unit Tests for RC Quiz Generator Inference Pipeline...\n")
    unittest.main(verbosity=2)
