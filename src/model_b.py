import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

STOPWORDS = set([
    'the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of', 'and', 'or', 
    'is', 'are', 'was', 'were', 'what', 'which', 'who', 'whom', 'whose', 
    'when', 'where', 'why', 'how', 'did', 'does', 'do', 'has', 'have', 'had', 
    'her', 'his', 'their', 'our', 'my', 'your', 'it', 'its', 'this', 'that', 
    'these', 'those', 'from', 'with', 'by', 'as', 'be', 'been'
])

class ModelB_Generator:
    def __init__(self, vectorizer_path='models/ohe_vectorizer.pkl'):
        try:
            if os.path.exists(vectorizer_path):
                self.vectorizer = joblib.load(vectorizer_path)
            else:
                self.vectorizer = None
        except Exception as e:
            print(f"Warning: Could not load vectorizer. Error: {e}")
            self.vectorizer = None

    def clean_text(self, text):
        if not isinstance(text, str): return ""
        return re.sub(r'[^\w\s]', '', text.lower())

    def get_stemmed_keywords(self, text):
        if not isinstance(text, str): return set()
        words = re.sub(r'[^\w\s]', '', text.lower()).split()
        # Remove stopwords and naively stem by taking first 5 chars
        return set([w[:6] for w in words if w not in STOPWORDS and len(w) > 2])

    def generate_distractors(self, article, correct_answer, top_n=3):
        """
        Generates plausible distractors by extracting high-frequency noun-like words 
        (capitalized or >5 letters) directly from the article.
        """
        clean_ans = self.clean_text(correct_answer)
        ans_words = set(clean_ans.split())
        
        # Extract candidate words from the original article to preserve capitalization
        words = re.findall(r'\b[A-Za-z]{3,}\b', article)
        
        candidates = {}
        for w in words:
            # We want plausible noun-like concepts: either Capitalized or long (len > 5)
            if w[0].isupper() or len(w) > 5:
                w_clean = w.lower()
                if w_clean not in ans_words and w_clean not in STOPWORDS:
                    candidates[w_clean] = candidates.get(w_clean, 0) + 1
                    
        # Fallback if no capitalized/long words found
        if len(candidates) < top_n:
            for w in words:
                w_clean = w.lower()
                if w_clean not in ans_words and w_clean not in STOPWORDS and len(w_clean) > 4:
                    candidates[w_clean] = candidates.get(w_clean, 0) + 1

        if not candidates:
            return ["Option A", "Option B", "Option C"]
            
        # Sort by frequency descending (highest frequency in article = most plausible)
        sorted_cands = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        
        distractors = [x[0] for x in sorted_cands[:top_n]]
        
        while len(distractors) < top_n:
            distractors.append(f"Distractor_{len(distractors)}")
            
        return distractors

    def generate_hints(self, article, question, top_k=3):
        """
        Rule-Based Graduated Hint Extraction.
        Returns a strict 3-tier hierarchy: General -> Specific -> Most Specific
        """
        if not isinstance(article, str) or not isinstance(question, str):
            return ["General Hint: No article provided.", "Specific Hint: No question provided.", "Most Specific Hint: Read carefully."]
            
        sentences = re.split(r'(?<=[.!?]) +', article)
        clean_q_words = self.get_stemmed_keywords(question)
        
        # 1. HINT 1: General (Topic Keyword)
        if clean_q_words:
            keyword = list(clean_q_words)[0]
            hint1 = f"General Hint: Look for information regarding '{keyword}' or related concepts."
        else:
            hint1 = "General Hint: Try to identify the main subject of the question in the first paragraph."

        # Find the sentence with the highest overlap
        scored_sentences = []
        for sent in sentences:
            clean_s_words = self.get_stemmed_keywords(sent)
            if not clean_s_words: continue
            overlap = len(clean_q_words.intersection(clean_s_words))
            scored_sentences.append((overlap, sent.strip()))
            
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        if not scored_sentences:
            return [hint1, "Specific Hint: Consider the main characters.", "Most Specific Hint: Re-read the conclusion."]
            
        best_sentence = scored_sentences[0][1]
        
        # 2. HINT 2: Specific (Context Sentence)
        hint2 = f"Specific Hint: Focus closely on this sentence: '{best_sentence}'"
        
        # 3. HINT 3: Most Specific (Zooming in)
        words = best_sentence.split()
        if len(words) > 6:
            # Take a small chunk of the sentence
            chunk = " ".join(words[:6])
            hint3 = f"Most Specific Hint: The answer is very closely related to the phrase: '{chunk}...'"
        else:
            hint3 = f"Most Specific Hint: The answer is hidden right here: '{best_sentence}'"
            
        return [hint1, hint2, hint3]

if __name__ == "__main__":
    generator = ModelB_Generator('../models/ohe_vectorizer.pkl')
    article = "The quick brown fox jumps over the lazy dog. The dog woke up and barked loudly at the fox. It was a sunny summer day."
    q = "What did the dog do when it woke up?"
    ans = "barked loudly"
    
    print("Distractors:", generator.generate_distractors(article, ans))
    print("Hints:", generator.generate_hints(article, q))
