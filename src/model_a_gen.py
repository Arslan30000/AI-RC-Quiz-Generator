import joblib
import random
import os
import re

class ModelA_QG:
    def __init__(self, tfidf_path='models/tfidf_vectorizer.pkl'):
        """Initializes the Template-Based Question Generator using Traditional ML."""
        try:
            if os.path.exists(tfidf_path):
                self.tfidf = joblib.load(tfidf_path)
                self.feature_names = self.tfidf.get_feature_names_out()
            else:
                self.tfidf = None
                self.feature_names = None
        except Exception as e:
            print(f"Warning: Could not load TF-IDF vectorizer. Error: {e}")
            self.tfidf = None
            self.feature_names = None

        # We will dynamically generate Cloze (Fill-in-the-blank) templates 
        # by extracting the sentence containing the keyword.

    def _fallback_keyword(self, article):
        """Fallback keyword extraction if TF-IDF fails (e.g., words not in vocab)."""
        words = re.findall(r'\b[A-Za-z]{4,}\b', article)
        if not words:
            return "the main topic"
        
        # Pick the most frequent capitalized word or long word
        freq = {}
        for w in words:
            if w[0].isupper() or len(w) > 5:
                freq[w] = freq.get(w, 0) + 1
        
        if freq:
            return max(freq, key=freq.get)
        return random.choice(words)

    def generate_qa(self, article):
        """
        Extracts the most statistically significant keyword using TF-IDF 
        and slots it into a question template.
        """
        if not article or len(article.split()) < 5:
            return "What is the passage about?", "the text"
            
        keyword = None
        
        if self.tfidf and self.feature_names is not None:
            # Transform the single article
            vec = self.tfidf.transform([article]).toarray()[0]
            
            # Find the index with the highest TF-IDF score
            if vec.max() > 0:
                best_idx = vec.argmax()
                keyword = self.feature_names[best_idx]
                
        # If TF-IDF failed (e.g. no words in vocab), use fallback
        if not keyword:
            keyword = self._fallback_keyword(article)
            
        # Find the sentence containing the keyword
        sentences = re.split(r'(?<=[.!?]) +', article)
        question_sentence = None
        for sent in sentences:
            if keyword.lower() in sent.lower():
                question_sentence = sent
                break
                
        # If we couldn't find a clean sentence match, fallback to a template
        if not question_sentence:
            question = f"According to the text, what is the significance of the word '{keyword}'?"
        else:
            # Create a Cloze (Fill-in-the-blank) question
            # Replace the keyword with blanks (case-insensitive)
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            question = pattern.sub("______", question_sentence)
            
            # Make sure it ends with punctuation if it doesn't already
            if not question.endswith(('.', '?', '!')):
                question += "."
        
        return question, keyword

if __name__ == "__main__":
    generator = ModelA_QG('../models/tfidf_vectorizer.pkl')
    article = "The Apollo 11 mission was the spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin formed the American crew."
    q, a = generator.generate_qa(article)
    print(f"Article: {article}")
    print(f"Generated Question: {q}")
    print(f"Generated Answer: {a}")
