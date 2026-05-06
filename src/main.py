import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# STEP 1: IMPORT LIBRARIES AND LOAD DATASET 
# ==========================================
print("Loading datasets...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw')

train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
val_df = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

TARGET = 'answer'

# Handle Missing Values 
train_df.fillna("", inplace=True)
val_df.fillna("", inplace=True)
test_df.fillna("", inplace=True)

# Combine text for the classifier to read
print("Structuring text data...")
def combine_text(df):
    return df['article'] + " " + df['question'] + " " + df['A'] + " " + df['B'] + " " + df['C'] + " " + df['D']

X_train_text = combine_text(train_df)
y_train = train_df[TARGET]

X_val_text = combine_text(val_df)
y_val = val_df[TARGET]

X_test_text = combine_text(test_df)
y_test = test_df[TARGET]

# ==========================================
# MODEL A: ANSWER VERIFICATION
# ==========================================
print("\n--- MODEL A: ANSWER VERIFICATION ---")
print("Converting text to numerical features using TF-IDF (Lab 14)...")

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

X_train_vec = vectorizer.fit_transform(X_train_text)
X_val_vec = vectorizer.transform(X_val_text)
X_test_vec = vectorizer.transform(X_test_text)

print("Training Multinomial Naive Bayes model (Lab 13)...")
# Using Naive Bayes because Lab 13 states it is "effective for text classification"
model_a = MultinomialNB()
model_a.fit(X_train_vec, y_train)

print("Evaluating Model A on Test Data...")
y_test_pred = model_a.predict(X_test_vec)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report (Test Data):")
print(classification_report(y_test, y_test_pred, zero_division=0))


# ==========================================
# MODEL B: EXTRACTIVE HINT GENERATION
# ==========================================
print("\n--- MODEL B: EXTRACTIVE HINT GENERATION ---")
print("Using TF-IDF and Cosine Similarity to generate hints...")

def generate_hint(article, question):
    """
    Splits the article into sentences, converts them to TF-IDF vectors, 
    and finds the sentence most similar to the question using Cosine Similarity.
    """
    # Split article into sentences (simple split by period)
    sentences = [s.strip() for s in article.split('.') if len(s.strip()) > 10]
    
    if not sentences:
        return "No hint available."
    
    # Create a new TF-IDF vectorizer just for this article/question pair
    hint_vectorizer = TfidfVectorizer(stop_words='english')
    
    # Fit the vectorizer on the sentences and the question
    all_text = sentences + [question]
    tfidf_matrix = hint_vectorizer.fit_transform(all_text)
    
    # The question is the last item in the matrix
    question_vec = tfidf_matrix[-1]
    sentence_vecs = tfidf_matrix[:-1]
    
    # Calculate cosine similarity between the question and all sentences
    similarities = cosine_similarity(question_vec, sentence_vecs).flatten()
    
    # Find the index of the sentence with the highest similarity score
    best_sentence_idx = np.argmax(similarities)
    
    return sentences[best_sentence_idx]

# Test Model B on the first question in the test dataset
sample_article = test_df.iloc[0]['article']
sample_question = test_df.iloc[0]['question']
actual_answer = test_df.iloc[0]['answer']

print("\n[Sample Hint Generation Test]")
print("QUESTION:", sample_question)
print("HINT EXTRACTED:", generate_hint(sample_article, sample_question))
print("CORRECT ANSWER WAS:", actual_answer)
print("\nProject execution complete!")