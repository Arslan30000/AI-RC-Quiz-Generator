import pandas as pd
import numpy as np
from scipy import sparse

print("=== Data Preprocessing Verification ===\n")

# 1. Text Cleaning Check
try:
    raw_df = pd.read_csv('data/raw/train.csv', nrows=2)
    clean_df = pd.read_csv('data/processed/train_clean.csv', nrows=2)
    
    print("1. Text Cleaning (First Row):")
    print(f"RAW ARTICLE:   {raw_df.loc[0, 'article'][:100]}...")
    print(f"CLEAN ARTICLE: {clean_df.loc[0, 'article'][:100]}...\n")
    
    print(f"RAW QUESTION:  {raw_df.loc[0, 'question']}")
    print(f"CLEAN QUESTION: {clean_df.loc[0, 'question']}\n")
except Exception as e:
    print(f"Error loading CSVs: {e}")

# 2. Lexical Features Check
try:
    lexical = np.load('data/processed/train_lexical.npy')
    print("2. Lexical Features (First Row):")
    # Features are: [article_len, question_len, overlap_ratio]
    print(f"Article Word Count:  {lexical[0][0]}")
    print(f"Question Word Count: {lexical[0][1]}")
    print(f"Word Overlap Ratio:  {lexical[0][2]:.4f}\n")
except Exception as e:
    print(f"Error loading Lexical Features: {e}")

# 3. OHE & TF-IDF Check
try:
    ohe = sparse.load_npz('data/processed/train_ohe.npz')
    tfidf = sparse.load_npz('data/processed/train_tfidf.npz')
    
    print("3. Matrices Check:")
    print(f"OHE Matrix Shape:    {ohe.shape} (Rows, Max Features)")
    print(f"TF-IDF Matrix Shape: {tfidf.shape}")
    print(f"Non-zero elements in OHE row 0: {ohe[0].nnz}")
except Exception as e:
    print(f"Error loading sparse matrices: {e}")

print("\n=== Verification Complete ===")
