import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse
import joblib
import os
import time

class RacePreprocessor:
    def __init__(self, max_features=5000):
        # Using CountVectorizer with binary=True creates the required One-Hot Encoding
        self.ohe_vectorizer = CountVectorizer(binary=True, max_features=max_features, stop_words='english')
        # Using TF-IDF Vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        
    def _clean_text_node(self, text):
        """Applies lowercasing and punctuation removal."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def clean_corpus(self, df):
        """Iterates through and cleans all relevant text columns in the DataFrame."""
        clean_df = df.copy()
        text_columns = ['article', 'question', 'A', 'B', 'C', 'D']
        for col in text_columns:
            if col in clean_df.columns:
                clean_df[col] = clean_df[col].apply(self._clean_text_node)
            else:
                clean_df[col] = "" # Fill missing option columns if any
        
        # Fill NA answers just in case
        if 'answer' not in clean_df.columns:
            clean_df['answer'] = 'A'
            
        return clean_df

    def compute_lexical_features(self, df):
        """Computes handcrafted lexical features for the DataFrame."""
        features = pd.DataFrame(index=df.index)
        
        features['article_len'] = df['article'].apply(lambda x: len(str(x).split()))
        features['question_len'] = df['question'].apply(lambda x: len(str(x).split()))
        
        # Word overlap between question and article
        def overlap(row):
            q_words = set(str(row['question']).split())
            a_words = set(str(row['article']).split())
            if not q_words:
                return 0
            return len(q_words.intersection(a_words)) / len(q_words)
            
        features['q_article_overlap'] = df.apply(overlap, axis=1)
        
        return features.values

    def build_features(self, df, is_training=False, split_name='train'):
        """Combines text and applies One-Hot Encoding, TF-IDF, and extracts features."""
        print(f"Building features for {split_name}...")
        combined_text = df['article'] + " " + df['question'] + " " + \
                        df['A'] + " " + df['B'] + " " + df['C'] + " " + df['D']
        
        # 1. OHE and TF-IDF
        if is_training:
            os.makedirs('models', exist_ok=True)
            ohe_matrix = self.ohe_vectorizer.fit_transform(combined_text)
            joblib.dump(self.ohe_vectorizer, 'models/ohe_vectorizer.pkl') 
            
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(combined_text)
            joblib.dump(self.tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
        else:
            ohe_matrix = self.ohe_vectorizer.transform(combined_text)
            tfidf_matrix = self.tfidf_vectorizer.transform(combined_text)
            
        # 2. Handcrafted features
        lexical_matrix = self.compute_lexical_features(df)
        
        return ohe_matrix, tfidf_matrix, lexical_matrix

    def process_and_save(self, input_df, output_dir, split_name, is_training=False):
        """Full pipeline execution from raw to processed."""
        clean_df = self.clean_corpus(input_df)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save cleaned text data
        clean_csv_path = os.path.join(output_dir, f'{split_name}_clean.csv')
        clean_df.to_csv(clean_csv_path, index=False)
        
        # Generate matrices
        ohe_matrix, tfidf_matrix, lexical_matrix = self.build_features(clean_df, is_training, split_name)
        
        # Save matrices
        sparse.save_npz(os.path.join(output_dir, f'{split_name}_ohe.npz'), ohe_matrix)
        sparse.save_npz(os.path.join(output_dir, f'{split_name}_tfidf.npz'), tfidf_matrix)
        np.save(os.path.join(output_dir, f'{split_name}_lexical.npy'), lexical_matrix)
        
        if 'answer' in clean_df.columns:
            np.save(os.path.join(output_dir, f'{split_name}_labels.npy'), clean_df['answer'].values)
            
        print(f"[{split_name}] Processed {len(input_df)} rows -> Saved to {output_dir}")
        print(f"[{split_name}] OHE matrix shape: {ohe_matrix.shape}")
        
        return clean_df, ohe_matrix, tfidf_matrix, lexical_matrix

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    
    start_time = time.time()
    processor = RacePreprocessor(max_features=5000) 
    
    print("Loading original train.csv for 80/10/10 split...")
    full_df = pd.read_csv('data/raw/train.csv')
    
    # 80/10/10 Split
    train_df, temp_df = train_test_split(full_df, test_size=0.20, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
    
    print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Process Train
    processor.process_and_save(train_df, 'data/processed', 'train', is_training=True)
    
    # Process Val
    processor.process_and_save(val_df, 'data/processed', 'val', is_training=False)
    
    # Process Test
    processor.process_and_save(test_df, 'data/processed', 'test', is_training=False)
    
    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds.")