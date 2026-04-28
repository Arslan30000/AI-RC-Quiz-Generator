import pandas as pd
import os

def load_and_clean_data():
    print("Loading raw RACE data...")
    train_df = pd.read_csv('data/raw/train.csv')
    val_df = pd.read_csv('data/raw/val.csv')
    test_df = pd.read_csv('data/raw/test.csv')

    print("Cleaning text data...")
    train_df['article'] = train_df['article'].fillna('').str.lower()
    val_df['article'] = val_df['article'].fillna('').str.lower()
    test_df['article'] = test_df['article'].fillna('').str.lower()


    print("Saving processed data to data/processed/ ...")
    train_df.to_csv('data/processed/train_clean.csv', index=False)
    val_df.to_csv('data/processed/val_clean.csv', index=False)
    test_df.to_csv('data/processed/test_clean.csv', index=False)
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    load_and_clean_data()
    