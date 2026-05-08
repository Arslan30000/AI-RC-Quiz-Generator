import pandas as pd
import joblib
import os
import time
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.metrics.cluster import contingency_matrix

class ModelA_Pipeline:
    def __init__(self, random_seed=42):
        """
        Initializes the Model A suite with custom configurations for 
        Traditional, Unsupervised, and Ensemble requirements.
        """
        self.seed = random_seed
        
        # Supervised Baselines
        self.log_reg = LogisticRegression(max_iter=1000, random_state=self.seed)
        
        # LinearSVC with class_weight='balanced' to fix the low Macro F1 issue
        self.svm_model = LinearSVC(class_weight='balanced', random_state=self.seed, dual=False)
        
        # Naive Bayes as suggested
        self.nb_model = ComplementNB()
        
        # Unsupervised/Clustering
        self.kmeans = KMeans(n_clusters=4, random_state=self.seed, n_init='auto') 
        
        # Ensemble Strategy: Hard Voting with all three models
        self.ensemble = VotingClassifier(
            estimators=[('lr', self.log_reg), ('svm', self.svm_model), ('nb', self.nb_model)],
            voting='hard'
        )

    def load_processed_data(self, data_dir='data/processed'):
        """Loads the sparse matrices and labels directly from disk."""
        print(f"Loading data from {data_dir}...")
        self.X_train = sparse.load_npz(os.path.join(data_dir, 'train_ohe.npz'))
        self.y_train = np.load(os.path.join(data_dir, 'train_labels.npy'), allow_pickle=True)
        
        self.X_val = sparse.load_npz(os.path.join(data_dir, 'val_ohe.npz'))
        self.y_val = np.load(os.path.join(data_dir, 'val_labels.npy'), allow_pickle=True)
        print(f"Loaded Train: {self.X_train.shape}, Val: {self.X_val.shape}")

    def purity_score(self, y_true, y_pred):
        """Calculate clustering purity score."""
        cm = contingency_matrix(y_true, y_pred)
        return np.sum(np.amax(cm, axis=0)) / np.sum(cm)

    def _evaluate_and_print(self, model, model_name, is_unsupervised=False):
        """Calculates and prints required metrics."""
        
        if is_unsupervised:
            print(f"--- {model_name} (Unsupervised) ---")
            predictions = model.predict(self.X_val)
            
            purity = self.purity_score(self.y_val, predictions)
            print(f"Clustering Purity: {purity:.4f}")
            
            # Silhouette Score on a random subset of 5000 to prevent OOM errors
            sil_score = silhouette_score(self.X_val, predictions, sample_size=5000, random_state=self.seed)
            print(f"Silhouette Score (Sampled): {sil_score:.4f}\n")
            return

        predictions = model.predict(self.X_val)
        acc = accuracy_score(self.y_val, predictions)
        f1 = f1_score(self.y_val, predictions, average='macro')
        
        print(f"--- {model_name} Performance ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"Exact Match (EM): {acc:.4f} (Same as accuracy for discrete classification)")
        print(f"Macro F1: {f1:.4f}\n")

    def execute_training_suite(self):
        """Executes training and evaluation for all required Model A components."""
        print("\nStarting Model A Training Suite...\n")

        # 1. Traditional ML: Logistic Regression
        t0 = time.time()
        print("Training Logistic Regression...")
        self.log_reg.fit(self.X_train, self.y_train)
        print(f"Finished in {time.time()-t0:.1f}s")
        self._evaluate_and_print(self.log_reg, "Logistic Regression")

        # 2. Traditional ML: SVM
        t0 = time.time()
        print("Training Support Vector Machine (LinearSVC)...")
        self.svm_model.fit(self.X_train, self.y_train)
        print(f"Finished in {time.time()-t0:.1f}s")
        self._evaluate_and_print(self.svm_model, "Support Vector Machine")
        
        # 3. Traditional ML: Naive Bayes
        t0 = time.time()
        print("Training Naive Bayes (ComplementNB)...")
        self.nb_model.fit(self.X_train, self.y_train)
        print(f"Finished in {time.time()-t0:.1f}s")
        self._evaluate_and_print(self.nb_model, "Naive Bayes")

        # 4. Unsupervised ML: K-Means
        t0 = time.time()
        print("Running K-Means Clustering...")
        self.kmeans.fit(self.X_train)
        print(f"Finished in {time.time()-t0:.1f}s")
        self._evaluate_and_print(self.kmeans, "K-Means Clustering", is_unsupervised=True)

        # 5. Ensemble ML: Hard Voting
        t0 = time.time()
        print("Training Hard Voting Ensemble (LR + SVM + NB)...")
        self.ensemble.fit(self.X_train, self.y_train)
        print(f"Finished in {time.time()-t0:.1f}s")
        self._evaluate_and_print(self.ensemble, "Hard Voting Ensemble (LR + SVM + NB)")

    def persist_models(self, export_dir='models/model_a/traditional/'):
        """Saves the trained models to disk using joblib."""
        os.makedirs(export_dir, exist_ok=True)
        joblib.dump(self.log_reg, os.path.join(export_dir, 'logistic_regression.pkl'))
        joblib.dump(self.svm_model, os.path.join(export_dir, 'svm_classifier.pkl'))
        joblib.dump(self.nb_model, os.path.join(export_dir, 'naive_bayes.pkl'))
        joblib.dump(self.kmeans, os.path.join(export_dir, 'kmeans_clustering.pkl'))
        joblib.dump(self.ensemble, os.path.join(export_dir, 'ensemble_voting.pkl'))
        print(f"All models successfully exported to {export_dir}")

if __name__ == "__main__":
    pipeline = ModelA_Pipeline()
    pipeline.load_processed_data(data_dir='data/processed')
    pipeline.execute_training_suite()
    pipeline.persist_models()