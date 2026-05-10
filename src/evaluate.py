"""
src/evaluate.py

Metric computation script for Model A.

Loads all trained Model A classifiers from disk, runs them on the test split
(the held-out 10% never seen during training), and prints a full comparison
table of Accuracy, Macro F1, Precision, Recall, and Exact Match.

Run from project root:
    venv\\Scripts\\python.exe src\\evaluate.py
"""

import numpy as np
import joblib
import os
import sys
from scipy import sparse
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.metrics.cluster import contingency_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def purity_score(y_true, y_pred):
    """Clustering purity: fraction of samples in the majority cluster class."""
    cm = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)


def load_test_data(data_dir='data/processed'):
    """Load the held-out test split matrices and labels."""
    print(f"Loading test data from {data_dir}...")
    X_test = sparse.load_npz(os.path.join(data_dir, 'test_ohe.npz'))
    y_test = np.load(os.path.join(data_dir, 'test_labels.npy'), allow_pickle=True)
    print(f"Test set shape: {X_test.shape}, Labels: {y_test.shape}\n")
    return X_test, y_test


def evaluate_model(model, X_test, y_test, model_name):
    """Run predictions and print all required metrics for a supervised model."""
    print(f"{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    macro_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    macro_rec = recall_score(y_test, y_pred, average='macro', zero_division=0)

    print(f"  Accuracy       : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Exact Match(EM): {acc:.4f}  (identical to Accuracy for classification)")
    print(f"  Macro F1       : {macro_f1:.4f}  ({macro_f1*100:.2f}%)")
    print(f"  Macro Precision: {macro_prec:.4f}  ({macro_prec*100:.2f}%)")
    print(f"  Macro Recall   : {macro_rec:.4f}  ({macro_rec*100:.2f}%)")

    print(f"\n  Classification Report (per class):")
    report = classification_report(y_test, y_pred, zero_division=0)
    for line in report.splitlines():
        print(f"    {line}")

    print(f"  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=['A', 'B', 'C', 'D'])
    print(f"        Pred_A  Pred_B  Pred_C  Pred_D")
    for i, label in enumerate(['A', 'B', 'C', 'D']):
        print(f"  True_{label}  {'  '.join(f'{v:5d}' for v in cm[i])}")

    return {
        'Model': model_name,
        'Accuracy': f"{acc*100:.2f}%",
        'Macro F1': f"{macro_f1*100:.2f}%",
        'Macro Precision': f"{macro_prec*100:.2f}%",
        'Macro Recall': f"{macro_rec*100:.2f}%",
        'Exact Match': f"{acc*100:.2f}%",
    }


def evaluate_kmeans(model, X_test, y_test):
    """Evaluate K-Means clustering with purity and silhouette score."""
    from sklearn.metrics import silhouette_score

    print(f"{'='*55}")
    print(f"  K-Means Clustering (Unsupervised)")
    print(f"{'='*55}")
    y_pred = model.predict(X_test)

    purity = purity_score(y_test, y_pred)
    sil = silhouette_score(X_test, y_pred, sample_size=3000, random_state=42)

    print(f"  Clustering Purity    : {purity:.4f}  ({purity*100:.2f}%)")
    print(f"  Silhouette Score     : {sil:.4f}")
    print(f"  (Negative Silhouette indicates overlapping clusters in OHE space — expected)")

    return {
        'Model': 'K-Means Clustering',
        'Clustering Purity': f"{purity*100:.2f}%",
        'Silhouette Score': f"{sil:.4f}",
    }


if __name__ == "__main__":
    model_dir = 'models/model_a/traditional'

    # Check files exist
    required = [
        'data/processed/test_ohe.npz',
        'data/processed/test_labels.npy',
        f'{model_dir}/logistic_regression.pkl',
        f'{model_dir}/svm_classifier.pkl',
        f'{model_dir}/naive_bayes.pkl',
        f'{model_dir}/kmeans_clustering.pkl',
        f'{model_dir}/ensemble_voting.pkl',
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("ERROR: Missing required files:")
        for f in missing:
            print(f"  - {f}")
        print("\nRun src/preprocessing.py then src/model_a_train.py first.")
        sys.exit(1)

    X_test, y_test = load_test_data()

    print("Loading trained models...\n")
    lr = joblib.load(f'{model_dir}/logistic_regression.pkl')
    svm = joblib.load(f'{model_dir}/svm_classifier.pkl')
    nb = joblib.load(f'{model_dir}/naive_bayes.pkl')
    kmeans = joblib.load(f'{model_dir}/kmeans_clustering.pkl')
    ensemble = joblib.load(f'{model_dir}/ensemble_voting.pkl')

    results = []
    results.append(evaluate_model(lr, X_test, y_test, "Logistic Regression"))
    results.append(evaluate_model(svm, X_test, y_test, "Support Vector Machine (LinearSVC)"))
    results.append(evaluate_model(nb, X_test, y_test, "Naive Bayes (ComplementNB)"))
    results.append(evaluate_model(ensemble, X_test, y_test, "Hard Voting Ensemble (LR+SVM+NB)"))
    kmeans_result = evaluate_kmeans(kmeans, X_test, y_test)

    print(f"\n{'='*55}")
    print("  SUMMARY TABLE (Test Set — 8,787 samples)")
    print(f"{'='*55}")
    print(f"{'Model':<35} {'Accuracy':>10} {'Macro F1':>10} {'Macro P':>10} {'Macro R':>10}")
    print("-"*75)
    for r in results:
        print(f"{r['Model']:<35} {r['Accuracy']:>10} {r['Macro F1']:>10} {r['Macro Precision']:>10} {r['Macro Recall']:>10}")
    print(f"\nK-Means Purity: {kmeans_result['Clustering Purity']}  |  Silhouette: {kmeans_result['Silhouette Score']}")
    print(f"\nRandom Baseline (guessing 1 of 4): 25.00%")
