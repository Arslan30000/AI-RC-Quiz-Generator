import numpy as np
from scipy import sparse
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
import time

print("Loading data...")
X_train = sparse.load_npz('data/processed/train_ohe.npz')
y_train = np.load('data/processed/train_labels.npy', allow_pickle=True)
X_val = sparse.load_npz('data/processed/val_ohe.npz')
y_val = np.load('data/processed/val_labels.npy', allow_pickle=True)

print("Training LinearSVC...")
t0 = time.time()
svm = LinearSVC(class_weight='balanced', max_iter=1000, random_state=42, dual=False)
svm.fit(X_train, y_train)
print(f"Trained in {time.time()-t0:.2f}s")

preds = svm.predict(X_val)
acc = accuracy_score(y_val, preds)
f1 = f1_score(y_val, preds, average='macro')

print(f"LinearSVC Acc: {acc:.4f}, Macro F1: {f1:.4f}")
