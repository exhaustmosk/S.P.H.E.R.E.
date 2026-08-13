"""
train_xgboost_smote.py
────────────────────────────────────────────────────────────────────────────
Objective 1 Model Training with SMOTE
  Head A: XGBClassifier for Mental_Health_Status (4 classes)
  Class imbalance is handled by upsampling minority classes using SMOTE.
────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

# ── colour helpers ────────────────────────────────────────────────────────────
RESET = "\033[0m"; BOLD = "\033[1m"; GREEN = "\033[92m"
YELLOW = "\033[93m"; RED = "\033[91m"; CYAN = "\033[96m"; MAGENTA = "\033[95m"
def ok(m):   return f"{GREEN}✔  {m}{RESET}"
def hdr(m):  return f"\n{BOLD}{CYAN}{'─'*62}\n  {m}\n{'─'*62}{RESET}"
def sub(m):  return f"{MAGENTA}  ▸ {m}{RESET}"

# ── 1. LOAD DATA & RE-SPLIT ──────────────────────────────────────────────────
print(hdr("1 · Loading and Re-Splitting Dataset"))

DATA_DIR = Path("dataset_splits")

# Reconstruct the full dataset from the previous 80/10/10 split
X_train_old = np.load(DATA_DIR / "X_train.npy")
X_val_old   = np.load(DATA_DIR / "X_val.npy")
X_test_old  = np.load(DATA_DIR / "X_test.npy")

y_train_old = np.load(DATA_DIR / "y_train_class.npy")
y_val_old   = np.load(DATA_DIR / "y_val_class.npy")
y_test_old  = np.load(DATA_DIR / "y_test_class.npy")

X_full = np.vstack([X_train_old, X_val_old, X_test_old])
y_full = np.concatenate([y_train_old, y_val_old, y_test_old])

print(sub(f"Reconstructed Full Dataset - X: {X_full.shape} | y: {y_full.shape}"))

# 80/20 Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.20, random_state=42, stratify=y_full
)

print(ok(f"New Train Split: {X_train.shape} | New Test Split: {X_test.shape}"))

# ── 2. APPLY SMOTE ───────────────────────────────────────────────────────────
print(hdr("2 · Applying SMOTE on Training Data"))

# Display class distribution before SMOTE
classes, counts = np.unique(y_train, return_counts=True)
print(sub(f"Before SMOTE counts: {dict(zip(classes, counts))}"))

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Display class distribution after SMOTE
classes_res, counts_res = np.unique(y_train_res, return_counts=True)
print(ok(f"After SMOTE counts:  {dict(zip(classes_res, counts_res))}"))
print(ok(f"Resampled X_train shape: {X_train_res.shape}"))

# ── 3. TRAIN CLASSIFIER ──────────────────────────────────────────────────────
print(hdr("3 · Training XGBClassifier on Resampled Data"))

clf = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='multi:softprob',
    num_class=4,
    random_state=42,
    n_jobs=-1
)

print(sub("Fitting XGBClassifier... (this may take a moment)"))
clf.fit(X_train_res, y_train_res)
print(ok("Model trained!"))

# ── 4. EVALUATE ──────────────────────────────────────────────────────────────
print(hdr("4 · Evaluating on Test Set"))

y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)

class_names = ["Healthy", "Mild_Stress", "Moderate_Stress", "Severe_Stress"]

acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

print(f"\n{BOLD}Classification Metrics:{RESET}")
print(f"Accuracy : {acc:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}\n")
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (SMOTE)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix_smote.png')

joblib.dump(clf, "xgb_classifier_head_A_smote.joblib")
print(ok("Saved new confusion_matrix_smote.png and xgb_classifier_head_A_smote.joblib"))

print(f"\n{BOLD}{GREEN}{'═'*62}")
print("  Training Complete.")
print(f"{'═'*62}{RESET}\n")
