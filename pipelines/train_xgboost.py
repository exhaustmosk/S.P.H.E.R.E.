"""
train_xgboost.py
────────────────────────────────────────────────────────────────────────────
Objective 1 & 2 Model Training
  Head A: XGBClassifier for Mental_Health_Status (4 classes)
  Head B: MultiOutputRegressor(XGBRegressor) for Depression, Anxiety, Stress
────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier, XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    classification_report, accuracy_score, roc_auc_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
)
from sklearn.utils.class_weight import compute_sample_weight

import warnings
warnings.filterwarnings('ignore')

# ── colour helpers ────────────────────────────────────────────────────────────
RESET = "\033[0m"; BOLD = "\033[1m"; GREEN = "\033[92m"
YELLOW = "\033[93m"; RED = "\033[91m"; CYAN = "\033[96m"; MAGENTA = "\033[95m"
def ok(m):   return f"{GREEN}✔  {m}{RESET}"
def hdr(m):  return f"\n{BOLD}{CYAN}{'─'*62}\n  {m}\n{'─'*62}{RESET}"
def sub(m):  return f"{MAGENTA}  ▸ {m}{RESET}"

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────
print(hdr("1 · Loading Split Datasets"))

DATA_DIR = Path("dataset_splits")

X_train = np.load(DATA_DIR / "X_train.npy")
X_test  = np.load(DATA_DIR / "X_test.npy")

y_train_class = np.load(DATA_DIR / "y_train_class.npy")
y_test_class  = np.load(DATA_DIR / "y_test_class.npy")

y_train_reg = np.load(DATA_DIR / "y_train_reg.npy")
y_test_reg  = np.load(DATA_DIR / "y_test_reg.npy")

print(ok(f"X_train: {X_train.shape} | y_train_class: {y_train_class.shape} | y_train_reg: {y_train_reg.shape}"))
print(ok(f"X_test:  {X_test.shape}  | y_test_class:  {y_test_class.shape}  | y_test_reg:  {y_test_reg.shape}"))

# ── 2. HEAD A: CLASSIFICATION ────────────────────────────────────────────────
print(hdr("2 · Head A: Training Classification Model (Objective 1)"))

# Compute sample weights to handle class imbalance (e.g., Severe_Stress)
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_class)

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
clf.fit(X_train, y_train_class, sample_weight=sample_weights)
print(ok("Model trained!"))

print(sub("Evaluating on Test Set"))
y_pred_class = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)

# Metrics
acc = accuracy_score(y_test_class, y_pred_class)
roc_auc = roc_auc_score(y_test_class, y_pred_proba, multi_class='ovr')
class_names = ["Healthy", "Mild_Stress", "Moderate_Stress", "Severe_Stress"]

print(f"\n{BOLD}Classification Metrics:{RESET}")
print(f"Accuracy : {acc:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}\n")
print(classification_report(y_test_class, y_pred_class, target_names=class_names))

# Confusion Matrix Plot
cm = confusion_matrix(y_test_class, y_pred_class)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix: Mental Health Status')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print(ok("Saved confusion_matrix.png"))

joblib.dump(clf, "xgb_classifier_head_A.joblib")
print(ok("Saved model artifact: xgb_classifier_head_A.joblib"))

# ── 3. HEAD B: REGRESSION ────────────────────────────────────────────────────
print(hdr("3 · Head B: Training Regression Model (Objective 2)"))

base_reg = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)
multi_reg = MultiOutputRegressor(base_reg)

print(sub("Fitting MultiOutputRegressor(XGBRegressor)..."))
multi_reg.fit(X_train, y_train_reg)
print(ok("Model trained!"))

print(sub("Evaluating on Test Set"))
y_pred_reg = multi_reg.predict(X_test)

mae = mean_absolute_error(y_test_reg, y_pred_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg, y_pred_reg)
evs = explained_variance_score(y_test_reg, y_pred_reg)

print(f"\n{BOLD}Regression Metrics (Averaged across Depression, Anxiety, Stress):{RESET}")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R² Score : {r2:.4f}")
print(f"Explained Variance : {evs:.4f}")

joblib.dump(multi_reg, "xgb_regressor_head_B.joblib")
print(ok("Saved model artifact: xgb_regressor_head_B.joblib"))

print(f"\n{BOLD}{GREEN}{'═'*62}")
print("  Training Complete.")
print(f"{'═'*62}{RESET}\n")
