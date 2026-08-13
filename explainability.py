"""
explainability.py
────────────────────────────────────────────────────────────────────────────
Explainable AI (XAI) using SHAP
Generates modality-level and feature-level importance visualizations.
────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import shap

import warnings
warnings.filterwarnings('ignore')

# ── colour helpers ────────────────────────────────────────────────────────────
RESET = "\033[0m"; BOLD = "\033[1m"; GREEN = "\033[92m"
YELLOW = "\033[93m"; RED = "\033[91m"; CYAN = "\033[96m"; MAGENTA = "\033[95m"
def ok(m):   return f"{GREEN}✔  {m}{RESET}"
def hdr(m):  return f"\n{BOLD}{CYAN}{'─'*62}\n  {m}\n{'─'*62}{RESET}"
def sub(m):  return f"{MAGENTA}  ▸ {m}{RESET}"

# ── 1. LOAD DATA & MODEL ─────────────────────────────────────────────────────
print(hdr("1 · Loading Model & Test Set"))

DATA_DIR = Path("dataset_splits")
X_test = np.load(DATA_DIR / "X_test.npy")

NUMERICAL_FEATURES = [
    "Sleep_Quality", "Social_Engagement", "Daily_App_Usage_Min", 
    "Typing_Speed_WPM", "Session_Frequency", "Idle_Time_Min", 
    "Facial_Emotion_Variance", "Eye_Blink_Rate", "Smile_Intensity", 
    "Head_Motion_Index", "MFCC_Mean", "MFCC_Variance", "Pitch_Mean", 
    "Speech_Rate", "Heart_Rate_BPM", "HRV_Index", "Skin_Temperature", 
    "GSR_Level"
]
num_speech_features = 1024
num_facial_features = 768

speech_features = [f"Speech_Emb_{i}" for i in range(num_speech_features)]
facial_features = [f"Facial_Emb_{i}" for i in range(num_facial_features)]

feature_names = NUMERICAL_FEATURES + speech_features + facial_features

X_test_df = pd.DataFrame(X_test, columns=feature_names)

clf = joblib.load("xgb_classifier_head_A.joblib")
print(ok("Loaded XGBClassifier Model and X_test data"))

# ── 2. CALCULATE SHAP VALUES ─────────────────────────────────────────────────
print(hdr("2 · Calculating SHAP Values (TreeExplainer)"))

# XGBoost tree explainer
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test_df)
print(ok("SHAP values calculated successfully"))

# If shap_values is a list (multi-class), calculate mean absolute SHAP over classes & samples
if isinstance(shap_values, list):
    # shap_values is list of (num_samples, num_features) arrays
    # Average absolute shap value for each feature over all samples and classes
    abs_shap = np.zeros(X_test_df.shape[1])
    for sv in shap_values:
        abs_shap += np.abs(sv).mean(axis=0)
    abs_shap /= len(shap_values)
elif len(shap_values.shape) == 3:
    # shape is (num_samples, num_features, num_classes)
    abs_shap = np.abs(shap_values).mean(axis=0).mean(axis=1)
else:
    abs_shap = np.abs(shap_values).mean(axis=0)

# ── 3. MODALITY IMPORTANCE (BAR CHART) ───────────────────────────────────────
print(hdr("3 · Generating Modality Importance Chart"))

tabular_imp = np.sum(abs_shap[:18])
speech_imp  = np.sum(abs_shap[18:18+1024])
facial_imp  = np.sum(abs_shap[18+1024:])

total_imp = tabular_imp + speech_imp + facial_imp
modality_percentages = [
    (tabular_imp / total_imp) * 100,
    (speech_imp / total_imp) * 100,
    (facial_imp / total_imp) * 100
]
modalities = ["Tabular/Physiological", "Acoustic/Speech", "Visual/Facial"]

plt.figure(figsize=(8, 5))
sns.barplot(x=modality_percentages, y=modalities, palette="viridis")
plt.xlabel("Percentage Contribution to Model Output (%)")
plt.title("Feature Importance by Modality")
plt.tight_layout()
plt.savefig("modality_bar_chart.png", dpi=300)
print(ok("Saved modality_bar_chart.png"))

# ── 4. TOP 15 SHAP FEATURES ──────────────────────────────────────────────────
print(hdr("4 · Generating Top 15 SHAP Features Plot"))

plt.figure(figsize=(10, 6))
# For multiclass, passing the list of shap_values produces a stacked bar plot
# which looks great and highlights class-specific importances.
shap.summary_plot(
    shap_values, X_test_df, plot_type="bar", 
    max_display=15, show=False, class_names=["Healthy", "Mild_Stress", "Moderate_Stress", "Severe_Stress"]
)
plt.title("Top 15 Individual Features (SHAP Value)")
plt.tight_layout()
plt.savefig("top_15_shap_features.png", dpi=300)
print(ok("Saved top_15_shap_features.png"))

print(f"\n{BOLD}{GREEN}{'═'*62}")
print("  Explainability Pipeline Complete.")
print(f"{'═'*62}{RESET}\n")
