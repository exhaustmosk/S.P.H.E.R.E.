"""
sequential_stacking_pipeline_v2.py
────────────────────────────────────────────────────────────────────────────
Fixed Sequential Stacking Architecture:
  Stage 1: PCA Dimensionality Reduction (speech→64, facial→64)
  Stage 2: Head B — MultiOutput XGBRegressor (OOF severity predictions)
  Stage 3: Head A — 2-Layer Stacking Ensemble with per-fold SMOTE
  Stage 4: SHAP Explainability
────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import joblib
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    classification_report, accuracy_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, explained_variance_score,
    confusion_matrix
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
)
from xgboost import XGBRegressor, XGBClassifier
from imblearn.over_sampling import SMOTE
import shap

warnings.filterwarnings("ignore")

# ── colour helpers ────────────────────────────────────────────────────────────
RESET = "\033[0m"; BOLD = "\033[1m"; GREEN = "\033[92m"
YELLOW = "\033[93m"; CYAN = "\033[96m"; MAGENTA = "\033[95m"
def ok(m):   return f"{GREEN}✔  {m}{RESET}"
def warn(m): return f"{YELLOW}⚠  {m}{RESET}"
def hdr(m):  return f"\n{BOLD}{CYAN}{'─'*62}\n  {m}\n{'─'*62}{RESET}"
def sub(m):  return f"{MAGENTA}  ▸ {m}{RESET}"

CLASS_MAP = {"Healthy": 0, "Mild_Stress": 1, "Moderate_Stress": 2, "Severe_Stress": 3}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
CLASS_NAMES = ["Healthy", "Mild_Stress", "Moderate_Stress", "Severe_Stress"]

NUMERICAL_FEATURES = [
    "Sleep_Quality", "Social_Engagement", "Daily_App_Usage_Min",
    "Typing_Speed_WPM", "Session_Frequency", "Idle_Time_Min",
    "Facial_Emotion_Variance", "Eye_Blink_Rate", "Smile_Intensity",
    "Head_Motion_Index", "MFCC_Mean", "MFCC_Variance", "Pitch_Mean",
    "Speech_Rate", "Heart_Rate_BPM", "HRV_Index", "Skin_Temperature",
    "GSR_Level"
]
REGRESSION_TARGETS = ["Depression_Score", "Anxiety_Score", "Stress_Score"]

PCA_COMPONENTS = 64  # More components to retain richer signal

# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 1: DATA PREP & PCA
# ═══════════════════════════════════════════════════════════════════════════════
print(hdr("Stage 1 · Data Prep & PCA Dimensionality Reduction"))

num_df     = pd.read_csv("numerical_data.csv")
speech_emb = np.load("speech_embeddings.npy")   # (2880, 1024)
facial_emb = np.load("facial_embeddings.npy")   # (28709, 768)

NUM_SAMPLES = num_df.shape[0]  # 4000
print(sub(f"Ground truth rows : {NUM_SAMPLES}"))
print(sub(f"Speech embeddings : {speech_emb.shape}"))
print(sub(f"Facial embeddings : {facial_emb.shape}"))

# Align to 4000 rows
speech_aligned = np.zeros((NUM_SAMPLES, speech_emb.shape[1]), dtype=np.float32)
n_speech = min(NUM_SAMPLES, speech_emb.shape[0])
speech_aligned[:n_speech] = speech_emb[:n_speech]

facial_aligned = np.zeros((NUM_SAMPLES, facial_emb.shape[1]), dtype=np.float32)
n_facial = min(NUM_SAMPLES, facial_emb.shape[0])
facial_aligned[:n_facial] = facial_emb[:n_facial]

# Targets
y_class = num_df["Mental_Health_Status"].map(CLASS_MAP).values.astype(np.int64)
y_reg   = num_df[REGRESSION_TARGETS].values.astype(np.float32)

# 80/20 stratified split
X_num_raw = num_df[NUMERICAL_FEATURES].values.astype(np.float32)

(X_num_train, X_num_test,
 X_speech_train, X_speech_test,
 X_facial_train, X_facial_test,
 y_class_train, y_class_test,
 y_reg_train, y_reg_test) = train_test_split(
    X_num_raw, speech_aligned, facial_aligned,
    y_class, y_reg,
    test_size=0.20, random_state=42, stratify=y_class
)

print(ok(f"Train: {X_num_train.shape[0]}  |  Test: {X_num_test.shape[0]}"))

# StandardScaler on numerical
scaler = StandardScaler()
X_num_train_sc = scaler.fit_transform(X_num_train).astype(np.float32)
X_num_test_sc  = scaler.transform(X_num_test).astype(np.float32)

# PCA on speech (1024 → 64)
pca_speech = PCA(n_components=PCA_COMPONENTS, random_state=42)
X_speech_train_pca = pca_speech.fit_transform(X_speech_train).astype(np.float32)
X_speech_test_pca  = pca_speech.transform(X_speech_test).astype(np.float32)
print(ok(f"Speech PCA: 1024 → {PCA_COMPONENTS}  (variance: {pca_speech.explained_variance_ratio_.sum():.2%})"))

# PCA on facial (768 → 64)
pca_facial = PCA(n_components=PCA_COMPONENTS, random_state=42)
X_facial_train_pca = pca_facial.fit_transform(X_facial_train).astype(np.float32)
X_facial_test_pca  = pca_facial.transform(X_facial_test).astype(np.float32)
print(ok(f"Facial PCA: 768 → {PCA_COMPONENTS}   (variance: {pca_facial.explained_variance_ratio_.sum():.2%})"))

# Concatenate → X_fused
N_NUM = len(NUMERICAL_FEATURES)  # 18
X_train_fused = np.hstack([X_num_train_sc, X_speech_train_pca, X_facial_train_pca])
X_test_fused  = np.hstack([X_num_test_sc,  X_speech_test_pca,  X_facial_test_pca])

FUSED_DIM = X_train_fused.shape[1]  # 18 + 64 + 64 = 146
FEATURE_NAMES = (
    NUMERICAL_FEATURES
    + [f"Speech_PCA_{i}" for i in range(PCA_COMPONENTS)]
    + [f"Facial_PCA_{i}" for i in range(PCA_COMPONENTS)]
)
print(ok(f"X_fused: {X_train_fused.shape}  ({FUSED_DIM} features = 18 + {PCA_COMPONENTS} + {PCA_COMPONENTS})"))

# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 2: HEAD B — SEVERITY REGRESSOR
# ═══════════════════════════════════════════════════════════════════════════════
print(hdr("Stage 2 · Head B: Multi-Output Severity Regressor"))

# OOF severity predictions via 5-Fold CV
skf_reg = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_severity = np.zeros_like(y_reg_train)

t0 = time.time()
for fold_idx, (tr_idx, val_idx) in enumerate(skf_reg.split(X_train_fused, y_class_train), 1):
    fold_reg = MultiOutputRegressor(
        XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                     random_state=42, n_jobs=-1)
    )
    fold_reg.fit(X_train_fused[tr_idx], y_reg_train[tr_idx])
    oof_severity[val_idx] = fold_reg.predict(X_train_fused[val_idx])
    print(sub(f"  Fold {fold_idx}/5 done"))

print(ok(f"OOF severity predictions in {time.time()-t0:.1f}s"))

# Full regressor for test predictions
final_reg = MultiOutputRegressor(
    XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                 random_state=42, n_jobs=-1)
)
final_reg.fit(X_train_fused, y_reg_train)
test_severity = final_reg.predict(X_test_fused)

# Evaluate Head B
mae  = mean_absolute_error(y_reg_test, test_severity)
mse  = mean_squared_error(y_reg_test, test_severity)
rmse = np.sqrt(mse)
r2   = r2_score(y_reg_test, test_severity)
evs  = explained_variance_score(y_reg_test, test_severity)

print(f"\n{BOLD}Head B — Regression Metrics (Test Set):{RESET}")
print(f"  MAE               : {mae:.4f}")
print(f"  MSE               : {mse:.4f}")
print(f"  RMSE              : {rmse:.4f}")
print(f"  R² Score          : {r2:.4f}")
print(f"  Explained Variance: {evs:.4f}")

joblib.dump(final_reg, "head_B_regressor_v2.joblib")
joblib.dump(scaler, "standard_scaler_v2.joblib")
joblib.dump(pca_speech, "pca_speech_v2.joblib")
joblib.dump(pca_facial, "pca_facial_v2.joblib")

# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 3: HEAD A — 2-LAYER STACKING ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════
print(hdr("Stage 3 · Head A: 2-Layer Stacking Ensemble"))

# Append severity predictions to create stacked features
STACKED_NAMES = FEATURE_NAMES + ["Pred_Depression", "Pred_Anxiety", "Pred_Stress"]
X_train_stacked = np.hstack([X_train_fused, oof_severity])
X_test_stacked  = np.hstack([X_test_fused, test_severity])
STACKED_DIM = X_train_stacked.shape[1]

print(ok(f"Stacked features: {STACKED_DIM} ({FUSED_DIM} fused + 3 severity)"))

# ── Layer 1: Base learners with per-fold SMOTE ───────────────────────────────
# KEY FIX: We SMOTE inside each fold but predict on the ORIGINAL validation set.
# This ensures the meta-learner sees calibrated probabilities for real data.

base_configs = {
    "XGBoost": lambda: XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        objective="multi:softprob", num_class=4,
        min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=1, random_state=42, n_jobs=-1
    ),
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "ExtraTrees": lambda: ExtraTreesClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "GradientBoosting": lambda: GradientBoostingClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        random_state=42
    ),
}

NUM_CLASSES = 4
skf_cls = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
smote = SMOTE(random_state=42)

# Storage for OOF probabilities
oof_probs = {name: np.zeros((X_train_stacked.shape[0], NUM_CLASSES)) for name in base_configs}
test_probs = {name: np.zeros((X_test_stacked.shape[0], NUM_CLASSES)) for name in base_configs}

print(sub("Layer 1: Per-fold SMOTE + OOF predictions on ORIGINAL validation…"))

for name, model_fn in base_configs.items():
    t1 = time.time()
    for fold_idx, (tr_idx, val_idx) in enumerate(skf_cls.split(X_train_stacked, y_class_train), 1):
        X_fold_tr, y_fold_tr = X_train_stacked[tr_idx], y_class_train[tr_idx]
        X_fold_val = X_train_stacked[val_idx]

        # SMOTE on fold training data only
        X_fold_res, y_fold_res = smote.fit_resample(X_fold_tr, y_fold_tr)

        model = model_fn()
        model.fit(X_fold_res, y_fold_res)

        # Predict on ORIGINAL (non-SMOTE) validation fold
        oof_probs[name][val_idx] = model.predict_proba(X_fold_val)

    # Train final base model on full SMOTE-resampled training data for test predictions
    X_full_res, y_full_res = smote.fit_resample(X_train_stacked, y_class_train)
    final_model = model_fn()
    final_model.fit(X_full_res, y_full_res)
    test_probs[name] = final_model.predict_proba(X_test_stacked)

    # Save for SHAP later (keep the XGBoost one)
    if name == "XGBoost":
        xgb_base_model = final_model

    print(sub(f"  {name} done ({time.time()-t1:.1f}s)"))

# ── Layer 2: Meta-learner ────────────────────────────────────────────────────
print(sub("Layer 2: Training XGBoost meta-learner…"))

# Stack all OOF probabilities → (N_train, 4_models * 4_classes = 16)
meta_train = np.hstack([oof_probs[name] for name in base_configs])
meta_test  = np.hstack([test_probs[name] for name in base_configs])

# SMOTE the meta features too
meta_train_res, y_meta_res = smote.fit_resample(meta_train, y_class_train)

meta_clf = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    objective="multi:softprob", num_class=4,
    random_state=42, n_jobs=-1
)
meta_clf.fit(meta_train_res, y_meta_res)
print(ok("Meta-learner trained on stacked OOF probabilities"))

# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 4: EVALUATION & SHAP
# ═══════════════════════════════════════════════════════════════════════════════
print(hdr("Stage 4 · Final Evaluation"))

y_pred_class = meta_clf.predict(meta_test)
y_pred_proba = meta_clf.predict_proba(meta_test)

acc = accuracy_score(y_class_test, y_pred_class)
try:
    roc = roc_auc_score(y_class_test, y_pred_proba, multi_class="ovr")
    roc_str = f"{roc:.4f}"
except:
    roc_str = "N/A"

print(f"\n{BOLD}Classification Metrics (Test Set — {X_test_stacked.shape[0]} samples):{RESET}")
print(f"Accuracy : {acc:.4f}")
print(f"ROC-AUC  : {roc_str}")
print(f"\n{classification_report(y_class_test, y_pred_class, target_names=CLASS_NAMES)}")

# Confusion Matrix
cm = confusion_matrix(y_class_test, y_pred_class)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix — Sequential Stacking v2")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_stacking_v2.png", dpi=300)
print(ok("Saved confusion_matrix_stacking_v2.png"))

# ── SHAP on the Layer 1 XGBoost base model ───────────────────────────────────
print(hdr("Stage 4b · SHAP Explainability"))
print(sub("Using Layer 1 XGBoost base model for TreeSHAP"))

X_shap_df = pd.DataFrame(X_test_stacked, columns=STACKED_NAMES)
explainer = shap.TreeExplainer(xgb_base_model)
shap_values = explainer.shap_values(X_shap_df)

# Mean absolute SHAP per feature
if isinstance(shap_values, list):
    abs_shap = np.zeros(len(STACKED_NAMES))
    for sv in shap_values:
        abs_shap += np.abs(sv).mean(axis=0)
    abs_shap /= len(shap_values)
elif len(shap_values.shape) == 3:
    abs_shap = np.abs(shap_values).mean(axis=0).mean(axis=1)
else:
    abs_shap = np.abs(shap_values).mean(axis=0)

# Modality aggregation
tabular_idx  = list(range(0, N_NUM))                                    # 0–17
speech_idx   = list(range(N_NUM, N_NUM + PCA_COMPONENTS))               # 18–81
facial_idx   = list(range(N_NUM + PCA_COMPONENTS, FUSED_DIM))           # 82–145
severity_idx = list(range(FUSED_DIM, STACKED_DIM))                     # 146–148

tabular_imp  = abs_shap[tabular_idx].sum() + abs_shap[severity_idx].sum()
speech_imp   = abs_shap[speech_idx].sum()
facial_imp   = abs_shap[facial_idx].sum()
total_imp    = tabular_imp + speech_imp + facial_imp

pcts = [
    tabular_imp / total_imp * 100,
    speech_imp / total_imp * 100,
    facial_imp / total_imp * 100,
]
mod_labels = [
    "Tabular/Physiological\n(+Severity Predictions)",
    "Acoustic/Speech\n(PCA-64)",
    "Visual/Facial\n(PCA-64)"
]

print(f"\n{BOLD}Modality Contributions:{RESET}")
for lab, pct in zip(mod_labels, pcts):
    print(f"  {lab.split(chr(10))[0]:35s} {pct:6.2f}%")

# Plot 1: Modality bar chart
plt.figure(figsize=(9, 5))
bars = plt.barh(mod_labels, pcts, color=["#2196F3", "#FF9800", "#4CAF50"], edgecolor="white")
for bar, pct in zip(bars, pcts):
    plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f"{pct:.1f}%", va="center", fontweight="bold")
plt.xlabel("Percentage Contribution (%)")
plt.title("Modality-Level Feature Importance (SHAP)")
plt.tight_layout()
plt.savefig("modality_bar_chart_v2.png", dpi=300)
print(ok("Saved modality_bar_chart_v2.png"))

# Plot 2: Top 15 features
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_shap_df, plot_type="bar",
                  max_display=15, show=False, class_names=CLASS_NAMES)
plt.title("Top 15 Individual Features (SHAP Value)")
plt.tight_layout()
plt.savefig("top_15_shap_v2.png", dpi=300)
print(ok("Saved top_15_shap_v2.png"))

# Save all model artifacts
joblib.dump(xgb_base_model, "head_A_xgb_base_v2.joblib")
joblib.dump(meta_clf, "head_A_meta_learner_v2.joblib")

print(f"\n{BOLD}{GREEN}{'═'*62}")
print("  Sequential Stacking Pipeline v2 Complete.")
print(f"{'═'*62}{RESET}\n")
