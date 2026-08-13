"""
autogluon_semantic_stacking_pipeline.py
────────────────────────────────────────────────────────────────────────────
End-to-End Sequential Stacking Pipeline with Semantic Target Alignment

Architecture:
  Stage 1 — Semantic Target Alignment & PCA
  Stage 2 — Head B: MultiOutput XGBRegressor (OOF severity predictions)
  Stage 3 — Head A: AutoGluon Multi-Layer Stack Ensemble (classification)
  Stage 4 — SHAP Modality Explainability Engine

Run with:  .venv312/bin/python autogluon_semantic_stacking_pipeline.py
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
from sklearn.model_selection import train_test_split, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    classification_report, accuracy_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, explained_variance_score,
    confusion_matrix
)
from xgboost import XGBRegressor, XGBClassifier
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
PCA_N = 32

# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 1: SEMANTIC TARGET ALIGNMENT & PCA
# ═══════════════════════════════════════════════════════════════════════════════
print(hdr("Stage 1 · Semantic Target Alignment & PCA"))

num_df     = pd.read_csv("numerical_data.csv")
speech_emb = np.load("speech_embeddings.npy")   # (2880, 1024)
facial_emb = np.load("facial_embeddings.npy")   # (28709, 768)

speech_meta = pd.read_csv("speech_metadata.csv")
facial_meta = pd.read_csv("facial_metadata.csv")

print(sub(f"Numerical CSV     : {num_df.shape}"))
print(sub(f"Speech embeddings : {speech_emb.shape}"))
print(sub(f"Facial embeddings : {facial_emb.shape}"))

N_COMPLETE = num_df.shape[0]

# Semantic mapping definition
SPEECH_MAP = {
    "Healthy": ["happy", "neutral", "surprised"],
    "Mild_Stress": ["neutral", "sad"],
    "Moderate_Stress": ["sad", "angry", "disgust"],
    "Severe_Stress": ["angry", "fearful", "disgust"]
}
FACIAL_MAP = {
    "Healthy": ["Happy", "Neutral", "Surprise"],
    "Mild_Stress": ["Neutral", "Sad"],
    "Moderate_Stress": ["Sad", "Angry", "Disgust"],
    "Severe_Stress": ["Angry", "Fear", "Disgust"]
}

# Group indices by mapping pool
speech_pools = {}
facial_pools = {}

for mhs in CLASS_MAP.keys():
    # Find all indices in metadata matching these emotions
    s_emotions = SPEECH_MAP[mhs]
    f_emotions = FACIAL_MAP[mhs]
    
    s_idx = speech_meta.index[speech_meta["Emotion_Label"].isin(s_emotions)].tolist()
    f_idx = facial_meta.index[facial_meta["Class_Label"].isin(f_emotions)].tolist()
    
    speech_pools[mhs] = s_idx
    facial_pools[mhs] = f_idx
    print(sub(f"Pool '{mhs}': {len(s_idx)} acoustic, {len(f_idx)} visual frames available"))

# Synthetically construct exactly 4000 rows perfectly aligned semantically
print(sub(f"\nSynthesizing 4,000 clinically cohesive multimodal records..."))

np.random.seed(42) # Replicability

X_num_raw = num_df[NUMERICAL_FEATURES].values.astype(np.float32)
y_class = num_df["Mental_Health_Status"].values
y_class_int = np.array([CLASS_MAP[c] for c in y_class], dtype=np.int64)
y_reg = num_df[REGRESSION_TARGETS].values.astype(np.float32)

X_speech = np.zeros((N_COMPLETE, speech_emb.shape[1]), dtype=np.float32)
X_facial = np.zeros((N_COMPLETE, facial_emb.shape[1]), dtype=np.float32)

for i, status in enumerate(y_class):
    # Randomly sample exactly 1 from respective pool
    sampled_s_idx = np.random.choice(speech_pools[status])
    sampled_f_idx = np.random.choice(facial_pools[status])
    
    X_speech[i] = speech_emb[sampled_s_idx]
    X_facial[i] = facial_emb[sampled_f_idx]

print(ok(f"Semantic Alignment complete. Generated ({X_speech.shape[0]}, 1024) audio and ({X_facial.shape[0]}, 768) face matrices."))

# ── 80/20 Stratified Split ───────────────────────────────────────────────────
(X_num_train, X_num_test,
 X_speech_train, X_speech_test,
 X_facial_train, X_facial_test,
 y_class_train, y_class_test,
 y_class_int_train, y_class_int_test,
 y_reg_train, y_reg_test) = train_test_split(
    X_num_raw, X_speech, X_facial,
    y_class, y_class_int, y_reg,
    test_size=0.20, random_state=42, stratify=y_class_int
)

print(ok(f"Train: {X_num_train.shape[0]} samples  |  Test: {X_num_test.shape[0]} samples"))

# ── StandardScaler on numerical (fit on train only) ──────────────────────────
scaler = StandardScaler()
X_num_train_sc = scaler.fit_transform(X_num_train).astype(np.float32)
X_num_test_sc  = scaler.transform(X_num_test).astype(np.float32)
print(ok("StandardScaler fitted on train numerical features"))

# ── PCA on speech (1024 → 32) ────────────────────────────────────────────────
pca_speech = PCA(n_components=PCA_N, random_state=42)
X_speech_train_pca = pca_speech.fit_transform(X_speech_train).astype(np.float32)
X_speech_test_pca  = pca_speech.transform(X_speech_test).astype(np.float32)
var_speech = pca_speech.explained_variance_ratio_.sum()
print(ok(f"Speech PCA: 1024 → {PCA_N}  (variance retained: {var_speech:.2%})"))

# ── PCA on facial (768 → 32) ─────────────────────────────────────────────────
pca_facial = PCA(n_components=PCA_N, random_state=42)
X_facial_train_pca = pca_facial.fit_transform(X_facial_train).astype(np.float32)
X_facial_test_pca  = pca_facial.transform(X_facial_test).astype(np.float32)
var_facial = pca_facial.explained_variance_ratio_.sum()
print(ok(f"Facial PCA: 768 → {PCA_N}   (variance retained: {var_facial:.2%})"))

# ── Concatenate → X_fused (82 features) ──────────────────────────────────────
X_train_fused = np.hstack([X_num_train_sc, X_speech_train_pca, X_facial_train_pca])
X_test_fused  = np.hstack([X_num_test_sc,  X_speech_test_pca,  X_facial_test_pca])

FUSED_NAMES = (
    NUMERICAL_FEATURES
    + [f"Speech_PCA_{i}" for i in range(PCA_N)]
    + [f"Facial_PCA_{i}" for i in range(PCA_N)]
)
print(ok(f"X_train_fused: {X_train_fused.shape}  |  X_test_fused: {X_test_fused.shape}"))
print(ok(f"Total fused features: {len(FUSED_NAMES)} (18 tabular + {PCA_N} speech + {PCA_N} facial)"))

# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 2: HEAD B — SYMPTOM SEVERITY REGRESSOR (Objective 2)
# ═══════════════════════════════════════════════════════════════════════════════
print(hdr("Stage 2 · Head B: Multi-Output Severity Regressor"))

REG_PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)

# ── 5-Fold OOF predictions ───────────────────────────────────────────────────
print(sub("Generating OOF severity predictions (5-Fold CV)…"))
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_severity = np.zeros_like(y_reg_train)

t0 = time.time()
for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train_fused), 1):
    fold_reg = MultiOutputRegressor(XGBRegressor(**REG_PARAMS))
    fold_reg.fit(X_train_fused[tr_idx], y_reg_train[tr_idx])
    oof_severity[val_idx] = fold_reg.predict(X_train_fused[val_idx])
    print(sub(f"  Fold {fold_idx}/5 complete"))

print(ok(f"OOF severity predictions generated in {time.time()-t0:.1f}s"))

# ── Final regressor for test predictions ──────────────────────────────────────
print(sub("Training final regressor on full training set…"))
final_reg = MultiOutputRegressor(XGBRegressor(**REG_PARAMS))
final_reg.fit(X_train_fused, y_reg_train)
test_severity = final_reg.predict(X_test_fused)

# ── Evaluate Head B ──────────────────────────────────────────────────────────
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

joblib.dump(final_reg, "head_B_regressor_semantic.joblib")
joblib.dump(scaler, "standard_scaler_semantic.joblib")
joblib.dump(pca_speech, "pca_speech_semantic.joblib")
joblib.dump(pca_facial, "pca_facial_semantic.joblib")
print(ok("Saved Head B model + preprocessing artifacts"))

# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 3: HEAD A — AUTOGLUON STACK ENSEMBLE (Objective 1)
# ═══════════════════════════════════════════════════════════════════════════════
print(hdr("Stage 3 · Head A: AutoGluon Multi-Layer Stack Ensemble"))

# ── Construct stacked DataFrames ──────────────────────────────────────────────
STACKED_NAMES = FUSED_NAMES + ["pred_depression", "pred_anxiety", "pred_stress"]

X_train_stacked = np.hstack([X_train_fused, oof_severity])
X_test_stacked  = np.hstack([X_test_fused, test_severity])

train_df = pd.DataFrame(X_train_stacked, columns=STACKED_NAMES)
train_df["Mental_Health_Status"] = y_class_train

test_df = pd.DataFrame(X_test_stacked, columns=STACKED_NAMES)
test_df["Mental_Health_Status"] = y_class_test

print(ok(f"train_df: {train_df.shape}  |  test_df: {test_df.shape}"))
print(sub(f"Features: {len(STACKED_NAMES)} (82 fused + 3 severity predictions)"))
print(sub(f"Label: Mental_Health_Status"))

# ── AutoGluon Training ───────────────────────────────────────────────────────
from autogluon.tabular import TabularPredictor

print(sub("Fitting AutoGluon TabularPredictor…"))
print(sub("  presets='best_quality', auto_stack=True, time_limit=1200"))

t0 = time.time()
predictor = TabularPredictor(
    label="Mental_Health_Status",
    eval_metric="f1_macro",
    problem_type="multiclass",
    path="autogluon_models_semantic",
).fit(
    train_data=train_df,
    presets="best_quality",
    time_limit=1200,
    auto_stack=True,
)
elapsed = time.time() - t0
print(ok(f"AutoGluon training complete in {elapsed:.0f}s"))

# ── Evaluation ────────────────────────────────────────────────────────────────
print(hdr("Stage 3b · Head A Evaluation"))

test_features = test_df.drop(columns=["Mental_Health_Status"])
y_pred_labels = predictor.predict(test_features)
y_pred_int    = np.array([CLASS_MAP[l] for l in y_pred_labels])

# Leaderboard
print(f"\n{BOLD}AutoGluon Leaderboard:{RESET}")
lb = predictor.leaderboard(test_df, silent=True)
print(lb.to_string())

# Metrics
acc = accuracy_score(y_class_int_test, y_pred_int)
print(f"\n{BOLD}Classification Metrics (Test Set — {len(y_pred_int)} samples):{RESET}")
print(f"  Accuracy : {acc:.4f}")

try:
    y_pred_proba = predictor.predict_proba(test_features)
    # Ensure column order matches CLASS_NAMES
    proba_ordered = y_pred_proba[[INV_CLASS_MAP[i] for i in range(4)]].values
    roc = roc_auc_score(y_class_int_test, proba_ordered, multi_class="ovr")
    print(f"  ROC-AUC  : {roc:.4f}")
except Exception as e:
    print(warn(f"  ROC-AUC could not be computed: {e}"))

print(f"\n{classification_report(y_class_int_test, y_pred_int, target_names=CLASS_NAMES)}")

# ── Confusion Matrix ─────────────────────────────────────────────────────────
cm = confusion_matrix(y_class_int_test, y_pred_int)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix — AutoGluon Stack Ensemble")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_semantic.png", dpi=300)
print(ok("Saved confusion_matrix_semantic.png"))

# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 4: SHAP MODALITY EXPLAINABILITY (Objective 3)
# ═══════════════════════════════════════════════════════════════════════════════
print(hdr("Stage 4 · SHAP Modality Explainability Engine"))

# Train an XGBClassifier on the same stacked features for SHAP TreeExplainer
# (AutoGluon's internal models are not directly accessible for SHAP)
print(sub("Training XGBClassifier proxy for TreeSHAP…"))

shap_clf = XGBClassifier(
    n_estimators=500, max_depth=8, learning_rate=0.05,
    objective="multi:softprob", num_class=4,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1
)
shap_clf.fit(X_train_stacked, y_class_int_train)
print(ok("XGBClassifier proxy trained"))

# ── SHAP values ──────────────────────────────────────────────────────────────
print(sub("Computing SHAP values on test set…"))
X_shap_df = pd.DataFrame(X_test_stacked, columns=STACKED_NAMES)

explainer = shap.TreeExplainer(shap_clf)
shap_values = explainer.shap_values(X_shap_df)

# Mean absolute SHAP per feature (across classes)
if isinstance(shap_values, list):
    abs_shap = np.zeros(len(STACKED_NAMES))
    for sv in shap_values:
        abs_shap += np.abs(sv).mean(axis=0)
    abs_shap /= len(shap_values)
elif len(shap_values.shape) == 3:
    abs_shap = np.abs(shap_values).mean(axis=0).mean(axis=1)
else:
    abs_shap = np.abs(shap_values).mean(axis=0)

print(ok("SHAP values computed"))

# ── Modality Aggregation ─────────────────────────────────────────────────────
N_NUM = len(NUMERICAL_FEATURES)  # 18
# Indices: tabular=0..17, speech_pca=18..49, facial_pca=50..81, severity=82..84
tabular_imp  = abs_shap[:N_NUM].sum() + abs_shap[82:85].sum()   # tabular + severity
speech_imp   = abs_shap[N_NUM:N_NUM+PCA_N].sum()                 # speech PCA
facial_imp   = abs_shap[N_NUM+PCA_N:N_NUM+2*PCA_N].sum()        # facial PCA
total_imp    = tabular_imp + speech_imp + facial_imp

pcts = [
    tabular_imp / total_imp * 100,
    speech_imp / total_imp * 100,
    facial_imp / total_imp * 100,
]
mod_labels = [
    "Physiological & Behavioral\n(Tabular + Severity Predictions)",
    "Acoustic / Speech\n(32 PCA Components)",
    "Visual / Facial\n(32 PCA Components)"
]

print(f"\n{BOLD}Modality Contributions:{RESET}")
for lab, pct in zip(mod_labels, pcts):
    name = lab.split(chr(10))[0]
    print(f"  {name:40s} {pct:6.2f}%")

# ── Plot 1: Modality Attribution Bar Chart ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#1976D2", "#FF6F00", "#2E7D32"]
bars = ax.barh(mod_labels, pcts, color=colors, edgecolor="white", height=0.6)
for bar, pct in zip(bars, pcts):
    ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2,
            f"{pct:.1f}%", va="center", fontweight="bold", fontsize=12)
ax.set_xlabel("Percentage Contribution (%)", fontsize=12)
ax.set_title("Modality-Level Feature Attribution (SHAP)", fontsize=14, fontweight="bold")
ax.set_xlim(0, max(pcts) * 1.2)
plt.tight_layout()
plt.savefig("modality_attribution_bar_chart_semantic.png", dpi=300, bbox_inches="tight")
print(ok("Saved modality_attribution_bar_chart_semantic.png"))

# ── Plot 2: Top 15 Features ──────────────────────────────────────────────────
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values, X_shap_df, plot_type="bar",
    max_display=15, show=False,
    class_names=CLASS_NAMES,
)
plt.title("Top 15 Individual Features (SHAP Value)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("top_15_shap_features_semantic.png", dpi=300, bbox_inches="tight")
print(ok("Saved top_15_shap_features_semantic.png"))

# ── Save final artifacts ─────────────────────────────────────────────────────
joblib.dump(shap_clf, "shap_xgb_proxy_semantic.joblib")

print(f"\n{BOLD}{GREEN}{'═'*62}")
print("  End-to-End Semantic Stacking Pipeline Complete.")
print(f"{'═'*62}{RESET}\n")
