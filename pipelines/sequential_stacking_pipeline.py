"""
sequential_stacking_pipeline.py
────────────────────────────────────────────────────────────────────────────
Sequential Stacking Architecture:
  Stage 1: PCA Dimensionality Reduction (1024→32 speech, 768→32 facial)
  Stage 2: Head B — MultiOutput XGBRegressor (OOF severity predictions)
  Stage 3: Head A — AutoGluon Multi-Layer Stack Ensemble (classification)
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
from sklearn.model_selection import train_test_split, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import (
    classification_report, accuracy_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, explained_variance_score,
    confusion_matrix
)
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ── colour helpers ────────────────────────────────────────────────────────────
RESET = "\033[0m"; BOLD = "\033[1m"; GREEN = "\033[92m"
YELLOW = "\033[93m"; CYAN = "\033[96m"; MAGENTA = "\033[95m"
def ok(m):   return f"{GREEN}✔  {m}{RESET}"
def warn(m): return f"{YELLOW}⚠  {m}{RESET}"
def hdr(m):  return f"\n{BOLD}{CYAN}{'─'*62}\n  {m}\n{'─'*62}{RESET}"
def sub(m):  return f"{MAGENTA}  ▸ {m}{RESET}"

# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 1: DATA PREP & PCA DIMENSIONALITY REDUCTION
# ═══════════════════════════════════════════════════════════════════════════════
print(hdr("Stage 1 · Data Prep & PCA Dimensionality Reduction"))

# ── Load raw data ─────────────────────────────────────────────────────────────
num_df    = pd.read_csv("numerical_data.csv")
speech_emb = np.load("speech_embeddings.npy")   # (2880, 1024)
facial_emb = np.load("facial_embeddings.npy")   # (28709, 768)

NUM_SAMPLES = num_df.shape[0]   # 4000 — ground truth rows
print(sub(f"Numerical CSV     : {num_df.shape}  (ground truth)"))
print(sub(f"Speech embeddings : {speech_emb.shape}"))
print(sub(f"Facial embeddings : {facial_emb.shape}"))

NUMERICAL_FEATURES = [
    "Sleep_Quality", "Social_Engagement", "Daily_App_Usage_Min",
    "Typing_Speed_WPM", "Session_Frequency", "Idle_Time_Min",
    "Facial_Emotion_Variance", "Eye_Blink_Rate", "Smile_Intensity",
    "Head_Motion_Index", "MFCC_Mean", "MFCC_Variance", "Pitch_Mean",
    "Speech_Rate", "Heart_Rate_BPM", "HRV_Index", "Skin_Temperature",
    "GSR_Level"
]

REGRESSION_TARGETS = ["Depression_Score", "Anxiety_Score", "Stress_Score"]

CLASS_MAP = {"Healthy": 0, "Mild_Stress": 1, "Moderate_Stress": 2, "Severe_Stress": 3}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

# ── Align to NUM_SAMPLES (4000) ──────────────────────────────────────────────
# Zero-pad modalities that are shorter; truncate those that are longer
speech_aligned = np.zeros((NUM_SAMPLES, speech_emb.shape[1]), dtype=np.float32)
n_speech = min(NUM_SAMPLES, speech_emb.shape[0])
speech_aligned[:n_speech] = speech_emb[:n_speech]

facial_aligned = np.zeros((NUM_SAMPLES, facial_emb.shape[1]), dtype=np.float32)
n_facial = min(NUM_SAMPLES, facial_emb.shape[0])
facial_aligned[:n_facial] = facial_emb[:n_facial]

print(ok(f"Aligned speech to ({NUM_SAMPLES}, {speech_aligned.shape[1]})  [{n_speech} real, {NUM_SAMPLES - n_speech} padded]"))
print(ok(f"Aligned facial to ({NUM_SAMPLES}, {facial_aligned.shape[1]})  [{n_facial} real, {0 if n_facial >= NUM_SAMPLES else NUM_SAMPLES - n_facial} padded]"))

# ── Extract targets ───────────────────────────────────────────────────────────
y_class = num_df["Mental_Health_Status"].map(CLASS_MAP).values.astype(np.int64)
y_reg   = num_df[REGRESSION_TARGETS].values.astype(np.float32)

print(sub(f"y_class shape: {y_class.shape}  |  y_reg shape: {y_reg.shape}"))
print(sub(f"Class distribution: {dict(zip(*np.unique(y_class, return_counts=True)))}"))

# ── 80/20 Stratified Split ───────────────────────────────────────────────────
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

print(ok(f"Train: {X_num_train.shape[0]} samples  |  Test: {X_num_test.shape[0]} samples"))

# ── StandardScaler on numerical features (fit on train only) ──────────────────
scaler = StandardScaler()
X_num_train_sc = scaler.fit_transform(X_num_train)
X_num_test_sc  = scaler.transform(X_num_test)
print(ok("Numerical features standardised (fit on train)"))

# ── PCA on speech embeddings (1024 → 32) ─────────────────────────────────────
pca_speech = PCA(n_components=32, random_state=42)
X_speech_train_pca = pca_speech.fit_transform(X_speech_train)
X_speech_test_pca  = pca_speech.transform(X_speech_test)
var_speech = pca_speech.explained_variance_ratio_.sum()
print(ok(f"Speech PCA: 1024 → 32  (variance retained: {var_speech:.2%})"))

# ── PCA on facial embeddings (768 → 32) ──────────────────────────────────────
pca_facial = PCA(n_components=32, random_state=42)
X_facial_train_pca = pca_facial.fit_transform(X_facial_train)
X_facial_test_pca  = pca_facial.transform(X_facial_test)
var_facial = pca_facial.explained_variance_ratio_.sum()
print(ok(f"Facial PCA: 768 → 32   (variance retained: {var_facial:.2%})"))

# ── Concatenate → X_fused (82 features) ──────────────────────────────────────
X_train_fused = np.hstack([X_num_train_sc, X_speech_train_pca, X_facial_train_pca])
X_test_fused  = np.hstack([X_num_test_sc, X_speech_test_pca, X_facial_test_pca])

FEATURE_NAMES = (
    NUMERICAL_FEATURES
    + [f"Speech_PCA_{i}" for i in range(32)]
    + [f"Facial_PCA_{i}" for i in range(32)]
)

print(ok(f"X_train_fused: {X_train_fused.shape}  |  X_test_fused: {X_test_fused.shape}"))
print(ok(f"Total features: {len(FEATURE_NAMES)}"))

# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 2: HEAD B — SYMPTOM SEVERITY REGRESSOR (OBJECTIVE 2)
# ═══════════════════════════════════════════════════════════════════════════════
print(hdr("Stage 2 · Head B: Multi-Output Severity Regressor"))

base_reg = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

# ── Out-Of-Fold (OOF) predictions via 5-Fold CV ──────────────────────────────
print(sub("Generating OOF severity predictions (5-Fold CV)…"))
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_severity = np.zeros_like(y_reg_train)

t0 = time.time()
for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train_fused), 1):
    fold_reg = MultiOutputRegressor(
        XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                     random_state=42, n_jobs=-1)
    )
    fold_reg.fit(X_train_fused[tr_idx], y_reg_train[tr_idx])
    oof_severity[val_idx] = fold_reg.predict(X_train_fused[val_idx])
    print(sub(f"  Fold {fold_idx}/5 complete"))

print(ok(f"OOF severity predictions generated in {time.time()-t0:.1f}s"))

# ── Train final regressor on full training data ──────────────────────────────
print(sub("Training final regressor on full training set…"))
final_reg = MultiOutputRegressor(
    XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                 random_state=42, n_jobs=-1)
)
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

joblib.dump(final_reg, "head_B_regressor.joblib")
joblib.dump(scaler, "standard_scaler.joblib")
joblib.dump(pca_speech, "pca_speech.joblib")
joblib.dump(pca_facial, "pca_facial.joblib")
print(ok("Saved Head B model + preprocessing artifacts"))

# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 3: HEAD A — AUTOGLUON STACK ENSEMBLE (OBJECTIVE 1)
# ═══════════════════════════════════════════════════════════════════════════════
print(hdr("Stage 3 · Head A: AutoGluon Multi-Layer Stack Ensemble"))

# ── Construct stacked feature DataFrames ──────────────────────────────────────
# Append the 3 OOF severity predictions to X_train_fused → 85 features
STACKED_FEATURE_NAMES = FEATURE_NAMES + ["Pred_Depression", "Pred_Anxiety", "Pred_Stress"]

X_train_stacked = np.hstack([X_train_fused, oof_severity])
X_test_stacked  = np.hstack([X_test_fused, test_severity])

train_df = pd.DataFrame(X_train_stacked, columns=STACKED_FEATURE_NAMES)
train_df["Mental_Health_Status"] = [INV_CLASS_MAP[c] for c in y_class_train]

test_df = pd.DataFrame(X_test_stacked, columns=STACKED_FEATURE_NAMES)
test_df["Mental_Health_Status"]  = [INV_CLASS_MAP[c] for c in y_class_test]

print(ok(f"train_df: {train_df.shape}  |  test_df: {test_df.shape}"))
print(sub(f"Features: {len(STACKED_FEATURE_NAMES)} (82 fused + 3 severity predictions)"))

# ── AutoGluon Training ───────────────────────────────────────────────────────
try:
    from autogluon.tabular import TabularPredictor

    print(sub("Fitting AutoGluon TabularPredictor (best_quality, auto_stack)…"))
    t0 = time.time()

    predictor = TabularPredictor(
        label="Mental_Health_Status",
        eval_metric="f1_macro",
        path="autogluon_models",
    ).fit(
        train_data=train_df,
        presets="best_quality",
        time_limit=1200,
        auto_stack=True,
    )

    elapsed = time.time() - t0
    print(ok(f"AutoGluon training complete in {elapsed:.0f}s"))
    AUTOGLUON_AVAILABLE = True

except ImportError:
    print(warn("AutoGluon not available on Python 3.14 — falling back to manual ensemble"))
    AUTOGLUON_AVAILABLE = False

    # ── Manual Multi-Layer Stacking Fallback ─────────────────────────────────
    # Layer 1: Diverse base learners
    # Layer 2: Meta-learner stacking
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        ExtraTreesClassifier
    )
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_predict
    from imblearn.over_sampling import SMOTE

    print(sub("Building manual 2-layer stacking ensemble…"))

    # SMOTE on train only
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_stacked, y_class_train)
    print(ok(f"SMOTE resampled: {X_train_res.shape[0]} samples (balanced)"))

    # Layer 1: Base learners with OOF predictions
    base_models = {
        "XGBoost": XGBClassifier(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            objective="multi:softprob", num_class=4,
            min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            random_state=42
        ),
    }

    # Generate OOF probabilities for each base model (Layer 1)
    print(sub("Layer 1: Generating OOF predictions from 4 base learners…"))
    oof_probs_list = []
    test_probs_list = []

    for name, model in base_models.items():
        t1 = time.time()
        # OOF predictions on resampled data
        oof_proba = cross_val_predict(
            model, X_train_res, y_train_res,
            cv=5, method="predict_proba", n_jobs=-1
        )
        oof_probs_list.append(oof_proba)

        # Fit on full resampled data and predict test
        model.fit(X_train_res, y_train_res)
        test_proba = model.predict_proba(X_test_stacked)
        test_probs_list.append(test_proba)
        print(sub(f"  {name} done ({time.time()-t1:.1f}s)"))

    # Layer 2: Meta-learner stacking
    print(sub("Layer 2: Training XGBoost meta-learner on stacked OOF predictions…"))
    meta_train = np.hstack(oof_probs_list)   # (N_resampled, 4*4=16)
    meta_test  = np.hstack(test_probs_list)

    meta_clf = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        objective="multi:softprob", num_class=4,
        random_state=42, n_jobs=-1
    )
    meta_clf.fit(meta_train, y_train_res)

    # Save ensemble artifacts
    joblib.dump(base_models, "layer1_base_models.joblib")
    joblib.dump(meta_clf, "layer2_meta_learner.joblib")
    print(ok("Manual 2-layer stacking ensemble trained"))

# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 4: FINAL EVALUATION & SHAP (OBJECTIVE 3)
# ═══════════════════════════════════════════════════════════════════════════════
print(hdr("Stage 4 · Final Evaluation & SHAP Explainability"))

CLASS_NAMES = ["Healthy", "Mild_Stress", "Moderate_Stress", "Severe_Stress"]

if AUTOGLUON_AVAILABLE:
    # AutoGluon evaluation
    y_pred_labels = predictor.predict(test_df.drop(columns=["Mental_Health_Status"]))
    y_pred_class = np.array([CLASS_MAP[l] for l in y_pred_labels])

    print(f"\n{BOLD}AutoGluon Leaderboard:{RESET}")
    lb = predictor.leaderboard(test_df, silent=True)
    print(lb.to_string())

    # Try to extract best tree model for SHAP
    try:
        best_model_name = None
        for name in lb["model"].values:
            lower = name.lower()
            if any(k in lower for k in ["xgboost", "lightgbm", "catboost", "gbm"]):
                best_model_name = name
                break
        if best_model_name:
            inner_model = predictor._trainer.load_model(best_model_name)
            print(ok(f"Extracted tree model '{best_model_name}' for SHAP"))
        else:
            print(warn("No tree model found in leaderboard; skipping SHAP"))
    except Exception as e:
        print(warn(f"Could not extract inner model for SHAP: {e}"))
        best_model_name = None

else:
    # Manual ensemble evaluation
    y_pred_class = meta_clf.predict(meta_test)

print(f"\n{BOLD}Classification Metrics (Test Set — {X_test_stacked.shape[0]} samples):{RESET}")
print(f"Accuracy : {accuracy_score(y_class_test, y_pred_class):.4f}")

try:
    # Need probabilities for ROC-AUC
    if AUTOGLUON_AVAILABLE:
        y_proba = predictor.predict_proba(test_df.drop(columns=["Mental_Health_Status"]))
        roc = roc_auc_score(y_class_test, y_proba.values, multi_class="ovr")
    else:
        y_proba = meta_clf.predict_proba(meta_test)
        roc = roc_auc_score(y_class_test, y_proba, multi_class="ovr")
    print(f"ROC-AUC  : {roc:.4f}")
except Exception as e:
    print(warn(f"ROC-AUC could not be computed: {e}"))

print(f"\n{classification_report(y_class_test, y_pred_class, target_names=CLASS_NAMES)}")

# ── Confusion Matrix ─────────────────────────────────────────────────────────
cm = confusion_matrix(y_class_test, y_pred_class)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix — Sequential Stacking Ensemble")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_stacking.png", dpi=300)
print(ok("Saved confusion_matrix_stacking.png"))

# ── SHAP Explainability ──────────────────────────────────────────────────────
print(hdr("Stage 4b · SHAP Explainability"))

import shap

# Use the best available tree model for SHAP
if not AUTOGLUON_AVAILABLE:
    # Use the XGBoost base model from Layer 1 (trained on stacked features)
    shap_model = base_models["XGBoost"]
    shap_features = STACKED_FEATURE_NAMES
    X_shap = X_test_stacked
    print(sub("Using Layer 1 XGBoost for TreeSHAP"))
else:
    # Use AutoGluon's best tree model if available
    if best_model_name:
        shap_model = inner_model
    else:
        print(warn("No suitable tree model for SHAP, skipping"))
        shap_model = None
    shap_features = STACKED_FEATURE_NAMES
    X_shap = X_test_stacked

if shap_model is not None:
    X_shap_df = pd.DataFrame(X_shap, columns=shap_features)

    explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(X_shap_df)

    # Compute mean absolute SHAP per feature
    if isinstance(shap_values, list):
        abs_shap = np.zeros(len(shap_features))
        for sv in shap_values:
            abs_shap += np.abs(sv).mean(axis=0)
        abs_shap /= len(shap_values)
    elif len(shap_values.shape) == 3:
        abs_shap = np.abs(shap_values).mean(axis=0).mean(axis=1)
    else:
        abs_shap = np.abs(shap_values).mean(axis=0)

    # ── Modality Aggregation ─────────────────────────────────────────────────
    tabular_imp = np.sum(abs_shap[:18])
    speech_imp  = np.sum(abs_shap[18:50])    # Speech PCA features (18–49)
    facial_imp  = np.sum(abs_shap[50:82])    # Facial PCA features (50–81)

    # Also count the 3 severity predictions separately or include in tabular
    severity_imp = np.sum(abs_shap[82:85]) if len(abs_shap) > 82 else 0
    tabular_imp += severity_imp   # severity predictions are tabular-derived

    total_imp = tabular_imp + speech_imp + facial_imp
    pcts = [
        (tabular_imp / total_imp) * 100,
        (speech_imp / total_imp) * 100,
        (facial_imp / total_imp) * 100,
    ]
    modalities = ["Tabular/Physiological\n(+Severity Predictions)", "Acoustic/Speech\n(PCA-32)", "Visual/Facial\n(PCA-32)"]

    print(f"\n{BOLD}Modality Contributions:{RESET}")
    for mod, pct in zip(modalities, pcts):
        print(f"  {mod.split(chr(10))[0]:35s} {pct:6.2f}%")

    # ── Plot 1: Modality Bar Chart ───────────────────────────────────────────
    plt.figure(figsize=(9, 5))
    bars = plt.barh(modalities, pcts, color=["#2196F3", "#FF9800", "#4CAF50"], edgecolor="white")
    for bar, pct in zip(bars, pcts):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{pct:.1f}%", va="center", fontweight="bold")
    plt.xlabel("Percentage Contribution (%)")
    plt.title("Modality-Level Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig("modality_bar_chart_stacking.png", dpi=300)
    print(ok("Saved modality_bar_chart_stacking.png"))

    # ── Plot 2: Top 15 Feature Summary ───────────────────────────────────────
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_shap_df, plot_type="bar",
        max_display=15, show=False,
        class_names=CLASS_NAMES
    )
    plt.title("Top 15 Individual Features (SHAP Value)")
    plt.tight_layout()
    plt.savefig("top_15_shap_stacking.png", dpi=300)
    print(ok("Saved top_15_shap_stacking.png"))

print(f"\n{BOLD}{GREEN}{'═'*62}")
print("  Sequential Stacking Pipeline Complete.")
print(f"{'═'*62}{RESET}\n")
