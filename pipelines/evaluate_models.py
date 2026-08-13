import numpy as np
import pandas as pd
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, accuracy_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, explained_variance_score,
    confusion_matrix, precision_recall_fscore_support
)
from autogluon.tabular import TabularPredictor

warnings.filterwarnings("ignore")

# 1. Recreate Data & Splits (Exact same seed=42)
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

num_df = pd.read_csv("numerical_data.csv")
speech_emb = np.load("speech_embeddings.npy")
facial_emb = np.load("facial_embeddings.npy")
speech_meta = pd.read_csv("speech_metadata.csv")
facial_meta = pd.read_csv("facial_metadata.csv")

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

speech_pools = {}
facial_pools = {}
for mhs in CLASS_MAP.keys():
    s_idx = speech_meta.index[speech_meta["Emotion_Label"].isin(SPEECH_MAP[mhs])].tolist()
    f_idx = facial_meta.index[facial_meta["Class_Label"].isin(FACIAL_MAP[mhs])].tolist()
    speech_pools[mhs] = s_idx
    facial_pools[mhs] = f_idx

np.random.seed(42)
N_COMPLETE = num_df.shape[0]

X_num_raw = num_df[NUMERICAL_FEATURES].values.astype(np.float32)
y_class = num_df["Mental_Health_Status"].values
y_class_int = np.array([CLASS_MAP[c] for c in y_class], dtype=np.int64)
y_reg = num_df[REGRESSION_TARGETS].values.astype(np.float32)

X_speech = np.zeros((N_COMPLETE, speech_emb.shape[1]), dtype=np.float32)
X_facial = np.zeros((N_COMPLETE, facial_emb.shape[1]), dtype=np.float32)

for i, status in enumerate(y_class):
    X_speech[i] = speech_emb[np.random.choice(speech_pools[status])]
    X_facial[i] = facial_emb[np.random.choice(facial_pools[status])]

from sklearn.model_selection import train_test_split
(_, X_num_test, _, X_speech_test, _, X_facial_test, _, y_class_test, _, y_class_int_test, _, y_reg_test) = train_test_split(
    X_num_raw, X_speech, X_facial, y_class, y_class_int, y_reg,
    test_size=0.20, random_state=42, stratify=y_class_int
)

# 2. Load Preprocessors and Transform Test Set
scaler = joblib.load("standard_scaler_semantic.joblib")
pca_speech = joblib.load("pca_speech_semantic.joblib")
pca_facial = joblib.load("pca_facial_semantic.joblib")

X_num_test_sc = scaler.transform(X_num_test).astype(np.float32)
X_speech_test_pca = pca_speech.transform(X_speech_test).astype(np.float32)
X_facial_test_pca = pca_facial.transform(X_facial_test).astype(np.float32)

X_test_fused = np.hstack([X_num_test_sc, X_speech_test_pca, X_facial_test_pca])

# 3. Evaluate Head B (Regression)
head_B = joblib.load("head_B_regressor_semantic.joblib")
test_severity = head_B.predict(X_test_fused)

mae = mean_absolute_error(y_reg_test, test_severity)
mse = mean_squared_error(y_reg_test, test_severity)
rmse = np.sqrt(mse)
r2 = r2_score(y_reg_test, test_severity)
evs = explained_variance_score(y_reg_test, test_severity)

# 4. Evaluate Head A (Classification)
FUSED_NAMES = (
    NUMERICAL_FEATURES
    + [f"Speech_PCA_{i}" for i in range(PCA_N)]
    + [f"Facial_PCA_{i}" for i in range(PCA_N)]
)
STACKED_NAMES = FUSED_NAMES + ["pred_depression", "pred_anxiety", "pred_stress"]

X_test_stacked = np.hstack([X_test_fused, test_severity])
test_df = pd.DataFrame(X_test_stacked, columns=STACKED_NAMES)
test_df["Mental_Health_Status"] = y_class_test

predictor = TabularPredictor.load("autogluon_models_semantic")
test_features = test_df.drop(columns=["Mental_Health_Status"])
y_pred_labels = predictor.predict(test_features)
y_pred_int = np.array([CLASS_MAP[l] for l in y_pred_labels])
y_pred_proba = predictor.predict_proba(test_features)
proba_ordered = y_pred_proba[[INV_CLASS_MAP[i] for i in range(4)]].values

acc = accuracy_score(y_class_int_test, y_pred_int)
roc_auc = roc_auc_score(y_class_int_test, proba_ordered, multi_class="ovr")
precision, recall, f1, support = precision_recall_fscore_support(y_class_int_test, y_pred_int)
macro_f1 = np.mean(f1)
weighted_f1 = np.average(f1, weights=support)

# Generate Confusion Matrix
cm = confusion_matrix(y_class_int_test, y_pred_int)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix — Final Aligned Stacking Pipeline")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("final_aligned_confusion_matrix.png", dpi=300)

# Print Final Report
report = f"""
================================================================================
FINAL HACKATHON EVALUATION REPORT
================================================================================

## OBJECTIVE 2: HEAD B (Regression / Severity Scores)
--------------------------------------------------------------------------------
Model            : MultiOutputRegressor(XGBRegressor)
Target Variables : Depression, Anxiety, Stress (Averaged Metrics)

• MAE (Mean Absolute Error)     : {mae:.4f}
• MSE (Mean Squared Error)      : {mse:.4f}
• RMSE (Root Mean Squared Error): {rmse:.4f}
• R² Score                      : {r2:.4f}
• Explained Variance Score      : {evs:.4f}

## OBJECTIVE 1: HEAD A (Classification / Mental Health Status)
--------------------------------------------------------------------------------
Model            : AutoGluon Multi-Layer Stack Ensemble
Target Variable  : Mental Health Status (4 Classes)

• Overall Accuracy              : {acc:.4%}
• ROC-AUC (OvR)                 : {roc_auc:.4f}
• Macro F1-Score                : {macro_f1:.4f}
• Weighted F1-Score             : {weighted_f1:.4f}

CLASS-WISE PERFORMANCE REPORT:
--------------------------------------------------------------------------------
                     Precision    Recall / Sens.   F1-Score   Support
"""
for i, name in enumerate(CLASS_NAMES):
    report += f"  {name:18} {precision[i]:<12.4f} {recall[i]:<16.4f} {f1[i]:<10.4f} {support[i]:<7}\n"

report += f"""
================================================================================
CONFUSION MATRIX SAVED TO: final_aligned_confusion_matrix.png
================================================================================
"""
print(report)
