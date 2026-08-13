"""
merge_multimodal_dataset.py
────────────────────────────────────────────────────────────────────────────
Data Integration Pipeline
  Inputs:
    - numerical_data.csv (Target labels + 18 physiological/behavioral features)
    - speech_embeddings.npy (N, 1024)
    - facial_embeddings.npy (M, 768)
  Output:
    - X_train.npy, X_val.npy, X_test.npy
    - y_train_class.npy, y_val_class.npy, y_test_class.npy
    - y_train_reg.npy, y_val_reg.npy, y_test_reg.npy
────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

# ── colour helpers ────────────────────────────────────────────────────────────
RESET = "\033[0m"; BOLD = "\033[1m"; GREEN = "\033[92m"
YELLOW = "\033[93m"; RED = "\033[91m"; CYAN = "\033[96m"; MAGENTA = "\033[95m"
def ok(m):   return f"{GREEN}✔  {m}{RESET}"
def warn(m): return f"{YELLOW}⚠  {m}{RESET}"
def err(m):  return f"{RED}✘  {m}{RESET}"
def hdr(m):  return f"\n{BOLD}{CYAN}{'─'*62}\n  {m}\n{'─'*62}{RESET}"
def sub(m):  return f"{MAGENTA}  ▸ {m}{RESET}"

# ── 1. CONFIGURATION ─────────────────────────────────────────────────────────
NUMERICAL_CSV    = Path("numerical_data.csv")
SPEECH_EMB_PATH  = Path("speech_embeddings.npy")
FACIAL_EMB_PATH  = Path("facial_embeddings.npy")

NUMERICAL_FEATURES = [
    "Sleep_Quality", "Social_Engagement", "Daily_App_Usage_Min", 
    "Typing_Speed_WPM", "Session_Frequency", "Idle_Time_Min", 
    "Facial_Emotion_Variance", "Eye_Blink_Rate", "Smile_Intensity", 
    "Head_Motion_Index", "MFCC_Mean", "MFCC_Variance", "Pitch_Mean", 
    "Speech_Rate", "Heart_Rate_BPM", "HRV_Index", "Skin_Temperature", 
    "GSR_Level"
]

CLASS_MAP = {
    "Healthy": 0,
    "Mild_Stress": 1,
    "Moderate_Stress": 2,
    "Severe_Stress": 3
}

REGRESSION_TARGETS = ["Depression_Score", "Anxiety_Score", "Stress_Score"]

print(hdr("1 · Loading Data"))
try:
    num_df = pd.read_csv(NUMERICAL_CSV)
    print(ok(f"Loaded {NUMERICAL_CSV.name} ({num_df.shape[0]} rows)"))
    
    speech_emb = np.load(SPEECH_EMB_PATH)
    print(ok(f"Loaded {SPEECH_EMB_PATH.name} {speech_emb.shape}"))
    
    facial_emb = np.load(FACIAL_EMB_PATH)
    print(ok(f"Loaded {FACIAL_EMB_PATH.name} {facial_emb.shape}"))
except Exception as e:
    print(err(f"Failed to load data: {e}"))
    sys.exit(1)

# ── 2. DATA ALIGNMENT & PADDING ──────────────────────────────────────────────
print(hdr("2 · Aligning & Padding Modalities"))

# Using index as the Sample ID. We need to find the max length across modalities.
num_samples = max(num_df.shape[0], speech_emb.shape[0], facial_emb.shape[0])
print(sub(f"Total Unique Samples (max index): {num_samples}"))

# Prepare padded numerical features
X_num = np.zeros((num_samples, len(NUMERICAL_FEATURES)), dtype=np.float32)
available_num = min(num_samples, num_df.shape[0])
X_num[:available_num, :] = num_df[NUMERICAL_FEATURES].values[:available_num]

# Prepare padded speech features
X_speech = np.zeros((num_samples, speech_emb.shape[1]), dtype=np.float32)
available_speech = min(num_samples, speech_emb.shape[0])
X_speech[:available_speech, :] = speech_emb[:available_speech, :]

# Prepare padded facial features
X_facial = np.zeros((num_samples, facial_emb.shape[1]), dtype=np.float32)
available_facial = min(num_samples, facial_emb.shape[0])
X_facial[:available_facial, :] = facial_emb[:available_facial, :]

# Fuse all features
X_fused = np.concatenate([X_num, X_speech, X_facial], axis=1)
print(ok(f"Fused Feature Matrix X_fused shape: {X_fused.shape}"))

# Extract Targets
print(hdr("3 · Extracting Targets"))

# Classification Targets
y_class = np.zeros(num_samples, dtype=np.int64) # Default to 0 (Healthy) if missing
if "Mental_Health_Status" in num_df.columns:
    mapped = num_df["Mental_Health_Status"].map(CLASS_MAP).fillna(0).astype(int).values
    y_class[:available_num] = mapped[:available_num]
print(ok(f"Classification Target y_class shape: {y_class.shape}"))

# Regression Targets
y_reg = np.zeros((num_samples, len(REGRESSION_TARGETS)), dtype=np.float32)
y_reg[:available_num, :] = num_df[REGRESSION_TARGETS].values[:available_num]
print(ok(f"Regression Target y_reg shape: {y_reg.shape}"))

# ── 4. SPLITTING DATA ────────────────────────────────────────────────────────
print(hdr("4 · Stratified Train/Val/Test Split (80/10/10)"))

# We stratify based on y_class
X_temp, X_test, y_class_temp, y_test_class, y_reg_temp, y_test_reg = train_test_split(
    X_fused, y_class, y_reg, test_size=0.10, random_state=42, stratify=y_class
)

# 80/10 split from the remaining 90% (80/90 = 0.8888) -> test_size = 0.1111 (10% of total)
X_train, X_val, y_train_class, y_val_class, y_train_reg, y_val_reg = train_test_split(
    X_temp, y_class_temp, y_reg_temp, test_size=1/9, random_state=42, stratify=y_class_temp
)

print(sub(f"Train split : {X_train.shape[0]} samples ({X_train.shape[0]/num_samples*100:.1f}%)"))
print(sub(f"Val split   : {X_val.shape[0]} samples ({X_val.shape[0]/num_samples*100:.1f}%)"))
print(sub(f"Test split  : {X_test.shape[0]} samples ({X_test.shape[0]/num_samples*100:.1f}%)"))

# ── 5. SAVING ARTIFACTS ──────────────────────────────────────────────────────
print(hdr("5 · Saving Final Datasets"))

OUTPUT_DIR = Path("dataset_splits")
OUTPUT_DIR.mkdir(exist_ok=True)

np.save(OUTPUT_DIR / "X_train.npy", X_train)
np.save(OUTPUT_DIR / "X_val.npy", X_val)
np.save(OUTPUT_DIR / "X_test.npy", X_test)

np.save(OUTPUT_DIR / "y_train_class.npy", y_train_class)
np.save(OUTPUT_DIR / "y_val_class.npy", y_val_class)
np.save(OUTPUT_DIR / "y_test_class.npy", y_test_class)

np.save(OUTPUT_DIR / "y_train_reg.npy", y_train_reg)
np.save(OUTPUT_DIR / "y_val_reg.npy", y_val_reg)
np.save(OUTPUT_DIR / "y_test_reg.npy", y_test_reg)

total_size_mb = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.npy")) / (1024 * 1024)
print(ok(f"Saved 9 arrays to {OUTPUT_DIR}/ (Total size: {total_size_mb:.1f} MB)"))

print(f"\n{BOLD}{GREEN}{'═'*62}")
print("  Data Integration Complete.")
print(f"{'═'*62}{RESET}\n")
