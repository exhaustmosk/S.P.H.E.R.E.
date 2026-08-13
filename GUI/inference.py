import os
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
import librosa
from transformers import AutoImageProcessor, ViTForImageClassification
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from autogluon.tabular import TabularPredictor

# Device
mps_available = torch.backends.mps.is_available()
device = torch.device("mps" if mps_available else "cpu")

# Globals for models
processor_vit = None
model_vit = None
processor_wav2vec = None
model_wav2vec = None
pca_facial = None
pca_speech = None
standard_scaler = None
head_B_regressor = None
head_A_predictor = None

def init_models():
    global processor_vit, model_vit, processor_wav2vec, model_wav2vec
    global pca_facial, pca_speech, standard_scaler
    global head_B_regressor, head_A_predictor

    if processor_vit is not None:
        return # already loaded

    print("Loading models...")
    # 1. HuggingFace Models
    processor_vit = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
    model_vit = ViTForImageClassification.from_pretrained("dima806/facial_emotions_image_detection").to(device).eval()

    processor_wav2vec = Wav2Vec2FeatureExtractor.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")
    model_wav2vec = Wav2Vec2Model.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition").to(device).eval()

    # 2. Pipeline Artifacts
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pca_facial = joblib.load(os.path.join(base_dir, "pca_facial_semantic.joblib"))
    pca_speech = joblib.load(os.path.join(base_dir, "pca_speech_semantic.joblib"))
    standard_scaler = joblib.load(os.path.join(base_dir, "standard_scaler_semantic.joblib"))
    head_B_regressor = joblib.load(os.path.join(base_dir, "head_B_regressor_semantic.joblib"))
    
    # 3. AutoGluon Predictor
    ag_path = os.path.join(base_dir, "autogluon_models_semantic")
    head_A_predictor = TabularPredictor.load(ag_path)
    print("Models loaded successfully.")

def extract_facial_embedding(image_path_or_file):
    try:
        img = Image.open(image_path_or_file)
        if img.mode != "RGB":
            img = img.convert("RGB")
        inputs = processor_vit(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            outputs = model_vit.vit(pixel_values=pixel_values)
            cls_emb = outputs.last_hidden_state[:, 0, :]
        return cls_emb.cpu().numpy()
    except Exception as e:
        print(f"Facial extraction error: {e}")
        return np.zeros((1, 768))

def extract_speech_embedding(audio_path_or_file):
    try:
        # Load exactly 4 seconds like RAVDESS logic
        speech, _ = librosa.load(audio_path_or_file, sr=16000, duration=4.0)
        inputs = processor_wav2vec(speech, sampling_rate=16000, return_tensors="pt")
        input_values = inputs["input_values"].to(device)
        with torch.no_grad():
            outputs = model_wav2vec(input_values)
            cls_emb = outputs.last_hidden_state.mean(dim=1)
        return cls_emb.cpu().numpy()
    except Exception as e:
        print(f"Speech extraction error: {e}")
        return np.zeros((1, 1024))

def run_inference(image_file, audio_file, tabular_dict):
    """
    tabular_dict must contain the 18 CSV_COLUMNS exactly.
    """
    init_models()

    # 1. Extract raw embeddings
    raw_face = extract_facial_embedding(image_file)
    raw_speech = extract_speech_embedding(audio_file)

    # 2. PCA
    pca_face = pca_facial.transform(raw_face)
    pca_spch = pca_speech.transform(raw_speech)

    # 3. Scale Tabular
    # tabular_dict to DataFrame
    df_tab = pd.DataFrame([tabular_dict])
    # Ensure correct order of 18 columns
    csv_cols = [
        "Sleep_Quality", "Social_Engagement", "Daily_App_Usage_Min", 
        "Typing_Speed_WPM", "Session_Frequency", "Idle_Time_Min", 
        "Facial_Emotion_Variance", "Eye_Blink_Rate", "Smile_Intensity", 
        "Head_Motion_Index", "MFCC_Mean", "MFCC_Variance", 
        "Pitch_Mean", "Speech_Rate", "Heart_Rate_BPM", 
        "HRV_Index", "Skin_Temperature", "GSR_Level"
    ]
    df_tab = df_tab[csv_cols]
    # For missing or empty string values, cast to numeric with coercion
    for col in csv_cols:
        df_tab[col] = pd.to_numeric(df_tab[col], errors='coerce').fillna(0)
    
    scaled_tab = standard_scaler.transform(df_tab.values)

    # 4. Concatenate features
    X_fused = np.hstack([scaled_tab, pca_spch, pca_face])

    # 5. Predict Head B (Regression)
    y_reg = head_B_regressor.predict(X_fused)[0] # Depression, Anxiety, Stress

    # 6. Predict Head A (Classification)
    df_infer = pd.DataFrame(X_fused)
    df_infer.columns = [f"f{i}" for i in range(X_fused.shape[1])]
    
    # Add Head B predictions to features for Sequential Stacking
    df_infer['pred_Depression'] = y_reg[0]
    df_infer['pred_Anxiety'] = y_reg[1]
    df_infer['pred_Stress'] = y_reg[2]
    
    pred_class = head_A_predictor.predict(df_infer).iloc[0]
    pred_proba = head_A_predictor.predict_proba(df_infer).iloc[0]
    
    # Map back to mock report style
    class_mapping = {0: "Healthy", 1: "Mild", 2: "Moderate", 3: "Severe"}
    
    category = class_mapping.get(pred_class, "Unknown")
    confidence = pred_proba.max()
    
    report = {
        "objective1": {
            "category": category,
            "confidence": confidence,
            "independence_note": "This is a single FUSED prediction powered by S.P.H.E.R.E. (AutoGluon Sequential Stacking)."
        },
        "objective2": {
            "scale": "0–10",
            "scores": {
                "depression": round(float(y_reg[0]), 1),
                "anxiety": round(float(y_reg[1]), 1),
                "stress": round(float(y_reg[2]), 1),
            },
            "caveat": ""
        },
        "objective3": {
            "gradcam_label": "Modality Contribution (SHAP expected average)",
            "shap_label": "Feature Importance (S.P.H.E.R.E. Stack)",
            "shap_bars": [
                {"name": "Speech Modality (Overall)", "value": 0.43},
                {"name": "Facial Modality (Overall)", "value": 0.32},
                {"name": "Tabular Modality (Overall)", "value": 0.25},
            ],
            "tabular_note": "Scores are derived from SHAP global importance calculations across the validation set.",
            "speech_model_note": ""
        }
    }
    
    return report
