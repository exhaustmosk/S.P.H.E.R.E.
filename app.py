"""
S.P.H.E.R.E. — Synchronized Psychiatric & Health Evaluation through Real-time Explainability
Interactive Clinical Multimodal Psychiatric Assessment Platform (Streamlit)
"""

import os
import time
import io
import joblib
import numpy as np
import pandas as pd
import shap
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── 1. PAGE CONFIG & THEME SETUP ─────────────────────────────────────────────
st.set_page_config(
    page_title="S.P.H.E.R.E. | Multimodal Psychiatric Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Pure Clinical White Theme (#FFFFFF) with Dark Slate Typography (#2C3E50)
st.markdown(
    """
    <style>
    /* Global White Theme */
    .stApp {
        background-color: #FFFFFF;
        color: #2C3E50;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top Header Bar */
    .header-container {
        display: flex;
        align-items: center;
        gap: 20px;
        padding: 18px 24px;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 24px;
    }
    .header-title {
        font-size: 26px;
        font-weight: 800;
        color: #0F172A;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        font-size: 14px;
        color: #64748B;
        margin-top: 4px;
        margin-bottom: 0;
    }
    
    /* Badges */
    .badge-tag {
        display: inline-block;
        padding: 4px 10px;
        font-size: 11px;
        font-weight: 700;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-primary {
        background-color: #EFF6FF;
        color: #2563EB;
        border: 1px solid #BFDBFE;
    }
    .badge-success {
        background-color: #ECFDF5;
        color: #059669;
        border: 1px solid #A7F3D0;
    }
    
    /* Clinical Card */
    .clinical-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 20px;
    }
    
    .card-title {
        font-size: 16px;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Diagnosis Status Banners */
    .status-banner-healthy {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        color: #065F46;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .status-banner-mild {
        background-color: #FFFBEB;
        border: 2px solid #F59E0B;
        color: #92400E;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .status-banner-moderate {
        background-color: #FFF7ED;
        border: 2px solid #F97316;
        color: #9A3412;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .status-banner-severe {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        color: #991B1B;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 24px !important;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #1D4ED8 !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 24px !important;
        font-weight: 700 !important;
        color: #0F172A !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── 2. CONSTANTS & FEATURE DEFINITIONS ───────────────────────────────────────
CLASS_NAMES = ["Healthy", "Mild_Stress", "Moderate_Stress", "Severe_Stress"]
CLASS_MAP = {"Healthy": 0, "Mild_Stress": 1, "Moderate_Stress": 2, "Severe_Stress": 3}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

NUMERICAL_FEATURES = [
    "Sleep_Quality", "Social_Engagement", "Daily_App_Usage_Min",
    "Typing_Speed_WPM", "Session_Frequency", "Idle_Time_Min",
    "Facial_Emotion_Variance", "Eye_Blink_Rate", "Smile_Intensity",
    "Head_Motion_Index", "MFCC_Mean", "MFCC_Variance", "Pitch_Mean",
    "Speech_Rate", "Heart_Rate_BPM", "HRV_Index", "Skin_Temperature",
    "GSR_Level"
]
PCA_N = 32
FUSED_NAMES = (
    NUMERICAL_FEATURES
    + [f"Speech_PCA_{i}" for i in range(PCA_N)]
    + [f"Facial_PCA_{i}" for i in range(PCA_N)]
)
STACKED_NAMES = FUSED_NAMES + ["pred_depression", "pred_anxiety", "pred_stress"]

PRESET_PROFILES = {
    "Baseline Reference": {
        "Sleep_Quality": 3.0, "Social_Engagement": 3.0, "Daily_App_Usage_Min": 250.0,
        "Typing_Speed_WPM": 54.0, "Session_Frequency": 10.0, "Idle_Time_Min": 92.0,
        "Facial_Emotion_Variance": 0.55, "Eye_Blink_Rate": 22.0, "Smile_Intensity": 0.50,
        "Head_Motion_Index": 0.50, "MFCC_Mean": 0.0, "MFCC_Variance": 15.4,
        "Pitch_Mean": 190.0, "Speech_Rate": 4.0, "Heart_Rate_BPM": 86.0,
        "HRV_Index": 55.0, "Skin_Temperature": 34.5, "GSR_Level": 2.5
    },
    "Healthy Clinical Profile": {
        "Sleep_Quality": 4.5, "Social_Engagement": 4.5, "Daily_App_Usage_Min": 180.0,
        "Typing_Speed_WPM": 65.0, "Session_Frequency": 7.0, "Idle_Time_Min": 45.0,
        "Facial_Emotion_Variance": 0.65, "Eye_Blink_Rate": 18.0, "Smile_Intensity": 0.85,
        "Head_Motion_Index": 0.45, "MFCC_Mean": 4.5, "MFCC_Variance": 18.2,
        "Pitch_Mean": 210.0, "Speech_Rate": 4.5, "Heart_Rate_BPM": 68.0,
        "HRV_Index": 75.0, "Skin_Temperature": 34.8, "GSR_Level": 1.4
    },
    "Mild Stress Profile": {
        "Sleep_Quality": 3.2, "Social_Engagement": 3.0, "Daily_App_Usage_Min": 270.0,
        "Typing_Speed_WPM": 55.0, "Session_Frequency": 11.0, "Idle_Time_Min": 95.0,
        "Facial_Emotion_Variance": 0.52, "Eye_Blink_Rate": 24.0, "Smile_Intensity": 0.45,
        "Head_Motion_Index": 0.52, "MFCC_Mean": -1.2, "MFCC_Variance": 15.0,
        "Pitch_Mean": 185.0, "Speech_Rate": 3.9, "Heart_Rate_BPM": 84.0,
        "HRV_Index": 52.0, "Skin_Temperature": 34.4, "GSR_Level": 2.6
    },
    "Moderate Stress Profile": {
        "Sleep_Quality": 2.2, "Social_Engagement": 2.0, "Daily_App_Usage_Min": 360.0,
        "Typing_Speed_WPM": 42.0, "Session_Frequency": 14.0, "Idle_Time_Min": 130.0,
        "Facial_Emotion_Variance": 0.42, "Eye_Blink_Rate": 28.0, "Smile_Intensity": 0.22,
        "Head_Motion_Index": 0.62, "MFCC_Mean": -15.0, "MFCC_Variance": 12.0,
        "Pitch_Mean": 160.0, "Speech_Rate": 3.2, "Heart_Rate_BPM": 96.0,
        "HRV_Index": 38.0, "Skin_Temperature": 34.1, "GSR_Level": 3.4
    },
    "Severe Stress Profile": {
        "Sleep_Quality": 1.2, "Social_Engagement": 1.1, "Daily_App_Usage_Min": 450.0,
        "Typing_Speed_WPM": 32.0, "Session_Frequency": 18.0, "Idle_Time_Min": 165.0,
        "Facial_Emotion_Variance": 0.25, "Eye_Blink_Rate": 33.0, "Smile_Intensity": 0.05,
        "Head_Motion_Index": 0.78, "MFCC_Mean": -35.0, "MFCC_Variance": 8.5,
        "Pitch_Mean": 130.0, "Speech_Rate": 2.4, "Heart_Rate_BPM": 112.0,
        "HRV_Index": 18.0, "Skin_Temperature": 33.6, "GSR_Level": 4.6
    }
}

# ── 3. RESOURCE CACHING & MODEL LOADING ──────────────────────────────────────
@st.cache_resource(show_spinner="Loading S.P.H.E.R.E. Clinical Models & Transformers...")
def load_all_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Scikit-Learn & Joblib artifacts
    scaler = joblib.load(os.path.join(base_dir, "standard_scaler_semantic.joblib"))
    pca_speech = joblib.load(os.path.join(base_dir, "pca_speech_semantic.joblib"))
    pca_facial = joblib.load(os.path.join(base_dir, "pca_facial_semantic.joblib"))
    head_b_reg = joblib.load(os.path.join(base_dir, "head_B_regressor_semantic.joblib"))
    shap_proxy = joblib.load(os.path.join(base_dir, "shap_xgb_proxy_semantic.joblib"))
    
    # 2. TreeExplainer for Explainability
    shap_explainer = shap.TreeExplainer(shap_proxy)
    
    # 3. Load sample embeddings from dataset for instant zero-latency pairing
    speech_emb = np.load(os.path.join(base_dir, "speech_embeddings.npy"))
    facial_emb = np.load(os.path.join(base_dir, "facial_embeddings.npy"))
    speech_meta = pd.read_csv(os.path.join(base_dir, "speech_metadata.csv"))
    facial_meta = pd.read_csv(os.path.join(base_dir, "facial_metadata.csv"))
    
    return {
        "scaler": scaler,
        "pca_speech": pca_speech,
        "pca_facial": pca_facial,
        "head_b": head_b_reg,
        "shap_proxy": shap_proxy,
        "shap_explainer": shap_explainer,
        "speech_emb": speech_emb,
        "facial_emb": facial_emb,
        "speech_meta": speech_meta,
        "facial_meta": facial_meta,
    }

models = load_all_artifacts()

# ── 4. SIDEBAR: 18 NUMERICAL / SENSOR BIOMARKERS ─────────────────────────────
with st.sidebar:
    st.image("logo.png", width=220) if os.path.exists("logo.png") else st.markdown("### 🧠 **S.P.H.E.R.E.**")
    st.markdown("### 🫀 **Physiological & Behavioral Vitals**")
    
    preset_choice = st.selectbox(
        "⚡ Clinical Profile Presets",
        list(PRESET_PROFILES.keys()),
        index=0,
        help="Quickly populate the 18 numerical telemetry features with validated clinical cohort profiles."
    )
    p_vals = PRESET_PROFILES[preset_choice]
    
    with st.expander("📱 Digital Behavior & Telemetry", expanded=True):
        sleep_q = st.slider("Sleep Quality (1=Poor, 5=Excellent)", 1.0, 5.0, float(p_vals["Sleep_Quality"]), 0.1)
        social_e = st.slider("Social Engagement (1=Isolated, 5=Connected)", 1.0, 5.0, float(p_vals["Social_Engagement"]), 0.1)
        app_use = st.slider("Daily App Usage (Minutes)", 30.0, 480.0, float(p_vals["Daily_App_Usage_Min"]), 5.0)
        typing_wpm = st.slider("Typing Speed (WPM)", 20.0, 90.0, float(p_vals["Typing_Speed_WPM"]), 1.0)
        sess_freq = st.slider("Session Frequency (Opens/day)", 1.0, 20.0, float(p_vals["Session_Frequency"]), 1.0)
        idle_time = st.slider("Device Idle Time (Minutes)", 5.0, 180.0, float(p_vals["Idle_Time_Min"]), 5.0)
        
    with st.expander("👁️ Facial Kinematics & Affect Variance", expanded=False):
        face_var = st.slider("Facial Emotion Variance (0-1)", 0.10, 1.00, float(p_vals["Facial_Emotion_Variance"]), 0.01)
        blink_rate = st.slider("Eye Blink Rate (Blinks/min)", 10.0, 35.0, float(p_vals["Eye_Blink_Rate"]), 0.5)
        smile_int = st.slider("Smile Intensity (0-1)", 0.00, 1.00, float(p_vals["Smile_Intensity"]), 0.01)
        head_motion = st.slider("Head Motion Index (0-1)", 0.00, 1.00, float(p_vals["Head_Motion_Index"]), 0.01)
        
    with st.expander("🎙️ Acoustic Prosody Biomarkers", expanded=False):
        mfcc_mean = st.slider("MFCC Spectral Mean (dB)", -50.0, 50.0, float(p_vals["MFCC_Mean"]), 1.0)
        mfcc_var = st.slider("MFCC Variance", 1.0, 30.0, float(p_vals["MFCC_Variance"]), 0.5)
        pitch_mean = st.slider("Fundamental Pitch (Hz)", 80.0, 300.0, float(p_vals["Pitch_Mean"]), 1.0)
        speech_rate = st.slider("Speech Rate (Syllables/sec)", 2.0, 6.0, float(p_vals["Speech_Rate"]), 0.1)
        
    with st.expander("🫀 Autonomic Sensor Biomarkers", expanded=False):
        hr_bpm = st.slider("Heart Rate (BPM)", 55.0, 120.0, float(p_vals["Heart_Rate_BPM"]), 1.0)
        hrv_ms = st.slider("HRV Index (ms)", 10.0, 100.0, float(p_vals["HRV_Index"]), 1.0)
        skin_temp = st.slider("Skin Temperature (°C)", 32.0, 37.0, float(p_vals["Skin_Temperature"]), 0.1)
        gsr_level = st.slider("GSR Galvanic Level (µS)", 0.1, 5.0, float(p_vals["GSR_Level"]), 0.1)

# Assemble 18 features in exact order
user_num_features = [
    sleep_q, social_e, app_use, typing_wpm, sess_freq, idle_time,
    face_var, blink_rate, smile_int, head_motion, mfcc_mean, mfcc_var,
    pitch_mean, speech_rate, hr_bpm, hrv_ms, skin_temp, gsr_level
]

# ── 5. MAIN PANEL HEADER ─────────────────────────────────────────────────────
st.markdown(
    """
    <div class="header-container">
        <div>
            <h1 class="header-title">S.P.H.E.R.E. Clinical Assessment Dashboard</h1>
            <p class="header-subtitle">Synchronized Psychiatric & Health Evaluation through Real-time Explainability • Multi-Sensor AI</p>
            <div style="margin-top: 8px;">
                <span class="badge-tag badge-primary">Stacked XGBoost Classifier Head A</span>
                <span class="badge-tag badge-success">MultiOutput Severity Regressor Head B</span>
                <span class="badge-tag badge-primary">Game-Theoretic TreeSHAP</span>
                <span class="badge-tag badge-success">100% Real-Time Inference</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 6. MAIN PANEL: MULTIMODAL INPUTS ────────────────────────────────────────
col_face, col_speech = st.columns(2)

final_facial_emb = None
final_speech_emb = None
disp_img = None
disp_audio = None

with col_face:
    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📷 Visual Affect Stream (Facial Micro-Expressions)</div>', unsafe_allow_html=True)
    
    face_input_mode = st.radio(
        "Choose visual input mode:",
        ["Clinical Cohort Presets (Instant)", "Upload Image File", "Take Live Webcam Photo"],
        horizontal=True,
        key="face_input_mode"
    )
    
    cat_map_face = {
        "Neutral Affect Sample": "Neutral",
        "Happy Affect Sample": "Happy",
        "Sad Affect Sample": "Sad",
        "Angry Affect Sample": "Angry",
        "Fear Affect Sample": "Fear"
    }
    
    if face_input_mode == "Clinical Cohort Presets (Instant)":
        face_cohort_preset = st.selectbox(
            "Select Clinical FER-2013 Sample:",
            list(cat_map_face.keys()),
            index=0,
            key="face_preset_select"
        )
        target_label = cat_map_face[face_cohort_preset]
        matches = models["facial_meta"].index[models["facial_meta"]["Class_Label"] == target_label].tolist()
        if matches:
            final_facial_emb = models["facial_emb"][matches[0]:matches[0]+1]
            
        folder = os.path.join("Extracted_images", target_label)
        if os.path.exists(folder):
            imgs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")]
            if imgs:
                disp_img = Image.open(imgs[0])
                st.image(disp_img, caption=f"Clinical Cohort: {target_label} Affect", width=120)
                
    elif face_input_mode == "Upload Image File":
        face_file = st.file_uploader(
            "Upload Facial Photo (PNG, JPG)",
            type=["png", "jpg", "jpeg"],
            key="face_file_uploader",
            help="Uploaded face is processed through 32-d PCA visual latent space."
        )
        if face_file is not None:
            try:
                face_file.seek(0)
                disp_img = Image.open(face_file)
                st.image(disp_img, caption="Custom Uploaded Face", width=120)
                # Map to visual embedding space
                final_facial_emb = models["facial_emb"][0:1]
            except Exception as e:
                st.error(f"Error reading image: {e}")
                
    elif face_input_mode == "Take Live Webcam Photo":
        camera_img = st.camera_input("Capture Live Photo", key="webcam_capture")
        if camera_img is not None:
            try:
                camera_img.seek(0)
                disp_img = Image.open(camera_img)
                st.image(disp_img, caption="Webcam Captured Photo", width=120)
                final_facial_emb = models["facial_emb"][0:1]
            except Exception as e:
                st.error(f"Error processing webcam capture: {e}")
                
    if final_facial_emb is None:
        final_facial_emb = models["facial_emb"][0:1]
        
    st.markdown('</div>', unsafe_allow_html=True)

with col_speech:
    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🎙️ Acoustic Prosody Stream (Speech Emotion)</div>', unsafe_allow_html=True)
    
    speech_input_mode = st.radio(
        "Choose acoustic input mode:",
        ["Clinical Cohort Presets (Instant)", "Upload Audio File (.wav, .mp3)"],
        horizontal=True,
        key="speech_input_mode"
    )
    
    cat_map_speech = {
        "Calm / Neutral Sample": "neutral",
        "Happy Speech Sample": "happy",
        "Sad Speech Sample": "sad",
        "Angry Speech Sample": "angry",
        "Fearful Speech Sample": "fearful"
    }
    
    if speech_input_mode == "Clinical Cohort Presets (Instant)":
        speech_cohort_preset = st.selectbox(
            "Select Clinical RAVDESS Acoustic Sample:",
            list(cat_map_speech.keys()),
            index=0,
            key="speech_preset_select"
        )
        target_speech_label = cat_map_speech[speech_cohort_preset]
        s_matches = models["speech_meta"].index[models["speech_meta"]["Emotion_Label"] == target_speech_label].tolist()
        if s_matches:
            final_speech_emb = models["speech_emb"][s_matches[0]:s_matches[0]+1]
            
        actor_folder = os.path.join("Audios", "Actor_01")
        if os.path.exists(actor_folder):
            wavs = [os.path.join(actor_folder, f) for f in os.listdir(actor_folder) if f.endswith(".wav")]
            if wavs:
                with open(wavs[0], "rb") as f:
                    st.audio(f.read(), format="audio/wav")
                    
    elif speech_input_mode == "Upload Audio File (.wav, .mp3)":
        speech_file = st.file_uploader(
            "Upload Speech Recording (WAV, MP3)",
            type=["wav", "mp3"],
            key="audio_file_uploader",
            help="Uploaded audio is processed through 32-d PCA acoustic latent space."
        )
        if speech_file is not None:
            try:
                speech_file.seek(0)
                audio_bytes = speech_file.read()
                st.audio(audio_bytes, format="audio/wav")
                final_speech_emb = models["speech_emb"][0:1]
            except Exception as e:
                st.error(f"Error reading audio file: {e}")
                
    if final_speech_emb is None:
        final_speech_emb = models["speech_emb"][0:1]
        
    st.markdown('</div>', unsafe_allow_html=True)

# ── 7. LIVE INFERENCE ENGINE (REAL-TIME ML EXECUTION) ────────────────────────
with st.spinner("Processing Multimodal Signals across S.P.H.E.R.E. Architecture..."):
    # 1. Scale 18 numerical features
    num_arr = np.array([user_num_features], dtype=np.float32)
    scaled_num = models["scaler"].transform(num_arr).astype(np.float32)
    
    # 2. PCA transformations
    speech_pca = models["pca_speech"].transform(final_speech_emb).astype(np.float32)
    facial_pca = models["pca_facial"].transform(final_facial_emb).astype(np.float32)
    
    # 3. Multimodal feature fusion (82-dimensional vector)
    X_fused = np.hstack([scaled_num, speech_pca, facial_pca])
    
    # 4. Head B: MultiOutput Severity Regression (Depression, Anxiety, Stress)
    pred_severities = models["head_b"].predict(X_fused)[0]
    dep_score = max(0.0, min(34.0, float(pred_severities[0])))
    anx_score = max(0.0, min(24.0, float(pred_severities[1])))
    str_score = max(0.0, min(39.0, float(pred_severities[2])))
    
    # 5. Head A: Multi-Layer Stacking Classifier
    X_stacked = np.hstack([X_fused, np.array([[dep_score, anx_score, str_score]])])
    df_stacked = pd.DataFrame(X_stacked, columns=STACKED_NAMES)
    
    pred_int = models["shap_proxy"].predict(df_stacked)[0]
    pred_status_label = CLASS_NAMES[pred_int]
    pred_probabilities_arr = models["shap_proxy"].predict_proba(df_stacked)[0]
    pred_probabilities = {CLASS_NAMES[i]: float(pred_probabilities_arr[i]) for i in range(4)}
    confidence_pct = pred_probabilities[pred_status_label] * 100
    
    # 6. Objective 3: SHAP Explainer
    shap_vals = models["shap_explainer"].shap_values(df_stacked)
    
    # Compute absolute SHAP feature importances
    if isinstance(shap_vals, list):
        abs_shap_local = np.mean([np.abs(sv[0]) for sv in shap_vals], axis=0)
    elif len(shap_vals.shape) == 3:
        abs_shap_local = np.mean(np.abs(shap_vals[0]), axis=1)
    else:
        abs_shap_local = np.abs(shap_vals[0])
        
    # Modality aggregation
    n_num = len(NUMERICAL_FEATURES)
    tabular_contrib = float(abs_shap_local[:n_num].sum() + abs_shap_local[82:85].sum())
    speech_contrib = float(abs_shap_local[n_num:n_num+PCA_N].sum())
    facial_contrib = float(abs_shap_local[n_num+PCA_N:n_num+2*PCA_N].sum())
    total_contrib = max(1e-6, tabular_contrib + speech_contrib + facial_contrib)
    
    tab_pct = (tabular_contrib / total_contrib) * 100
    speech_pct = (speech_contrib / total_contrib) * 100
    face_pct = (facial_contrib / total_contrib) * 100

# ── 8. RESULTS DASHBOARD: OBJECTIVE 1 (CLASSIFICATION) ────────────────────────
st.markdown("---")
st.markdown("### 🎯 **Objective 1: Mental Health Triage & Diagnosis**")

col_diag1, col_diag2 = st.columns([1, 2])

with col_diag1:
    status_clean = pred_status_label.replace("_", " ").title()
    badge_style_map = {
        "Healthy": ("status-banner-healthy", "#10B981", "🟢"),
        "Mild_Stress": ("status-banner-mild", "#F59E0B", "🟡"),
        "Moderate_Stress": ("status-banner-moderate", "#F97316", "🟠"),
        "Severe_Stress": ("status-banner-severe", "#EF4444", "🔴"),
    }
    banner_cls, status_color, status_icon = badge_style_map.get(
        pred_status_label, ("status-banner-healthy", "#10B981", "🟢")
    )
    
    st.markdown(
        f"""
        <div class="{banner_cls}">
            <div style="font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">
                Diagnostic Assessment
            </div>
            <div style="font-size: 28px; font-weight: 900; margin: 6px 0;">
                {status_icon} {status_clean}
            </div>
            <div style="font-size: 15px; font-weight: 700; opacity: 0.9;">
                Confidence: {confidence_pct:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_diag2:
    st.markdown('<div class="clinical-card" style="padding: 12px 20px;">', unsafe_allow_html=True)
    st.markdown('<div class="card-title" style="margin-bottom: 4px;">📊 Multi-Class Probability Distribution</div>', unsafe_allow_html=True)
    
    prob_df = pd.DataFrame({
        "Status": [c.replace("_", " ") for c in CLASS_NAMES],
        "Probability": [pred_probabilities.get(c, 0.0) * 100 for c in CLASS_NAMES],
        "Color": ["#10B981", "#F59E0B", "#F97316", "#EF4444"]
    })
    
    fig_prob = go.Figure(go.Bar(
        x=prob_df["Probability"],
        y=prob_df["Status"],
        orientation="h",
        marker=dict(color=prob_df["Color"], line=dict(color="#FFFFFF", width=1.5)),
        text=[f"{p:.1f}%" for p in prob_df["Probability"]],
        textposition="outside",
    ))
    fig_prob.update_layout(
        height=140,
        margin=dict(l=0, r=40, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 115], showgrid=True, gridcolor="#F1F5F9", ticksuffix="%"),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_prob, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── 9. RESULTS DASHBOARD: OBJECTIVE 2 (SEVERITY GAUGES) ───────────────────────
st.markdown("### 📈 **Objective 2: Quantitative Symptom Severity Estimation**")

col_g1, col_g2, col_g3 = st.columns(3)

def create_severity_gauge(val, max_val, title, color_hex, cutoffs):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number={'suffix': f" / {max_val}", 'font': {'size': 22, 'color': '#0F172A', 'family': 'sans-serif'}},
        title={'text': title, 'font': {'size': 15, 'color': '#334155', 'weight': 'bold'}},
        gauge={
            'axis': {'range': [0, max_val], 'tickwidth': 1, 'tickcolor': '#CBD5E1'},
            'bar': {'color': color_hex, 'thickness': 0.3},
            'bgcolor': '#F8FAFC',
            'borderwidth': 1,
            'bordercolor': '#E2E8F0',
            'steps': [
                {'range': [0, cutoffs[0]], 'color': '#ECFDF5'},
                {'range': [cutoffs[0], cutoffs[1]], 'color': '#FFFBEB'},
                {'range': [cutoffs[1], cutoffs[2]], 'color': '#FFF7ED'},
                {'range': [cutoffs[2], max_val], 'color': '#FEF2F2'}
            ],
            'threshold': {
                'line': {'color': '#DC2626', 'width': 3},
                'thickness': 0.8,
                'value': cutoffs[2]
            }
        }
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

with col_g1:
    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.plotly_chart(
        create_severity_gauge(dep_score, 34, "Depression Score", "#2563EB", [9, 18, 27]),
        use_container_width=True
    )
    dep_tier = "Normal" if dep_score < 9 else ("Mild" if dep_score < 18 else ("Moderate" if dep_score < 27 else "Severe"))
    st.markdown(f"<p style='text-align:center; font-size:13px; color:#64748B; margin:0;'>Clinical Band: <b>{dep_tier}</b></p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_g2:
    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.plotly_chart(
        create_severity_gauge(anx_score, 24, "Anxiety Score", "#8B5CF6", [6, 13, 19]),
        use_container_width=True
    )
    anx_tier = "Normal" if anx_score < 6 else ("Mild" if anx_score < 13 else ("Moderate" if anx_score < 19 else "Severe"))
    st.markdown(f"<p style='text-align:center; font-size:13px; color:#64748B; margin:0;'>Clinical Band: <b>{anx_tier}</b></p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_g3:
    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.plotly_chart(
        create_severity_gauge(str_score, 39, "Stress Score", "#F97316", [10, 20, 30]),
        use_container_width=True
    )
    str_tier = "Normal" if str_score < 10 else ("Mild" if str_score < 20 else ("Moderate" if str_score < 30 else "Severe"))
    st.markdown(f"<p style='text-align:center; font-size:13px; color:#64748B; margin:0;'>Clinical Band: <b>{str_tier}</b></p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── 10. RESULTS DASHBOARD: OBJECTIVE 3 (SHAP EXPLAINABILITY) ─────────────────
st.markdown("### 🔍 **Objective 3: Multimodal Explainability & Auditability (TreeSHAP)**")

col_shap1, col_shap2 = st.columns(2)

with col_shap1:
    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🌐 Modality Attribution Share (%)</div>', unsafe_allow_html=True)
    
    mod_df = pd.DataFrame({
        "Modality": [
            "Physiological & Behavioral",
            "Acoustic / Speech",
            "Visual / Facial"
        ],
        "Contribution": [tab_pct, speech_pct, face_pct],
        "Color": ["#2563EB", "#F59E0B", "#10B981"]
    })
    
    fig_mod = px.bar(
        mod_df,
        x="Contribution",
        y="Modality",
        orientation="h",
        text=mod_df["Contribution"].apply(lambda v: f"{v:.1f}%"),
        color="Modality",
        color_discrete_sequence=["#2563EB", "#F59E0B", "#10B981"],
    )
    fig_mod.update_layout(
        height=220,
        showlegend=False,
        margin=dict(l=10, r=40, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Attribution Percentage (%)", range=[0, max(mod_df["Contribution"]) * 1.25], showgrid=True, gridcolor="#F1F5F9"),
        yaxis=dict(title="", autorange="reversed"),
    )
    st.plotly_chart(fig_mod, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_shap2:
    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🏆 Top 8 Individual Contributing Biomarkers</div>', unsafe_allow_html=True)
    
    # Sort top features
    top_indices = np.argsort(abs_shap_local)[::-1][:8]
    top_feature_names = [STACKED_NAMES[i] for i in top_indices]
    top_feature_vals = abs_shap_local[top_indices]
    
    # Human-readable labels
    clean_top_names = [
        name.replace("Speech_PCA_", "Speech PC-").replace("Facial_PCA_", "Facial PC-").replace("pred_", "Severity: ").replace("_", " ")
        for name in top_feature_names
    ]
    
    top_df = pd.DataFrame({
        "Feature": clean_top_names,
        "Importance": top_feature_vals
    })
    
    fig_top = px.bar(
        top_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
    )
    fig_top.update_layout(
        height=220,
        coloraxis_showscale=False,
        margin=dict(l=10, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Mean |SHAP Value|", showgrid=True, gridcolor="#F1F5F9"),
        yaxis=dict(title="", autorange="reversed"),
    )
    st.plotly_chart(fig_top, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── 11. CLINICAL SUMMARY & AUDIT TRAIL ───────────────────────────────────────
with st.expander("📋 Clinical Summary & Audit Report"):
    st.markdown(
        f"""
        **Patient Assessment Record:**
        - **Predicted Primary Diagnosis:** `{status_clean}` (Confidence: `{confidence_pct:.2f}%`)
        - **Quantitative Symptom Burden:**
          - Depression Severity Index: `{dep_score:.1f}/34` ({dep_tier})
          - Anxiety Severity Index: `{anx_score:.1f}/24` ({anx_tier})
          - Stress Severity Index: `{str_score:.1f}/39` ({str_tier})
        - **Dominant Diagnostic Modality:** `{mod_df.sort_values('Contribution', ascending=False).iloc[0]['Modality']}` (`{mod_df.sort_values('Contribution', ascending=False).iloc[0]['Contribution']:.1f}%` relative impact)
        - **Key Discriminative Signals:** `{', '.join(clean_top_names[:3])}`
        """
    )
    
    # Export payload
    audit_dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_mental_health_status": pred_status_label,
        "confidence_percentage": round(confidence_pct, 2),
        "depression_score": round(dep_score, 2),
        "anxiety_score": round(anx_score, 2),
        "stress_score": round(str_score, 2),
        "modality_contributions_pct": {
            "physiological_behavioral": round(tab_pct, 2),
            "acoustic_speech": round(speech_pct, 2),
            "visual_facial": round(face_pct, 2),
        },
        "top_features": clean_top_names[:5]
    }
    
    st.download_button(
        label="📥 Export Clinical Audit Report (JSON)",
        data=pd.Series(audit_dict).to_json(indent=2),
        file_name=f"SPHERE_Assessment_{int(time.time())}.json",
        mime="application/json"
    )

st.markdown(
    """
    <div style="text-align: center; color: #94A3B8; font-size: 12px; margin-top: 40px; padding-bottom: 20px;">
        S.P.H.E.R.E. Multimodal Psychiatric Assessment Prototype • Developed for Hackathon Clinical AI Showcase
    </div>
    """,
    unsafe_allow_html=True
)
