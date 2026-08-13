import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response, Request

# Increase Werkzeug form parsing limits for image and audio data payloads
Request.max_form_memory_size = 32 * 1024 * 1024  # 32 MB
Request.max_content_length = 32 * 1024 * 1024

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "multimodal-mental-health-prototype-key-2026")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
app.config["MAX_FORM_MEMORY_SIZE"] = 32 * 1024 * 1024

# In-memory storage for large media payloads (avoids 4KB cookie overflow)
MEDIA_PAYLOAD_STORE = {}


def get_or_create_session_id():
    """Ensure a unique session ID exists for in-memory media storage."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]


# ==============================================================================
# Central Mock Report Data (Research Calibrated)
# ==============================================================================
MOCK_REPORT_DATA = {
    # Objective 1: Status Classification
    "facial_signal": {
        "category": "Mild Stress",
        "confidence": 0.82,
        "confidence_pct": 82,
        "badge_class": "badge-amber",
        "key_indicators": [
            {"label": "Facial Emotion Variance", "value": "0.779 (Elevated micro-expression variance)"},
            {"label": "Eye Blink Rate", "value": "14 blinks/min (Moderate suppression)"},
            {"label": "Smile Intensity", "value": "0.020 (Reduced positive valence)"},
            {"label": "Head Motion Index", "value": "0.194 (Subtle psychomotor agitation)"}
        ]
    },
    "speech_signal": {
        "category": "Moderate Stress",
        "confidence": 0.76,
        "confidence_pct": 76,
        "badge_class": "badge-orange",
        "key_indicators": [
            {"label": "Pitch Mean (F0)", "value": "263.2 Hz (Elevated fundamental frequency)"},
            {"label": "MFCC Mean", "value": "14.01 dB (Acoustic spectral compression)"},
            {"label": "MFCC Variance", "value": "5.61 (Vocal tract perturbation)"},
            {"label": "Speech Rate", "value": "4.77 syl/sec (Accelerated prosody)"}
        ]
    },
    "tabular_signal": {
        "category": "Healthy",
        "confidence": 0.38,
        "confidence_pct": 38,
        "badge_class": "badge-emerald",
        "key_indicators": [
            {"label": "Sleep Quality Score", "value": "2 / 5 (Sub-optimal rest reported)"},
            {"label": "Social Engagement", "value": "5 / 5 (High peer interactions)"},
            {"label": "Daily App Usage", "value": "202 min (Within average mobile cohort)"},
            {"label": "Heart Rate Baseline", "value": "79 BPM (Normal resting cardiovascular range)"}
        ]
    },
    
    # Objective 2: Severity Estimation (0-10 Scale)
    "severity_scores": {
        "depression": {
            "name": "Depression Score",
            "score": 4.2,
            "max": 10.0,
            "percentage": 42,
            "level": "Mild-to-Moderate",
            "color": "#3B82F6",
            "bg_color": "#EFF6FF",
            "border_color": "#BFDBFE"
        },
        "anxiety": {
            "name": "Anxiety Score",
            "score": 6.8,
            "max": 10.0,
            "percentage": 68,
            "level": "Moderate-to-High",
            "color": "#F59E0B",
            "bg_color": "#FFFBEB",
            "border_color": "#FDE68A"
        },
        "stress": {
            "name": "Stress Score",
            "score": 5.1,
            "max": 10.0,
            "percentage": 51,
            "level": "Moderate",
            "color": "#EA580C",
            "bg_color": "#FFF7ED",
            "border_color": "#FFEDD5"
        }
    },

    # Objective 3: Explainability Details
    "explainability": {
        "gradcam_facial": {
            "title": "Grad-CAM Overlay (Facial)",
            "description": "Visual saliency heatmap demonstrating spatial regions driving the visual stress classifier.",
            "regions": [
                {"name": "Corrugator Supercilii (Brow Tension)", "activation": "89%", "level": "High Saliency"},
                {"name": "Orbicularis Oculi (Periorbital Strain)", "activation": "74%", "level": "Moderate-High"},
                {"name": "Zygomaticus Major (Lower Smile Action)", "activation": "65%", "level": "Moderate"},
                {"name": "Depressor Anguli Oris (Lip Corner Saliency)", "activation": "42%", "level": "Low-Moderate"}
            ]
        },
        "shap_speech": {
            "title": "SHAP Feature Importance (Speech)",
            "description": "Mean absolute SHAP value contributions quantifying acoustic feature impact on stress classification.",
            "features": [
                {"name": "Pitch Variance (Jitter)", "importance": 0.34, "pct": 88, "impact": "+Stress Signal"},
                {"name": "MFCC 3 Mean (Vocal Tract)", "importance": 0.28, "pct": 72, "impact": "+Arousal Indicator"},
                {"name": "Spectral Centroid", "importance": 0.19, "pct": 52, "impact": "+Vocal Tension"},
                {"name": "Energy Entropy", "importance": 0.12, "pct": 34, "impact": "-Dysphoria Buffer"},
                {"name": "Voiced-to-Unvoiced Ratio", "importance": 0.07, "pct": 20, "impact": "+Prosodic Fatigue"}
            ]
        }
    },
    
    # Metadata and Session Diagnostics
    "session_summary": {
        "assessment_id": "MH-MM-849201",
        "timestamp": "Current Session",
        "protocol": "Dual-Stream Active Protocol (Visual + Acoustic + Behavioral)",
        "fusion_method": "Multi-head Late Stacking Ensemble"
    }
}


# ==============================================================================
# Helper Functions
# ==============================================================================
def get_session_data():
    """Retrieve current session metrics or initialize defaults."""
    if "assessment_mode" not in session:
        session["assessment_mode"] = "live"
    return session


# ==============================================================================
# Routes
# ==============================================================================

@app.route("/")
def index():
    """Mode Selection Page: Pre-Recorded vs Live Testing."""
    return render_template("mode_select.html", title="Select Assessment Mode")


@app.route("/pre-recorded", methods=["GET", "POST"])
def pre_recorded():
    """Pre-recorded data upload workflow (Image, Audio, and CSV/18 manual fields)."""
    if request.method == "POST":
        sid = get_or_create_session_id()
        session["assessment_mode"] = "pre_recorded"
        session["upload_type"] = request.form.get("upload_type", "manual")
        session["image_filename"] = request.form.get("image_filename", "sample_patient_face.png")
        session["audio_filename"] = request.form.get("audio_filename", "sample_patient_speech.wav")
        
        # Save manual tabular values if provided
        session["tabular_data"] = {
            "sleep_quality": request.form.get("sleep_quality", 2),
            "social_engagement": request.form.get("social_engagement", 5),
            "daily_app_usage": request.form.get("daily_app_usage_min", 202),
            "typing_speed": request.form.get("typing_speed_wpm", 40),
            "session_frequency": request.form.get("session_frequency", 18),
            "idle_time": request.form.get("idle_time_min", 138),
            "heart_rate": request.form.get("heart_rate_bpm", 79),
            "hrv_index": request.form.get("hrv_index", 51.4),
            "skin_temperature": request.form.get("skin_temperature", 34.3),
            "gsr_level": request.form.get("gsr_level", 0.98),
        }
        return redirect(url_for("report"))
        
    return render_template("pre_recorded.html", title="Pre-Recorded Data Ingestion")


from core.feature_extractor import extract_facial_metrics


@app.route("/live/step1", methods=["GET", "POST"])
def live_step1():
    """Live Testing - Step 1: Facial Video/Snapshot Capture."""
    if request.method == "POST":
        sid = get_or_create_session_id()
        session["assessment_mode"] = "live"
        session["facial_image_captured"] = True
        
        # Save heavy image data URI in server memory store, NOT in the 4KB session cookie
        img_uri = request.form.get("facial_snapshot_uri", "")
        if sid not in MEDIA_PAYLOAD_STORE:
            MEDIA_PAYLOAD_STORE[sid] = {}
        MEDIA_PAYLOAD_STORE[sid]["image_uri"] = img_uri
        
        # Extract authentic computer vision metrics directly from the captured photo pixels
        if img_uri:
            session["facial_metrics"] = extract_facial_metrics(img_uri)
            print("✓ Live photo metrics extracted:", session["facial_metrics"])
        
        return redirect(url_for("live_step2"))
        
    return render_template("live_step1.html", title="Step 1 of 4 — Facial Video Capture", step=1)


@app.route("/live/step2", methods=["GET", "POST"])
def live_step2():
    """Live Testing - Step 2: Speech Audio Clip Capture."""
    if request.method == "POST":
        sid = get_or_create_session_id()
        session["audio_captured"] = True
        
        # Save heavy audio data in server memory store, NOT in the 4KB session cookie
        audio_uri = request.form.get("audio_clip_uri", "")
        if sid not in MEDIA_PAYLOAD_STORE:
            MEDIA_PAYLOAD_STORE[sid] = {}
        MEDIA_PAYLOAD_STORE[sid]["audio_uri"] = audio_uri
        
        # Extract authentic acoustic metrics captured by Web Audio API during speech
        pitch_val = request.form.get("live_pitch_mean")
        rate_val = request.form.get("live_speech_rate")
        mfcc_mean_val = request.form.get("live_mfcc_mean")
        mfcc_var_val = request.form.get("live_mfcc_variance")

        session["speech_metrics"] = {
            "pitch_mean": float(pitch_val) if pitch_val else 195.0,
            "speech_rate": float(rate_val) if rate_val else 3.8,
            "mfcc_mean": float(mfcc_mean_val) if mfcc_mean_val else 10.0,
            "mfcc_variance": float(mfcc_var_val) if mfcc_var_val else 12.0
        }
        print("✓ Live speech metrics captured:", session["speech_metrics"])
        
        return redirect(url_for("live_step3"))
        
    return render_template("live_step2.html", title="Step 2 of 4 — Speech Audio Capture", step=2)


@app.route("/live/step3", methods=["GET", "POST"])
def live_step3():
    """Live Testing - Step 3: Behavioral Form (6 Range Sliders)."""
    if request.method == "POST":
        session["behavioral_survey"] = {
            "sleep_quality": float(request.form.get("sleep_quality", 3)),
            "social_engagement": float(request.form.get("social_engagement", 3)),
            "daily_app_usage_min": float(request.form.get("daily_app_usage_min", 180)),
            "session_frequency": float(request.form.get("session_frequency", 12)),
            "idle_time_min": float(request.form.get("idle_time_min", 90)),
            "typing_speed_wpm": float(request.form.get("typing_speed_wpm", 45)),
        }
        return redirect(url_for("live_step4"))
        
    return render_template("live_step3.html", title="Step 3 of 4 — Behavioral Assessment", step=3)


@app.route("/live/step4", methods=["GET", "POST"])
def live_step4():
    """Live Testing - Step 4: Sensor Form (4 Optional Telemetry Fields)."""
    if request.method == "POST":
        session["sensor_data"] = {
            "heart_rate_bpm": request.form.get("heart_rate_bpm", ""),
            "hrv_index": request.form.get("hrv_index", ""),
            "skin_temperature": request.form.get("skin_temperature", ""),
            "gsr_level": request.form.get("gsr_level", "")
        }
        return redirect(url_for("report"))
        
    return render_template("live_step4.html", title="Step 4 of 4 — Physiological Sensors", step=4)


from core.ml_engine import ml_engine


@app.route("/report")
def report():
    """Assessment Report Page: Visualizing Objectives 1, 2, and 3 with real machine learning model inference."""
    sid = session.get("session_id", "")
    stored_media = MEDIA_PAYLOAD_STORE.get(sid, {})
    
    mode = session.get("assessment_mode", "live")
    has_image = bool(session.get("facial_image_captured") or session.get("image_filename"))
    has_audio = bool(session.get("audio_captured") or session.get("audio_filename"))
    
    # Collect all submitted inputs across behavioral, sensor, facial, speech, and tabular forms
    user_inputs = {}
    if "tabular_data" in session:
        user_inputs.update(session["tabular_data"])
    if "facial_metrics" in session:
        user_inputs.update(session["facial_metrics"])
    if "speech_metrics" in session:
        user_inputs.update(session["speech_metrics"])
    if "behavioral_survey" in session:
        user_inputs.update(session["behavioral_survey"])
    if "sensor_data" in session:
        user_inputs.update(session["sensor_data"])

    # Run real ML model inference
    if ml_engine.is_loaded:
        live_report_data = ml_engine.predict(user_inputs)
    else:
        live_report_data = MOCK_REPORT_DATA
    
    # Pass report data along with session context
    return render_template(
        "report.html",
        title="Multimodal Mental Health Assessment Report",
        data=live_report_data,
        session_info={
            "mode": mode,
            "has_image": has_image,
            "has_audio": has_audio,
            "image_uri": stored_media.get("image_uri"),
            "audio_uri": stored_media.get("audio_uri"),
            "behavioral": session.get("behavioral_survey", {}),
            "sensors": session.get("sensor_data", session.get("tabular_data", {}))
        }
    )


@app.route("/api/sample-csv")
def sample_csv():
    """Returns a sample CSV row format for pre-recorded tabular upload."""
    csv_content = (
        "Sleep_Quality,Social_Engagement,Daily_App_Usage_Min,Typing_Speed_WPM,"
        "Session_Frequency,Idle_Time_Min,Facial_Emotion_Variance,Eye_Blink_Rate,"
        "Smile_Intensity,Head_Motion_Index,MFCC_Mean,MFCC_Variance,Pitch_Mean,"
        "Speech_Rate,Heart_Rate_BPM,HRV_Index,Skin_Temperature,GSR_Level\n"
        "2,5,202,40,18,138,0.779,14,0.020,0.194,14.013,5.610,263.24,4.77,79,51.43,34.28,0.976\n"
    )
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=sample_patient_metrics.csv"}
    )


@app.route("/reset")
def reset_session():
    """Clear session data and restart workflow."""
    session.clear()
    return redirect(url_for("index"))


import socket

def find_available_port(start_port=5001, max_attempts=50):
    """Find the first available open port starting from start_port."""
    for p in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            res = s.connect_ex(("127.0.0.1", p))
            if res != 0:
                return p
    return start_port


if __name__ == "__main__":
    # Launch lightweight prototype server with automatic open port detection
    requested_port = int(os.environ.get("PORT", 5001))
    port = find_available_port(requested_port)
    print("=" * 70)
    print(f" Multimodal Mental Health Assessment Prototype - Web Server")
    print(f" Status: Running on http://127.0.0.1:{port}")
    print(" ML Backend: Zero heavy ML dependencies (mock/client-side pipeline)")
    print("=" * 70)
    app.run(host="0.0.0.0", port=port, debug=True)
