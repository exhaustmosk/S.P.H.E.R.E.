"""
GUI-only Flask app for the mental-health assessment flow.

No model loading, no inference, no feature extraction.
Every number on /report comes from MOCK_REPORT below.

Routes
------
GET  /                 Mode select (pre-recorded vs live)
GET  /pre-recorded     Upload image + audio; CSV row or manual 6+4 fields
POST /pre-recorded     Ignore payloads; set session mode; redirect to /report
GET  /live/step1       Photo/video capture UI
POST /live/step1       Redirect to /live/step2
GET  /live/step2       Audio capture UI
POST /live/step2       Redirect to /live/step3
GET  /live/step3       Behavioural sliders (6 fields)
POST /live/step3       Redirect to /live/step4
GET  /live/step4       Optional wearable / sensor fields
POST /live/step4       Redirect to /report (Skip does the same)
GET  /report           Full layout with MOCK_REPORT
"""

from flask import Flask, redirect, render_template, request, session, url_for
import os
import csv
from werkzeug.utils import secure_filename
from inference import run_inference

app = Flask(__name__)
app.secret_key = "gui-preview-only-not-a-secret"
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

SPEECH_PROMPT = "The quick brown fox jumps over the lazy dog."

BEHAVIORAL_FIELDS = [
    {
        "name": "Sleep_Quality",
        "label": "Sleep quality",
        "low": "Poor",
        "high": "Excellent",
        "hint": "Typical sleep over the past week.",
    },
    {
        "name": "Social_Engagement",
        "label": "Social engagement",
        "low": "Isolated",
        "high": "Very connected",
        "hint": "Time and quality of social contact.",
    },
    {
        "name": "Daily_App_Usage_Min",
        "label": "App usage",
        "low": "Very low",
        "high": "Very high",
        "hint": "Relative daily use of phone or assessment apps.",
    },
    {
        "name": "Session_Frequency",
        "label": "Session frequency",
        "low": "Rare",
        "high": "Very frequent",
        "hint": "How often you open or complete sessions.",
    },
    {
        "name": "Idle_Time_Min",
        "label": "Idle time",
        "low": "Almost none",
        "high": "Very high",
        "hint": "Relative time spent idle on the device.",
    },
    {
        "name": "Typing_Speed_WPM",
        "label": "Typing speed",
        "low": "Very slow",
        "high": "Very fast",
        "hint": "Relative typing pace.",
    },
]

SENSOR_FIELDS = [
    {
        "name": "Heart_Rate_BPM",
        "label": "Heart rate",
        "unit": "bpm",
        "placeholder": "e.g. 72",
        "step": "1",
        "min": "30",
        "max": "220",
    },
    {
        "name": "HRV_Index",
        "label": "HRV index",
        "unit": "ms",
        "placeholder": "e.g. 45",
        "step": "0.1",
        "min": "0",
        "max": "300",
    },
    {
        "name": "Skin_Temperature",
        "label": "Skin temperature",
        "unit": "°C",
        "placeholder": "e.g. 33.2",
        "step": "0.1",
        "min": "20",
        "max": "45",
    },
    {
        "name": "GSR_Level",
        "label": "GSR level",
        "unit": "µS",
        "placeholder": "e.g. 2.4",
        "step": "0.01",
        "min": "0",
        "max": "40",
    },
]

CSV_COLUMNS = [
    "Sleep_Quality",
    "Social_Engagement",
    "Daily_App_Usage_Min",
    "Typing_Speed_WPM",
    "Session_Frequency",
    "Idle_Time_Min",
    "Facial_Emotion_Variance",
    "Eye_Blink_Rate",
    "Smile_Intensity",
    "Head_Motion_Index",
    "MFCC_Mean",
    "MFCC_Variance",
    "Pitch_Mean",
    "Speech_Rate",
    "Heart_Rate_BPM",
    "HRV_Index",
    "Skin_Temperature",
    "GSR_Level",
]

# Single swap-in point for real backend results later.
MOCK_REPORT = {
    "objective1": {
        "facial": {"category": "Mild", "confidence": 0.76},
        "speech": {"category": "Moderate", "confidence": 0.81},
        "tabular": {"category": "Healthy", "confidence": 0.38},
        "independence_note": (
            "These are two independent signals, not a fused prediction — see methodology."
        ),
        "tabular_caveat": (
            "This model's accuracy is near a random baseline (~38%) — treat this signal as low-confidence."
        ),
    },
    "objective2": {
        "scale": "0–10",
        "scores": {
            "depression": 4.2,
            "anxiety": 5.8,
            "stress": 6.1,
        },
        "caveat": (
            "This model currently has no confirmed predictive accuracy for these scores "
            "(R² ≈ 0) — values shown are not clinically reliable."
        ),
    },
    "objective3": {
        "gradcam_label": "Grad-CAM overlay (facial)",
        "shap_label": "SHAP feature importance (speech)",
        "shap_bars": [
            {"name": "Pitch variability", "value": 0.82},
            {"name": "Speech rate", "value": 0.64},
            {"name": "MFCC-1", "value": 0.51},
            {"name": "Energy", "value": 0.38},
            {"name": "Pause ratio", "value": 0.22},
        ],
        "tabular_note": (
            "Tabular feature importances are near-zero and non-differentiated by design — "
            "see full report for details."
        ),
        "speech_model_note": (
            "This model was fine-tuned on acted speech (RAVDESS); its high accuracy reflects "
            "that training setup, not general real-world reliability."
        ),
    },
}


def _form_context(**extra):
    ctx = {
        "behavioral_fields": BEHAVIORAL_FIELDS,
        "sensor_fields": SENSOR_FIELDS,
        "speech_prompt": SPEECH_PROMPT,
        "csv_columns": CSV_COLUMNS,
    }
    ctx.update(extra)
    return ctx


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/pre-recorded", methods=["GET", "POST"])
def pre_recorded():
    if request.method == "POST":
        session["mode"] = "pre-recorded"
        
        # 1. Save media
        img = request.files.get("image_file")
        audio = request.files.get("audio_file")
        
        img_path = "/tmp/" + secure_filename(img.filename)
        img.save(img_path)
        
        audio_path = "/tmp/" + secure_filename(audio.filename)
        audio.save(audio_path)
        
        # 2. Get tabular dict
        tabular_dict = {}
        tmode = request.form.get("tabular_mode")
        if tmode == "csv":
            csv_file = request.files.get("csv_file")
            csv_path = "/tmp/" + secure_filename(csv_file.filename)
            csv_file.save(csv_path)
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                row = next(reader)
                for k in CSV_COLUMNS:
                    tabular_dict[k] = row.get(k, 0)
        else:
            for k in CSV_COLUMNS:
                tabular_dict[k] = request.form.get(k, 0)
                
        # 3. Run inference
        report = run_inference(img_path, audio_path, tabular_dict)
        session["real_report"] = report
        
        return redirect(url_for("report"))
    return render_template("pre_recorded.html", **_form_context())


@app.route("/live/step1", methods=["GET", "POST"])
def live_step1():
    if request.method == "POST":
        session["mode"] = "live"
        vf = request.files.get("video_file")
        if vf and vf.filename:
            path = "/tmp/live_" + secure_filename(vf.filename)
            vf.save(path)
            session["live_video_path"] = path
        return redirect(url_for("live_step2"))
    return render_template(
        "live_step1.html",
        **_form_context(step=1, step_label="Photo", back_url=url_for("index")),
    )


@app.route("/live/step2", methods=["GET", "POST"])
def live_step2():
    if request.method == "POST":
        session["mode"] = "live"
        af = request.files.get("audio_file")
        if af and af.filename:
            path = "/tmp/live_" + secure_filename(af.filename)
            af.save(path)
            session["live_audio_path"] = path
        return redirect(url_for("live_step3"))
    return render_template(
        "live_step2.html",
        **_form_context(
            step=2,
            step_label="Speech",
            back_url=url_for("live_step1"),
        ),
    )


@app.route("/live/step3", methods=["GET", "POST"])
def live_step3():
    if request.method == "POST":
        session["mode"] = "live"
        # store behavioural
        for f in BEHAVIORAL_FIELDS:
            session[f"live_{f['name']}"] = request.form.get(f['name'], 0)
        return redirect(url_for("live_step4"))
    return render_template(
        "live_step3.html",
        **_form_context(
            step=3,
            step_label="Behaviour",
            back_url=url_for("live_step2"),
        ),
    )


@app.route("/live/step4", methods=["GET", "POST"])
def live_step4():
    if request.method == "POST":
        session["mode"] = "live"
        # 1. Get sensors
        for f in SENSOR_FIELDS:
            session[f"live_{f['name']}"] = request.form.get(f['name'], 0)
            
        # 2. Build tabular dict
        tabular_dict = {}
        for col in CSV_COLUMNS:
            tabular_dict[col] = session.get(f"live_{col}", 0)
            
        # 3. Run inference
        img_path = session.get("live_video_path", "")
        audio_path = session.get("live_audio_path", "")
        
        report = run_inference(img_path, audio_path, tabular_dict)
        session["real_report"] = report
        
        return redirect(url_for("report"))
    return render_template(
        "live_step4.html",
        **_form_context(
            step=4,
            step_label="Sensors",
            back_url=url_for("live_step3"),
        ),
    )


@app.route("/report")
def report():
    mock = session.get("real_report", MOCK_REPORT)
    return render_template(
        "report.html",
        mock=mock,
        mode=session.get("mode", "preview"),
    )


from inference import run_inference, init_models

if __name__ == "__main__":
    print("Pre-loading machine learning models...")
    init_models()
    app.run(debug=True, use_reloader=False)
