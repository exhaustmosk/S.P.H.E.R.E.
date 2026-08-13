<div align="center">
  <img src="static/images/logo.png" alt="S.P.H.E.R.E. Logo" width="300">
</div>

### **Synchronized Psychiatric & Health Evaluation through Real-time Explainability**
> *An Explainable Multimodal AI Framework for Objective Clinical Decision Support in Psychiatric Evaluation*

[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![AutoGluon](https://img.shields.io/badge/AutoGluon-1.6.1-FF9900.svg?style=flat)](https://auto.gluon.ai/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.0-EB4034.svg?style=flat)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-TreeSHAP-059669.svg?style=flat)](https://shap.readthedocs.io/)
[![Hardware](https://img.shields.io/badge/Accelerated-Apple%20Silicon%20MPS-black.svg?style=flat&logo=apple)](https://developer.apple.com/metal/pytorch/)

---

## 📌 Executive Summary

Traditional psychiatric assessments rely heavily on subjective questionnaires (e.g., PHQ-9, GAD-7) and retrospective clinical interviews, which suffer from recall bias, masking, and high inter-clinician variability. 

**S.P.H.E.R.E.** introduces an end-to-end, multi-sensor artificial intelligence framework that integrates **Speech Acoustic Prosody**, **Facial Micro-Expressions**, **Behavioral Device Telemetry**, and **Autonomic Physiological Biomarkers** to perform automated psychiatric triage, quantitative symptom estimation, and game-theoretic clinical auditability.

---

## 🏛️ System Architecture

```
                                  ┌───────────────────────────────┐
                                  │   Multimodal Input Streams   │
                                  └───────────────┬───────────────┘
                     ┌────────────────────────────┼────────────────────────────┐
                     ▼                            ▼                            ▼
        ┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐
        │  Speech Emotion Stream  │  │ Visual Affect Stream   │  │ Physiological / Telemetry│
        │   1,440 Audio Files     │  │   28,709 Facial Images  │  │  4,000 Patient Records  │
        │ (Wav2Vec 2.0: 1024-d)   │  │   (ViT Tokens: 768-d)   │  │  (18 Numerical Features)│
        └────────────┬────────────┘  └────────────┬────────────┘  └────────────┬────────────┘
                     │                            │                            │
                     └────────────────────────────┼────────────────────────────┘
                                                  ▼
                               ┌─────────────────────────────────────┐
                               │   Semantic Target Alignment Engine  │
                               │  (Clinically Guided Emotion Pairing)│
                               └──────────────────┬──────────────────┘
                                                  ▼
                               ┌─────────────────────────────────────┐
                               │   Dimensionality Reduction (PCA)    │
                               │  Speech: 1024→32  |  Face: 768→32   │
                               │      Total Fused Matrix: 82-d       │
                               └──────────────────┬──────────────────┘
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 ▼                                                                 ▼
┌──────────────────────────────────┐                             ┌──────────────────────────────────┐
│   Head B: Severity Regressor     │                             │   Head A: Multi-Layer Stacking   │
│   (MultiOutput XGBoost Trees)    │ ──── 5-Fold OOF Scores ───> │     (AutoGluon Best Quality)     │
│   Depression | Anxiety | Stress  │       (85-d Stacked)        │    4-Class Mental Health Status  │
└──────────────────────────────────┘                             └─────────────────┬────────────────┘
                                                                                   ▼
                                                                 ┌──────────────────────────────────┐
                                                                 │  Objective 3: TreeSHAP Engine    │
                                                                 │   Speech: 43.5% | Face: 32.0%    │
                                                                 │        Tabular: 24.6%            │
                                                                 └──────────────────────────────────┘
```

---

## 🎯 Three Core Objectives & Benchmarks

| Objective | Target | Model Architecture | Metric Achieved |
|---|---|---|---|
| **Head A: Classification** | 4-Class Mental Health Status (`Healthy`, `Mild`, `Moderate`, `Severe`) | AutoGluon 3-Layer Stacked Ensemble (RandomForest, ExtraTrees, LightGBM, CatBoost, XGBoost) | **89.38% Accuracy**<br>**0.9785 ROC-AUC**<br>**0.8418 Macro F1** |
| **Head B: Regression** | Continuous Severity Scores (`Depression 0–34`, `Anxiety 0–24`, `Stress 0–39`) | MultiOutputRegressor (`XGBRegressor` w/ 5-Fold OOF) | **MAE: 6.99**<br>**RMSE: 8.54**<br>**R²: 0.1948** |
| **Objective 3: Explainability** | Transparent Modality Attribution & Feature Importance | Game-Theoretic TreeSHAP Engine | **Speech: 43.45%**<br>**Face: 31.99%**<br>**Physiology: 24.56%** |

---

## 📊 Detailed Quantitative Results

### 1. Classification Performance (Test Set: 800 Samples)
```
                     Precision    Recall / Sens.   F1-Score   Support
  Healthy            0.9935       0.9325           0.9620     326    
  Mild_Stress        0.7843       0.9717           0.8680     247    
  Moderate_Stress    0.9023       0.7811           0.8373     201    
  Severe_Stress      1.0000       0.5385           0.7000     26     

  Overall Accuracy:  89.38%
  Macro F1-Score:    0.8418
  Weighted F1-Score: 0.8931
  Multi-Class ROC:   0.9785
```

> **Clinical Safety Highlight:** Zero `Severe Stress` patients were misclassified as `Healthy`. All misclassifications remain bounded to adjacent severity grades.

---

## 📁 Repository Structure

```bash
S.P.H.E.R.E./
├── app.py                             # Main Flask Web Application entry point
├── requirements.txt                   # Production & ML dependencies
├── README.md                          # Comprehensive Documentation
├── .gitignore                         # Git exclusion rules
│
├── core/                              # Core ML & Inference Engine
│   ├── __init__.py
│   ├── ml_engine.py                   # MultimodalInferenceEngine (Live XGBoost inference)
│   └── feature_extractor.py           # Real-time computer vision & acoustic feature extraction
│
├── models/                            # Trained Serialized Model Artifacts & Preprocessors
│   ├── xgb_classifier_head_A_smote.joblib # 4-Class Mental Health Status Classifier
│   ├── xgb_regressor_head_B.joblib    # MultiOutput continuous severity regressor
│   ├── head_B_regressor_semantic.joblib
│   ├── pca_speech_semantic.joblib     # Fitted speech PCA transformer
│   ├── pca_facial_semantic.joblib     # Fitted facial PCA transformer
│   ├── standard_scaler_semantic.joblib# Fitted numerical feature scaler
│   └── shap_xgb_proxy_semantic.joblib # Serialized TreeSHAP proxy model
│
├── data/                              # Tabular Datasets & Pre-computed Embeddings
│   ├── numerical_data.csv             # 4,000-sample physiological/behavioral cohort
│   ├── speech_metadata.csv            # RAVDESS acoustic emotion annotations
│   ├── facial_metadata.csv            # FER2013 visual emotion annotations
│   ├── speech_embeddings.npy          # Wav2Vec 2.0 acoustic embeddings (11MB)
│   └── facial_embeddings.npy          # Vision Transformer facial embeddings (84MB)
│
├── pipelines/                         # Model Training, Stacking & Evaluation Pipelines
│   ├── autogluon_semantic_stacking_pipeline.py
│   ├── autogluon_stacking_pipeline.py
│   ├── sequential_stacking_pipeline_v2.py
│   ├── sequential_stacking_pipeline.py
│   ├── train_xgboost_smote.py
│   ├── train_xgboost.py
│   ├── evaluate_models.py
│   ├── explainability.py
│   ├── extract_facial_embeddings.py
│   ├── extract_speech_embeddings.py
│   ├── merge_multimodal_dataset.py
│   └── system_check.py
│
├── static/                            # Frontend Assets & Stylesheets
│   ├── css/
│   │   └── style.css                  # Medical-tech design system
│   ├── js/
│   │   ├── camera.js                  # Webcam capture & canvas stream
│   │   ├── audio.js                   # Web Audio API recorder & pitch analyser
│   │   └── navigation.js              # Step-by-step validation & form gating
│   └── images/
│       ├── logo.png
│       ├── final_aligned_confusion_matrix.png
│       ├── modality_attribution_bar_chart_semantic.png
│       └── top_15_shap_features_semantic.png
│
└── templates/                         # Interactive HTML5 Diagnostic UI
    ├── base.html                      # Layout shell with progress tracker
    ├── mode_select.html               # Mode selection (Live vs Pre-Recorded)
    ├── live_step1.html                # Step 1: Facial Video & Snapshot Capture
    ├── live_step2.html                # Step 2: Speech Audio & Prompt Reading
    ├── live_step3.html                # Step 3: 6-Slider Behavioral Survey
    ├── live_step4.html                # Step 4: Physiological Wearable Telemetry
    ├── pre_recorded.html              # Pre-recorded batch ingestion form
    ├── report.html                    # Clinical Assessment Dashboard (Objectives 1, 2, 3)
    └── partials/
        └── behavioral_sensor_form.html
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/exhaustmosk/S.P.H.E.R.E..git
cd S.P.H.E.R.E.

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Interactive Web Application
```bash
python3 app.py
```
Open **`http://127.0.0.1:5001`** in your browser to run the live diagnostic wizard.

### 3. Run Pipeline Evaluation & Training
```bash
# Standalone evaluation
python3 pipelines/evaluate_models.py

# SHAP Explainability analysis
python3 pipelines/explainability.py
```

---

## 👥 Research & Engineering Team
* **Akshaj S.** — Lead AI Engineer & Machine Learning Architect
* **Pranshu Verma** — Computer Vision & Deep Learning Specialist


*Department of DSBS — SRM Institute of Science and Technology (SRMIST)*
