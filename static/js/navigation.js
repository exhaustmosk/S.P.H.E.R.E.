/**
 * navigation.js - Stepper Validation & Form Interactivity Module
 * Dynamically enforces step completion gates and manages form UI state transitions.
 */

document.addEventListener("DOMContentLoaded", () => {
  // ----------------------------------------------------------------------------
  // 1. Step 3: 6 Range Sliders Interactive Tracker
  // ----------------------------------------------------------------------------
  const step3Sliders = document.querySelectorAll(".survey-range-slider");
  const step3NextBtn = document.getElementById("nav-next-btn");
  const touchedSliders = new Set();
  const autofillSlidersBtn = document.getElementById("autofill-survey-btn");

  if (step3Sliders.length > 0) {
    step3Sliders.forEach((slider) => {
      const valueBadge = document.getElementById(`${slider.id}-val`);
      const parentCard = slider.closest(".slider-card");

      // Update value display helper
      const updateValueDisplay = () => {
        if (valueBadge) {
          const unit = slider.dataset.unit || "";
          valueBadge.textContent = `${slider.value} ${unit}`.trim();
        }
        touchedSliders.add(slider.id);
        if (parentCard) parentCard.classList.add("interacted");

        // Check if all 6 sliders are touched
        if (touchedSliders.size >= step3Sliders.length && step3NextBtn) {
          step3NextBtn.removeAttribute("disabled");
          step3NextBtn.classList.remove("btn-disabled");
          const hint = document.getElementById("slider-completion-hint");
          if (hint) {
            hint.innerHTML = '<span style="color:#059669; font-weight:600;">✓ All 6 behavioral metrics recorded. Ready to continue.</span>';
          }
        }
      };

      slider.addEventListener("input", updateValueDisplay);
      slider.addEventListener("change", updateValueDisplay);
    });

    // Helper button to quickly set all sliders for testing
    if (autofillSlidersBtn) {
      autofillSlidersBtn.addEventListener("click", () => {
        step3Sliders.forEach((slider) => {
          touchedSliders.add(slider.id);
          const parentCard = slider.closest(".slider-card");
          if (parentCard) parentCard.classList.add("interacted");
          const valueBadge = document.getElementById(`${slider.id}-val`);
          if (valueBadge) {
            const unit = slider.dataset.unit || "";
            valueBadge.textContent = `${slider.value} ${unit}`.trim();
          }
        });
        if (step3NextBtn) {
          step3NextBtn.removeAttribute("disabled");
          step3NextBtn.classList.remove("btn-disabled");
        }
        const hint = document.getElementById("slider-completion-hint");
        if (hint) {
          hint.innerHTML = '<span style="color:#059669; font-weight:600;">✓ Sample behavioral values applied.</span>';
        }
      });
    }
  }

  // ----------------------------------------------------------------------------
  // 2. Pre-Recorded Ingestion Toggle (CSV vs 18 Manual Fields)
  // ----------------------------------------------------------------------------
  const uploadModeCsvBtn = document.getElementById("toggle-csv-mode");
  const uploadModeManualBtn = document.getElementById("toggle-manual-mode");
  const csvUploadSection = document.getElementById("csv-upload-section");
  const manualFieldsSection = document.getElementById("manual-fields-section");
  const uploadTypeInput = document.getElementById("upload_type_input");

  if (uploadModeCsvBtn && uploadModeManualBtn) {
    uploadModeCsvBtn.addEventListener("click", () => {
      uploadModeCsvBtn.classList.add("active");
      uploadModeManualBtn.classList.remove("active");
      if (csvUploadSection) csvUploadSection.style.display = "block";
      if (manualFieldsSection) manualFieldsSection.style.display = "none";
      if (uploadTypeInput) uploadTypeInput.value = "csv";
    });

    uploadModeManualBtn.addEventListener("click", () => {
      uploadModeManualBtn.classList.add("active");
      uploadModeCsvBtn.classList.remove("active");
      if (csvUploadSection) csvUploadSection.style.display = "none";
      if (manualFieldsSection) manualFieldsSection.style.display = "block";
      if (uploadTypeInput) uploadTypeInput.value = "manual";
    });
  }

  // ----------------------------------------------------------------------------
  // 3. Pre-Recorded File Dropzones & Mock File Selection
  // ----------------------------------------------------------------------------
  const imageInput = document.getElementById("prerecorded-image-input");
  const audioInput = document.getElementById("prerecorded-audio-input");
  const csvFileInput = document.getElementById("prerecorded-csv-input");

  const setupFilePreview = (inputElem, previewId, defaultLabel) => {
    if (!inputElem) return;
    inputElem.addEventListener("change", () => {
      const preview = document.getElementById(previewId);
      if (preview && inputElem.files && inputElem.files[0]) {
        const file = inputElem.files[0];
        preview.innerHTML = `
          <div class="file-preview-card">
            <span><strong>Selected:</strong> ${file.name} (${(file.size / 1024).toFixed(1)} KB)</span>
            <span style="color:#059669; font-weight:600;">✓ Attached</span>
          </div>
        `;
      }
    });
  };

  setupFilePreview(imageInput, "image-file-preview", "Face Image");
  setupFilePreview(audioInput, "audio-file-preview", "Speech Audio");
  setupFilePreview(csvFileInput, "csv-file-preview", "CSV Dataset Row");

  // Autofill sample values for 18 manual fields
  const autofill18Btn = document.getElementById("autofill-18-fields-btn");
  if (autofill18Btn) {
    autofill18Btn.addEventListener("click", () => {
      const defaults = {
        sleep_quality: 2,
        social_engagement: 5,
        daily_app_usage_min: 202,
        typing_speed_wpm: 40,
        session_frequency: 18,
        idle_time_min: 138,
        facial_emotion_variance: 0.779,
        eye_blink_rate: 14,
        smile_intensity: 0.020,
        head_motion_index: 0.194,
        mfcc_mean: 14.013,
        mfcc_variance: 5.610,
        pitch_mean: 263.24,
        speech_rate: 4.77,
        heart_rate_bpm: 79,
        hrv_index: 51.43,
        skin_temperature: 34.28,
        gsr_level: 0.976
      };

      for (const [key, val] of Object.entries(defaults)) {
        const input = document.querySelector(`[name="${key}"]`);
        if (input) input.value = val;
      }
    });
  }

  // Step 4 Autofill sensor helper
  const autofillSensorBtn = document.getElementById("autofill-sensor-btn");
  if (autofillSensorBtn) {
    autofillSensorBtn.addEventListener("click", () => {
      const defaults = {
        heart_rate_bpm: 79,
        hrv_index: 51.4,
        skin_temperature: 34.3,
        gsr_level: 0.98
      };
      for (const [key, val] of Object.entries(defaults)) {
        const input = document.querySelector(`[name="${key}"]`);
        if (input) input.value = val;
      }
    });
  }
});
