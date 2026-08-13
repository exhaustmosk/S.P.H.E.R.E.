(function () {
  const page = document.body.dataset.page;

  function showError(el, message) {
    if (!el) return;
    el.textContent = message || "";
    el.hidden = !message;
  }

  function setNextEnabled(btn, helper, ok, needMessage) {
    if (!btn) return;
    btn.disabled = !ok;
    if (helper) {
      helper.textContent = ok ? "" : needMessage;
      helper.hidden = ok;
    }
  }

  function initSliders(root, onChange) {
    const blocks = root.querySelectorAll("[data-slider-block]");
    blocks.forEach(function (block) {
      const slider = block.querySelector("input[type=range]");
      const valueEl = block.querySelector("[data-slider-value]");
      const touched = block.querySelector("input[data-touched]");
      if (!slider) return;

      function paint() {
        if (valueEl) {
          valueEl.textContent =
            touched && touched.value === "1" ? slider.value : "Not set";
        }
        block.classList.toggle("is-touched", touched && touched.value === "1");
      }

      slider.addEventListener("input", function () {
        if (touched) touched.value = "1";
        paint();
        if (onChange) onChange();
      });
      paint();
    });

    return function allTouched() {
      return Array.from(blocks).every(function (block) {
        const touched = block.querySelector("input[data-touched]");
        return touched && touched.value === "1";
      });
    };
  }

  if (page === "live-step1") {
    const preview = document.getElementById("preview");
    const overlay = document.getElementById("countdown");
    const recDot = document.getElementById("rec-dot");
    const captureBtn = document.getElementById("capture-btn");
    const retakeBtn = document.getElementById("retake-btn");
    const fileInput = document.getElementById("video-file");
    const captured = document.getElementById("video_captured");
    const nextBtn = document.getElementById("next-btn");
    const helper = document.getElementById("next-helper");
    const errorEl = document.getElementById("capture-error");
    const RECORD_MS = 4000;
    let stream = null;
    let blobUrl = null;

    function markCaptured(yes) {
      captured.value = yes ? "1" : "";
      retakeBtn.classList.toggle("hidden", !yes);
      captureBtn.classList.toggle("hidden", yes);
      setNextEnabled(nextBtn, helper, yes, "Capture a short clip before continuing.");
    }

    function stopStream() {
      if (stream) {
        stream.getTracks().forEach(function (t) {
          t.stop();
        });
        stream = null;
      }
    }

    function revoke() {
      if (blobUrl) {
        URL.revokeObjectURL(blobUrl);
        blobUrl = null;
      }
    }

    async function startLive() {
      showError(errorEl, "");
      revoke();
      stopStream();
      preview.removeAttribute("src");
      preview.controls = false;
      preview.muted = true;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });
        preview.srcObject = stream;
        await preview.play();
        markCaptured(false);
      } catch (err) {
        showError(
          errorEl,
          "Camera access is needed for live capture. Allow the camera, or choose a file below."
        );
      }
    }

    function useBlob(blob) {
      stopStream();
      revoke();
      blobUrl = URL.createObjectURL(blob);
      preview.srcObject = null;
      preview.src = blobUrl;
      preview.muted = false;
      preview.controls = true;
      preview.play().catch(function () {});
      try {
        sessionStorage.setItem("live_video_present", "1");
      } catch (e) {}
      markCaptured(true);
    }

    captureBtn.addEventListener("click", function () {
      if (!stream) {
        showError(errorEl, "Start the camera first, or choose a file.");
        return;
      }
      const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp8")
        ? "video/webm;codecs=vp8"
        : "video/webm";
      const recorder = new MediaRecorder(stream, { mimeType: mime });
      const chunks = [];
      recorder.ondataavailable = function (ev) {
        if (ev.data && ev.data.size) chunks.push(ev.data);
      };
      recorder.onstop = function () {
        recDot.classList.remove("is-on");
        overlay.classList.remove("is-on");
        useBlob(new Blob(chunks, { type: recorder.mimeType || "video/webm" }));
      };
      recorder.start();
      recDot.classList.add("is-on");
      overlay.classList.add("is-on");
      let left = Math.round(RECORD_MS / 1000);
      overlay.textContent = String(left);
      const tick = setInterval(function () {
        left -= 1;
        overlay.textContent = String(Math.max(left, 0));
        if (left <= 0) clearInterval(tick);
      }, 1000);
      setTimeout(function () {
        if (recorder.state === "recording") recorder.stop();
      }, RECORD_MS);
    });

    retakeBtn.addEventListener("click", function () {
      markCaptured(false);
      startLive();
    });

    fileInput.addEventListener("change", function () {
      const file = fileInput.files && fileInput.files[0];
      if (file) useBlob(file);
    });

    markCaptured(false);
    startLive();
  }

  if (page === "live-step2") {
    const canvas = document.getElementById("waveform");
    const ctx = canvas.getContext("2d");
    const timerEl = document.getElementById("timer");
    const recordBtn = document.getElementById("record-btn");
    const stopBtn = document.getElementById("stop-btn");
    const rerecordBtn = document.getElementById("rerecord-btn");
    const player = document.getElementById("playback");
    const captured = document.getElementById("audio_captured");
    const nextBtn = document.getElementById("next-btn");
    const helper = document.getElementById("next-helper");
    const errorEl = document.getElementById("capture-error");
    const fileInput = document.getElementById("audio-file");
    let stream = null;
    let recorder = null;
    let analyser = null;
    let audioCtx = null;
    let raf = null;
    let startedAt = 0;
    let blobUrl = null;
    let timerRaf = null;

    function drawIdle() {
      ctx.fillStyle = "#F7F9FC";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "#E5E7EB";
      ctx.beginPath();
      ctx.moveTo(0, canvas.height / 2);
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
    }

    function drawWave() {
      if (!analyser) return;
      const data = new Uint8Array(analyser.fftSize);
      analyser.getByteTimeDomainData(data);
      ctx.fillStyle = "#F7F9FC";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "#6B7280";
      ctx.lineWidth = 2;
      ctx.beginPath();
      const slice = canvas.width / data.length;
      for (let i = 0; i < data.length; i++) {
        const v = data[i] / 128.0;
        const y = (v * canvas.height) / 2;
        const x = i * slice;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      raf = requestAnimationFrame(drawWave);
    }

    function formatMs(ms) {
      const s = Math.floor(ms / 1000);
      const m = Math.floor(s / 60);
      const rem = s % 60;
      return String(m).padStart(2, "0") + ":" + String(rem).padStart(2, "0");
    }

    function tickTimer() {
      timerEl.textContent = formatMs(Date.now() - startedAt);
      timerRaf = requestAnimationFrame(tickTimer);
    }

    function markCaptured(yes) {
      captured.value = yes ? "1" : "";
      player.classList.toggle("hidden", !yes);
      rerecordBtn.classList.toggle("hidden", !yes);
      recordBtn.classList.toggle("hidden", yes);
      stopBtn.classList.add("hidden");
      setNextEnabled(nextBtn, helper, yes, "Record a clip of the prompt before continuing.");
    }

    function cleanupStream() {
      if (raf) cancelAnimationFrame(raf);
      if (timerRaf) cancelAnimationFrame(timerRaf);
      raf = timerRaf = null;
      if (stream) {
        stream.getTracks().forEach(function (t) {
          t.stop();
        });
        stream = null;
      }
      if (audioCtx) {
        audioCtx.close().catch(function () {});
        audioCtx = null;
      }
      analyser = null;
    }

    function useBlob(blob) {
      cleanupStream();
      if (blobUrl) URL.revokeObjectURL(blobUrl);
      blobUrl = URL.createObjectURL(blob);
      player.src = blobUrl;
      player.classList.remove("hidden");
      try {
        sessionStorage.setItem("live_audio_present", "1");
      } catch (e) {}
      markCaptured(true);
      drawIdle();
    }

    async function startRecording() {
      showError(errorEl, "");
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioCtx.createMediaStreamSource(stream);
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 2048;
        source.connect(analyser);
        const mime = MediaRecorder.isTypeSupported("audio/webm")
          ? "audio/webm"
          : "";
        recorder = mime
          ? new MediaRecorder(stream, { mimeType: mime })
          : new MediaRecorder(stream);
        const chunks = [];
        recorder.ondataavailable = function (ev) {
          if (ev.data && ev.data.size) chunks.push(ev.data);
        };
        recorder.onstop = function () {
          useBlob(new Blob(chunks, { type: recorder.mimeType || "audio/webm" }));
        };
        recorder.start();
        startedAt = Date.now();
        tickTimer();
        drawWave();
        recordBtn.classList.add("hidden");
        stopBtn.classList.remove("hidden");
        rerecordBtn.classList.add("hidden");
        player.classList.add("hidden");
      } catch (err) {
        showError(
          errorEl,
          "Microphone access is needed for live capture. Allow the mic, or choose a file below."
        );
      }
    }

    recordBtn.addEventListener("click", startRecording);
    stopBtn.addEventListener("click", function () {
      if (recorder && recorder.state === "recording") recorder.stop();
    });
    rerecordBtn.addEventListener("click", function () {
      markCaptured(false);
      timerEl.textContent = "00:00";
      startRecording();
    });
    fileInput.addEventListener("change", function () {
      const file = fileInput.files && fileInput.files[0];
      if (file) useBlob(file);
    });

    canvas.width = canvas.clientWidth * 2 || 800;
    canvas.height = 176;
    drawIdle();
    markCaptured(false);
  }

  if (page === "live-step3" || page === "pre-recorded") {
    const form = document.getElementById("flow-form");
    const nextBtn = document.getElementById("next-btn");
    const helper = document.getElementById("next-helper");
    const allTouched = initSliders(form || document, refresh);

    function csvMode() {
      const selected = document.querySelector("input[name=tabular_mode]:checked");
      return selected && selected.value === "csv";
    }

    function refresh() {
      if (page === "live-step3") {
        setNextEnabled(
          nextBtn,
          helper,
          allTouched(),
          "Move every slider at least once before continuing."
        );
        return;
      }

      const imageOk = document.getElementById("image-file").files.length > 0;
      const audioOk = document.getElementById("audio-file").files.length > 0;
      const csvOk = document.getElementById("csv-file").files.length > 0;
      const manualOk = allTouched();
      const tabularOk = csvMode() ? csvOk : manualOk;
      const ok = imageOk && audioOk && tabularOk;
      let msg = "Add the required files and tabular input before continuing.";
      if (!imageOk) msg = "Choose an image file to continue.";
      else if (!audioOk) msg = "Choose an audio file to continue.";
      else if (csvMode() && !csvOk) msg = "Upload a CSV row, or switch to the manual form.";
      else if (!csvMode() && !manualOk)
        msg = "Move every slider at least once, or upload a CSV instead.";
      setNextEnabled(nextBtn, helper, ok, msg);
    }

    if (page === "pre-recorded") {
      const csvPanel = document.getElementById("csv-panel");
      const manualPanel = document.getElementById("manual-panel");
      document.querySelectorAll("input[name=tabular_mode]").forEach(function (radio) {
        radio.addEventListener("change", function () {
          const csv = csvMode();
          csvPanel.classList.toggle("hidden", !csv);
          manualPanel.classList.toggle("hidden", csv);
          refresh();
        });
      });
      ["image-file", "audio-file", "csv-file"].forEach(function (id) {
        const el = document.getElementById(id);
        if (el) el.addEventListener("change", refresh);
      });
    }

    refresh();
  }

  if (page === "live-step4") {
    const nextBtn = document.getElementById("next-btn");
    const helper = document.getElementById("next-helper");
    if (nextBtn) nextBtn.disabled = false;
    if (helper) helper.hidden = true;
  }
})();
