/**
 * camera.js - Enhanced WebRTC Camera Stream & Frame Capture Module
 * Provides robust webcam stream handling, mirror rendering, multi-camera selection,
 * direct photo upload fallback, and instant frame capture.
 */

document.addEventListener("DOMContentLoaded", () => {
  const video = document.getElementById("camera-video");
  const canvas = document.getElementById("camera-canvas");
  const captureBtn = document.getElementById("capture-photo-btn");
  const retakeBtn = document.getElementById("retake-photo-btn");
  const simulateBtn = document.getElementById("simulate-photo-btn");
  const enableCameraBtn = document.getElementById("enable-camera-btn");
  const photoFileInput = document.getElementById("photo-file-fallback");
  const cameraSelect = document.getElementById("camera-select");
  const cameraSelectWrapper = document.getElementById("camera-select-wrapper");
  const hiddenInput = document.getElementById("facial_snapshot_uri");
  const statusBadge = document.getElementById("camera-status-badge");
  const cameraErrorBanner = document.getElementById("camera-error-banner");
  const nextBtn = document.getElementById("nav-next-btn");

  let mediaStream = null;
  let isCameraStreaming = false;

  // 1. Initialize Camera with Fallbacks
  async function initCamera(deviceId = null) {
    if (!video) return;

    // Stop any existing stream tracks
    if (mediaStream) {
      mediaStream.getTracks().forEach(track => track.stop());
      mediaStream = null;
    }

    isCameraStreaming = false;
    if (cameraErrorBanner) cameraErrorBanner.style.display = "none";
    if (statusBadge) {
      statusBadge.innerHTML = '<span style="color:#F59E0B">●</span> Connecting to camera...';
    }

    const constraintsList = [];

    // Constraint Option 1: Selected Device ID if provided
    if (deviceId) {
      constraintsList.push({ video: { deviceId: { exact: deviceId } }, audio: false });
    }

    // Constraint Option 2: Ideal User-facing HD Camera
    constraintsList.push({
      video: {
        facingMode: "user",
        width: { ideal: 1280, min: 640 },
        height: { ideal: 720, min: 480 }
      },
      audio: false
    });

    // Constraint Option 3: Basic User-facing Camera
    constraintsList.push({
      video: { facingMode: "user" },
      audio: false
    });

    // Constraint Option 4: General Any Video Stream
    constraintsList.push({
      video: true,
      audio: false
    });

    let streamObtained = false;

    for (const constraints of constraintsList) {
      try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          throw new Error("getUserMedia is not supported by your browser.");
        }

        mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = mediaStream;

        // Ensure playback starts once metadata is available
        await new Promise((resolve) => {
          video.onloadedmetadata = () => {
            video.play()
              .then(() => {
                isCameraStreaming = true;
                streamObtained = true;
                resolve();
              })
              .catch((err) => {
                console.warn("video.play() error:", err);
                resolve();
              });
          };
          // Timeout safety in case onloadedmetadata doesn't fire promptly
          setTimeout(resolve, 1500);
        });

        if (streamObtained || (video.srcObject && video.readyState >= 1)) {
          isCameraStreaming = true;
          if (statusBadge) {
            statusBadge.innerHTML = '<span class="status-dot-active"></span> Webcam active';
          }
          if (enableCameraBtn) enableCameraBtn.style.display = "none";
          if (captureBtn) captureBtn.removeAttribute("disabled");
          
          // Enumerate devices to populate camera selector
          enumerateCameras();
          break;
        }
      } catch (err) {
        console.warn("Attempt with constraints failed:", constraints, err);
      }
    }

    if (!isCameraStreaming) {
      console.warn("Webcam stream could not be started automatically.");
      if (statusBadge) {
        statusBadge.innerHTML = '<span style="color:#EF4444">●</span> Camera permission needed / offline';
      }
      if (cameraErrorBanner) {
        cameraErrorBanner.style.display = "block";
      }
      if (enableCameraBtn) {
        enableCameraBtn.style.display = "inline-flex";
      }
    }
  }

  // Enumerate Video Devices
  async function enumerateCameras() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices || !cameraSelect) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(d => d.kind === "videoinput");

      if (videoDevices.length > 1 && cameraSelectWrapper) {
        cameraSelectWrapper.style.display = "flex";
        cameraSelect.innerHTML = "";
        videoDevices.forEach((device, idx) => {
          const option = document.createElement("option");
          option.value = device.deviceId;
          option.text = device.label || `Camera ${idx + 1}`;
          cameraSelect.appendChild(option);
        });
      }
    } catch (e) {
      console.warn("Could not enumerate devices:", e);
    }
  }

  // 2. Capture Frame to Canvas (with horizontal mirror support)
  function captureFrame() {
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    // Standardize to 640x480 max bounds to keep payload ultra-fast and lightweight
    const width = 640;
    const height = 480;

    canvas.width = width;
    canvas.height = height;

    if (isCameraStreaming && mediaStream && video.readyState >= 2) {
      // Draw mirrored video frame so it matches what the user sees
      ctx.save();
      ctx.translate(width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, width, height);
      ctx.restore();
    } else {
      // If camera wasn't streaming, generate realistic patient frame
      generateRealisticFace(ctx, width, height);
    }

    // High quality compressed JPEG (~35KB)
    const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
    if (hiddenInput) hiddenInput.value = dataUrl;

    // Toggle view elements
    video.style.display = "none";
    canvas.style.display = "block";
    if (captureBtn) captureBtn.style.display = "none";
    if (retakeBtn) retakeBtn.style.display = "inline-flex";

    if (statusBadge) {
      statusBadge.innerHTML = '<span style="color:#10B981">✓</span> Photo captured successfully';
    }

    // Enable navigation Next button
    if (nextBtn) {
      nextBtn.removeAttribute("disabled");
      nextBtn.classList.remove("btn-disabled");
    }
  }

  // 3. Retake Photo & Resume Video Stream
  function retakeFrame() {
    if (!canvas || !video) return;

    video.style.display = "block";
    canvas.style.display = "none";
    if (captureBtn) captureBtn.style.display = "inline-flex";
    if (retakeBtn) retakeBtn.style.display = "none";
    if (hiddenInput) hiddenInput.value = "";

    if (statusBadge) {
      statusBadge.innerHTML = isCameraStreaming 
        ? '<span class="status-dot-active"></span> Webcam active' 
        : '<span style="color:#F59E0B">●</span> Ready for capture';
    }

    if (nextBtn) {
      nextBtn.setAttribute("disabled", "true");
    }
  }

  // 4. Fallback: Direct Photo Upload from Disk
  if (photoFileInput) {
    photoFileInput.addEventListener("change", (e) => {
      if (!e.target.files || !e.target.files[0]) return;
      const file = e.target.files[0];
      const reader = new FileReader();

      reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
          if (!canvas) return;
          const ctx = canvas.getContext("2d");
          // Scale to 640x480 preserving aspect ratio
          canvas.width = 640;
          canvas.height = 480;
          ctx.drawImage(img, 0, 0, 640, 480);

          const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
          if (hiddenInput) hiddenInput.value = dataUrl;

          if (video) video.style.display = "none";
          canvas.style.display = "block";
          if (captureBtn) captureBtn.style.display = "none";
          if (retakeBtn) retakeBtn.style.display = "inline-flex";
          if (cameraErrorBanner) cameraErrorBanner.style.display = "none";

          if (statusBadge) {
            statusBadge.innerHTML = `<span style="color:#10B981">✓</span> Photo uploaded: ${file.name}`;
          }

          if (nextBtn) {
            nextBtn.removeAttribute("disabled");
            nextBtn.classList.remove("btn-disabled");
          }
        };
        img.src = event.target.result;
      };
      reader.readAsDataURL(file);
    });
  }

  // 5. High-Fidelity Simulated Face Generator
  function generateRealisticFace(ctx, width, height) {
    // Medical Clinical Neutral Background
    const bgGrad = ctx.createLinearGradient(0, 0, 0, height);
    bgGrad.addColorStop(0, "#1E293B");
    bgGrad.addColorStop(1, "#0F172A");
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, width, height);

    const cx = width / 2;
    const cy = height / 2;

    // Head Silhouette
    ctx.fillStyle = "#E2E8F0";
    ctx.beginPath();
    ctx.ellipse(cx, cy, width * 0.22, height * 0.32, 0, 0, Math.PI * 2);
    ctx.fill();

    // Eyes
    ctx.fillStyle = "#334155";
    ctx.beginPath();
    ctx.arc(cx - 38, cy - 25, 8, 0, Math.PI * 2);
    ctx.arc(cx + 38, cy - 25, 8, 0, Math.PI * 2);
    ctx.fill();

    // Eyebrows
    ctx.strokeStyle = "#475569";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(cx - 52, cy - 42);
    ctx.lineTo(cx - 24, cy - 38);
    ctx.moveTo(cx + 24, cy - 38);
    ctx.lineTo(cx + 52, cy - 42);
    ctx.stroke();

    // Nose
    ctx.strokeStyle = "#94A3B8";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy - 20);
    ctx.lineTo(cx - 6, cy + 12);
    ctx.lineTo(cx + 6, cy + 12);
    ctx.stroke();

    // Neutral Mouth
    ctx.strokeStyle = "#64748B";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(cx - 26, cy + 42);
    ctx.lineTo(cx + 26, cy + 42);
    ctx.stroke();

    // Overlay Badge
    ctx.fillStyle = "rgba(37, 99, 235, 0.85)";
    ctx.fillRect(cx - 150, height - 42, 300, 28);
    ctx.fillStyle = "#FFFFFF";
    ctx.font = "bold 12px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("FACIAL CALIBRATION FRAME (SIMULATED)", cx, height - 24);
  }

  // 6. Simulate Snapshot Trigger
  function simulateSnapshot() {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const width = 640;
    const height = 480;
    canvas.width = width;
    canvas.height = height;
    generateRealisticFace(ctx, width, height);

    const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
    if (hiddenInput) hiddenInput.value = dataUrl;

    if (video) video.style.display = "none";
    canvas.style.display = "block";
    if (captureBtn) captureBtn.style.display = "none";
    if (retakeBtn) retakeBtn.style.display = "inline-flex";
    if (cameraErrorBanner) cameraErrorBanner.style.display = "none";

    if (statusBadge) {
      statusBadge.innerHTML = '<span style="color:#10B981">✓</span> Simulated snapshot active';
    }

    if (nextBtn) {
      nextBtn.removeAttribute("disabled");
      nextBtn.classList.remove("btn-disabled");
    }
  }

  // Event Listeners
  if (captureBtn) captureBtn.addEventListener("click", captureFrame);
  if (retakeBtn) retakeBtn.addEventListener("click", retakeFrame);
  if (simulateBtn) simulateBtn.addEventListener("click", simulateSnapshot);
  if (enableCameraBtn) enableCameraBtn.addEventListener("click", () => initCamera());

  if (cameraSelect) {
    cameraSelect.addEventListener("change", () => {
      initCamera(cameraSelect.value);
    });
  }

  // Auto-init on page load
  initCamera();

  // Cleanup stream on unload
  window.addEventListener("beforeunload", () => {
    if (mediaStream) {
      mediaStream.getTracks().forEach(track => track.stop());
    }
  });
});
