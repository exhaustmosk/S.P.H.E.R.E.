"""
extract_speech_embeddings.py
────────────────────────────────────────────────────────────────────────────
Deep Acoustic Feature Extraction Pipeline
  Model  : r-f/wav2vec-english-speech-emotion-recognition  (HuggingFace)
           Architecture: wav2vec2-large  →  1024-dim hidden states
  Device : Apple Silicon MPS → CPU fallback
  Input  : RAVDESS .wav files under ./Audios/Actor_XX/
  Output : speech_embeddings.npy  — (N, 1024) float32
            speech_metadata.csv   — filename, emotion, actor, path

  NOTE: Wav2Vec2Model is loaded (base encoder only). The fine-tuned
  classifier head weights (classifier.dense / out_proj) will appear as
  UNEXPECTED in the load report — this is expected and safe to ignore;
  we only need the encoder hidden states for embedding extraction.
────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ── 1. CONFIGURATION ─────────────────────────────────────────────────────────
AUDIO_ROOT      = Path("./Audios")
SAMPLE_RATE     = 16_000
BATCH_SIZE      = 16
CACHE_EVERY_N   = 10
MAX_AUDIO_SEC   = 10.0
OUT_EMBEDDINGS  = Path("speech_embeddings.npy")
OUT_METADATA    = Path("speech_metadata.csv")
HF_MODEL_ID     = "r-f/wav2vec-english-speech-emotion-recognition"

EMOTION_MAP = {
    "01": "neutral",   "02": "calm",      "03": "happy",
    "04": "sad",       "05": "angry",     "06": "fearful",
    "07": "disgust",   "08": "surprised",
}

# ── colour helpers ────────────────────────────────────────────────────────────
RESET = "\033[0m"; BOLD = "\033[1m"; GREEN = "\033[92m"
YELLOW = "\033[93m"; RED = "\033[91m"; CYAN = "\033[96m"; MAGENTA = "\033[95m"
def ok(m):   return f"{GREEN}✔  {m}{RESET}"
def warn(m): return f"{YELLOW}⚠  {m}{RESET}"
def err(m):  return f"{RED}✘  {m}{RESET}"
def hdr(m):  return f"\n{BOLD}{CYAN}{'─'*62}\n  {m}\n{'─'*62}{RESET}"
def sub(m):  return f"{MAGENTA}  ▸ {m}{RESET}"

# ─────────────────────────────────────────────────────────────────────────────
# 2. DEVICE SETUP
# ─────────────────────────────────────────────────────────────────────────────
print(hdr("1 · Device Setup"))
import torch

mps_available = torch.backends.mps.is_available()
device = torch.device("mps" if mps_available else "cpu")
print(ok(f"Device         : {BOLD}{device}{RESET}"))
print(sub(f"mps.is_built() : {torch.backends.mps.is_built()}"))
print(sub(f"mps.is_avail() : {mps_available}"))

# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAD MODEL & FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────
print(hdr("2 · Loading HuggingFace Model"))
print(sub(f"Model ID : {HF_MODEL_ID}"))
print(sub("Downloading / loading from cache …"))

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

t0 = time.time()
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(HF_MODEL_ID)
model = Wav2Vec2Model.from_pretrained(HF_MODEL_ID)
model = model.to(device)
model.eval()
print(ok(f"Model loaded & moved to '{device}' in {time.time()-t0:.1f}s"))
print(sub(f"Parameters     : {sum(p.numel() for p in model.parameters()):,}"))

# ─────────────────────────────────────────────────────────────────────────────
# 4. DISCOVER WAV FILES
# ─────────────────────────────────────────────────────────────────────────────
print(hdr("3 · Discovering WAV Files"))

if not AUDIO_ROOT.exists():
    print(err(f"Audio root not found: {AUDIO_ROOT.resolve()}"))
    sys.exit(1)

wav_files = sorted(AUDIO_ROOT.rglob("*.wav"))
print(ok(f"Found {len(wav_files):,} .wav files under {AUDIO_ROOT}"))

# ─────────────────────────────────────────────────────────────────────────────
# 5. PARSE RAVDESS FILENAME
# ─────────────────────────────────────────────────────────────────────────────
def parse_ravdess(path: Path) -> dict:
    """
    RAVDESS filename format:
        XX-XX-<Emotion>-XX-XX-XX-<Actor>.wav
    idx:  0   1    2         3  4  5   6
    """
    parts = path.stem.split("-")
    if len(parts) != 7:
        return {
            "Filename": path.name, "Emotion_Code": "??",
            "Emotion_Label": "unknown", "Actor_ID": "??",
            "File_Path": str(path),
        }
    emotion_code = parts[2]
    actor_id     = parts[6]
    return {
        "Filename":      path.name,
        "Emotion_Code":  emotion_code,
        "Emotion_Label": EMOTION_MAP.get(emotion_code, f"code_{emotion_code}"),
        "Actor_ID":      actor_id,
        "File_Path":     str(path),
    }

# ─────────────────────────────────────────────────────────────────────────────
# 6. BATCHED EMBEDDING EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
print(hdr("4 · Extracting Embeddings"))

import librosa
from tqdm import tqdm

MAX_LEN  = int(MAX_AUDIO_SEC * SAMPLE_RATE)
batches  = [wav_files[i : i + BATCH_SIZE] for i in range(0, len(wav_files), BATCH_SIZE)]

print(sub(f"Batch size     : {BATCH_SIZE}"))
print(sub(f"Total batches  : {len(batches)}"))
print(sub(f"Cache flush    : every {CACHE_EVERY_N} batches"))
print()

all_embeddings: list[np.ndarray] = []
all_metadata:   list[dict]       = []
failed_files:   list[str]        = []

pbar = tqdm(
    enumerate(batches),
    total=len(batches),
    desc="Embedding",
    unit="batch",
    dynamic_ncols=True,
    colour="cyan",
)

for batch_idx, batch_paths in pbar:
    raw_waveforms: list[np.ndarray] = []
    batch_meta:    list[dict]       = []

    # ── load audio ────────────────────────────────────────────────────────
    for wav_path in batch_paths:
        try:
            waveform, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
            if len(waveform) < MAX_LEN:
                waveform = np.pad(waveform, (0, MAX_LEN - len(waveform)))
            else:
                waveform = waveform[:MAX_LEN]
            raw_waveforms.append(waveform)
            batch_meta.append(parse_ravdess(wav_path))
        except Exception as exc:
            failed_files.append(f"{wav_path.name} — {exc}")

    if not raw_waveforms:
        continue

    # ── feature extraction + forward pass ─────────────────────────────────
    try:
        inputs = feature_extractor(
            raw_waveforms,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            outputs = model(input_values)
            hidden  = outputs.last_hidden_state    # (B, T, 768)

        # ── mean pooling over time axis ────────────────────────────────────
        embeddings    = hidden.mean(dim=1)         # (B, 1024) — wav2vec2-large
        embeddings_np = embeddings.cpu().float().numpy()

        all_embeddings.append(embeddings_np)
        all_metadata.extend(batch_meta)

    except Exception as exc:
        for m in batch_meta:
            failed_files.append(f"{m['Filename']} — forward pass: {exc}")
        continue

    # ── MPS memory management ──────────────────────────────────────────────
    if device.type == "mps" and (batch_idx + 1) % CACHE_EVERY_N == 0:
        torch.mps.empty_cache()

    pbar.set_postfix(
        embedded=len(all_metadata),
        failed=len(failed_files),
        refresh=True,
    )

pbar.close()

if device.type == "mps":
    torch.mps.empty_cache()

# ─────────────────────────────────────────────────────────────────────────────
# 7. SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
print(hdr("5 · Saving Outputs"))

if not all_embeddings:
    print(err("No embeddings extracted — check audio files and model"))
    sys.exit(1)

embedding_matrix = np.vstack(all_embeddings).astype(np.float32)
metadata_df      = pd.DataFrame(all_metadata)

print(sub(f"Embedding matrix shape : {embedding_matrix.shape}"))
print(sub(f"Metadata rows          : {len(metadata_df)}"))

np.save(OUT_EMBEDDINGS, embedding_matrix)
print(ok(f"Saved → {OUT_EMBEDDINGS}  ({OUT_EMBEDDINGS.stat().st_size / 1024**2:.1f} MB)"))

metadata_df.to_csv(OUT_METADATA, index=False)
print(ok(f"Saved → {OUT_METADATA}  ({len(metadata_df)} rows)"))

# ─────────────────────────────────────────────────────────────────────────────
# 8. SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────
print(hdr("6 · Summary"))
print(sub(f"Total files found     : {len(wav_files):,}"))
print(sub(f"Successfully embedded : {len(metadata_df):,}"))
print(sub(f"Failed / skipped      : {len(failed_files)}"))

if failed_files:
    print(f"\n{YELLOW}  Failed files:{RESET}")
    for f in failed_files[:10]:
        print(f"    {RED}•{RESET} {f}")
    if len(failed_files) > 10:
        print(f"    … and {len(failed_files) - 10} more")

if "Emotion_Label" in metadata_df.columns:
    print(f"\n{BOLD}  Emotion distribution:{RESET}")
    counts = metadata_df["Emotion_Label"].value_counts()
    total  = counts.sum()
    for label, cnt in counts.items():
        bar = "█" * int(cnt / total * 28)
        pct = cnt / total * 100
        print(f"    {label:<12}  {cnt:>4}  {pct:5.1f}%  {CYAN}{bar}{RESET}")

print(f"\n{BOLD}  Embedding stats:{RESET}")
print(sub(f"dtype     : {embedding_matrix.dtype}"))
print(sub(f"min / max : {embedding_matrix.min():.4f}  /  {embedding_matrix.max():.4f}"))
print(sub(f"mean      : {embedding_matrix.mean():.4f}"))
print(sub(f"std       : {embedding_matrix.std():.4f}"))

print(f"\n{BOLD}{GREEN}{'═'*62}")
print("  Extraction complete.")
print(f"{'═'*62}{RESET}\n")
