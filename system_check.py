"""
system_check.py
───────────────────────────────────────────────────────────────────────────────
Verification script for the H4H multimodal mental-health classification project.

Checks performed
────────────────
1.  Platform & Python environment info
2.  Apple Silicon MPS (Metal Performance Shaders) availability via PyTorch
3.  Global default device selection  (mps → cpu fallback)
4.  Local data-directory / file presence
5.  numerical_data.csv profiling  (shape, columns, target distributions)
───────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import platform
import textwrap
from pathlib import Path

# ── colour helpers ────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
MAGENTA= "\033[95m"
DIM    = "\033[2m"

def ok(msg: str)   -> str: return f"{GREEN}✔  {msg}{RESET}"
def warn(msg: str) -> str: return f"{YELLOW}⚠  {msg}{RESET}"
def err(msg: str)  -> str: return f"{RED}✘  {msg}{RESET}"
def hdr(msg: str)  -> str: return f"\n{BOLD}{CYAN}{'─'*60}\n  {msg}\n{'─'*60}{RESET}"
def sub(msg: str)  -> str: return f"{MAGENTA}  ▸ {msg}{RESET}"

# ─────────────────────────────────────────────────────────────────────────────
# 1.  SYSTEM / ENVIRONMENT INFO
# ─────────────────────────────────────────────────────────────────────────────
print(hdr("1 · System & Environment"))

uname = platform.uname()
print(sub(f"OS            : {uname.system} {uname.release}  ({uname.machine})"))
print(sub(f"Node          : {uname.node}"))
print(sub(f"Processor     : {uname.processor or uname.machine}"))
print(sub(f"Python        : {sys.version.split()[0]}  →  {sys.executable}"))

# ─────────────────────────────────────────────────────────────────────────────
# 2.  PyTorch import
# ─────────────────────────────────────────────────────────────────────────────
print(hdr("2 · PyTorch"))
try:
    import torch
    print(ok(f"PyTorch {torch.__version__} imported successfully"))
except ImportError as exc:
    print(err(f"PyTorch not found — {exc}"))
    print(warn("Install with:  pip install torch torchvision torchaudio"))
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  MPS AVAILABILITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
print(hdr("3 · Apple Silicon MPS Check"))

mps_built     = torch.backends.mps.is_built()
mps_available = torch.backends.mps.is_available()

print(sub(f"torch.backends.mps.is_built()     = {mps_built}"))
print(sub(f"torch.backends.mps.is_available() = {mps_available}"))

if mps_available:
    print(ok("MPS backend is AVAILABLE  — Apple Silicon GPU will be used"))
elif mps_built:
    print(warn("MPS is built but NOT available on this runtime "
               "(may need macOS 12.3+ and a physical Apple Silicon chip)"))
else:
    print(warn("MPS backend is NOT built in this PyTorch installation; "
               "falling back to CPU"))

# ─────────────────────────────────────────────────────────────────────────────
# 4.  GLOBAL DEFAULT DEVICE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
print(hdr("4 · Device Selection"))

device = torch.device("mps" if mps_available else "cpu")
print(ok(f"Global default device set to → {BOLD}{device}{RESET}"))

# Sanity-test: allocate a small tensor on the chosen device
try:
    _t = torch.zeros(3, 3, device=device)
    print(ok(f"Test tensor (3×3 zeros) allocated on '{device}' successfully"))
    del _t
except Exception as exc:
    print(err(f"Tensor allocation on '{device}' failed: {exc}"))

# ─────────────────────────────────────────────────────────────────────────────
# 5.  LOCAL DIRECTORY / FILE STRUCTURE CHECK
# ─────────────────────────────────────────────────────────────────────────────
print(hdr("5 · Data Directory Structure"))

BASE = Path(__file__).parent.resolve()

# Paths to verify — accept either naming variant
SPEECH_CANDIDATES  = [BASE / "data" / "speech", BASE / "Audios"]
FACIAL_CANDIDATES  = [BASE / "data" / "facial", BASE / "Extracted_images"]
CSV_FILE           = BASE / "numerical_data.csv"

# ── speech directory ──────────────────────────────────────────────────────────
speech_found = None
for candidate in SPEECH_CANDIDATES:
    if candidate.exists():
        speech_found = candidate
        break

if speech_found:
    wav_count = len(list(speech_found.rglob("*.wav")))
    print(ok(f"Speech dir     : {speech_found.relative_to(BASE)}  "
             f"({wav_count} .wav file{'s' if wav_count != 1 else ''})"))
else:
    checked = " | ".join(str(p.relative_to(BASE)) for p in SPEECH_CANDIDATES)
    print(err(f"Speech dir NOT found  (checked: {checked})"))

# ── facial directory ──────────────────────────────────────────────────────────
facial_found = next((p for p in FACIAL_CANDIDATES if p.exists()), None)
if facial_found:
    img_count = sum(
        1 for f in facial_found.rglob("*")
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    )
    print(ok(f"Facial dir     : {facial_found.relative_to(BASE)}  "
             f"({img_count} image file{'s' if img_count != 1 else ''})"))
else:
    checked = " | ".join(str(p.relative_to(BASE)) for p in FACIAL_CANDIDATES)
    print(err(f"Facial dir NOT found  (checked: {checked})"))

# ── numerical CSV ─────────────────────────────────────────────────────────────
if CSV_FILE.exists():
    size_kb = CSV_FILE.stat().st_size / 1024
    print(ok(f"numerical_data.csv : found  ({size_kb:.1f} KB)"))
else:
    print(err(f"numerical_data.csv NOT found at {CSV_FILE.relative_to(BASE)}"))

# ─────────────────────────────────────────────────────────────────────────────
# 6.  CSV PROFILING
# ─────────────────────────────────────────────────────────────────────────────
print(hdr("6 · numerical_data.csv — DataFrame Profile"))

try:
    import pandas as pd
except ImportError:
    print(err("pandas not installed — pip install pandas"))
    sys.exit(1)

if not CSV_FILE.exists():
    print(err("Skipping CSV profiling — file not found"))
    sys.exit(1)

df = pd.read_csv(CSV_FILE)

# ── shape & dtypes ────────────────────────────────────────────────────────────
print(sub(f"Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns"))
print(sub(f"Memory usage   : {df.memory_usage(deep=True).sum() / 1024:.1f} KB"))

print(f"\n{BOLD}  Columns ({df.shape[1]}){RESET}")
for i, col in enumerate(df.columns, 1):
    dtype = str(df[col].dtype)
    nulls = df[col].isna().sum()
    null_str = f"{RED}  ({nulls} nulls){RESET}" if nulls else ""
    print(f"   {DIM}{i:>2}.{RESET}  {col:<35}  {DIM}{dtype}{RESET}{null_str}")

# ── target column distributions ──────────────────────────────────────────────
TARGET_COLS = ["Mental_Health_Status", "Depression_Score",
               "Anxiety_Score",        "Stress_Score"]

for col in TARGET_COLS:
    if col not in df.columns:
        print(warn(f"Column '{col}' not found in CSV — skipping"))
        continue

    print(f"\n{BOLD}  ▌ {col}{RESET}")

    if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
        # Categorical / string target
        counts = df[col].value_counts()
        total  = counts.sum()
        for label, cnt in counts.items():
            bar  = "█" * int(cnt / total * 30)
            pct  = cnt / total * 100
            print(f"    {label:<25}  {cnt:>5}  {pct:5.1f}%  {GREEN}{bar}{RESET}")
    else:
        # Continuous / score target
        print(f"    min={df[col].min():.2f}  "
              f"mean={df[col].mean():.2f}  "
              f"median={df[col].median():.2f}  "
              f"max={df[col].max():.2f}  "
              f"std={df[col].std():.2f}")

        # Bucket distribution
        bins   = [0, 10, 20, 30, 40, 100]
        labels = ["0–9", "10–19", "20–29", "30–39", "40+"]
        bucketed = pd.cut(df[col], bins=bins, labels=labels, right=False)
        bcounts  = bucketed.value_counts().sort_index()
        total    = bcounts.sum()
        for label, cnt in bcounts.items():
            bar = "█" * int(cnt / total * 25)
            pct = cnt / total * 100
            print(f"    {str(label):<10}  {cnt:>5}  {pct:5.1f}%  {CYAN}{bar}{RESET}")

# ── missing value summary ─────────────────────────────────────────────────────
total_nulls = df.isna().sum().sum()
print(f"\n{sub('Total missing values : ')}"
      + (f"{GREEN}0{RESET}" if total_nulls == 0 else f"{RED}{total_nulls}{RESET}"))

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{BOLD}{GREEN}{'═'*60}")
print("  All checks complete.")
print(f"{'═'*60}{RESET}\n")
