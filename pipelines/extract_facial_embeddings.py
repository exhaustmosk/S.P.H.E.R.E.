"""
extract_facial_embeddings.py
────────────────────────────────────────────────────────────────────────────
Facial Feature Embedding Extraction Pipeline
  Model  : dima806/facial_emotions_image_detection  (HuggingFace ViT)
           Architecture : ViT-base  →  768-dim pooler_output (CLS token)
  Device : Apple Silicon MPS → CPU fallback
  Input  : 48×48 grayscale PNGs under ./Extracted_images/<ClassName>/
  Output : facial_embeddings.npy  — (N, 768) float32
            facial_metadata.csv   — File_Path, Class_Label, Class_Index

  NOTE: The classifier head (classifier.*) keys will appear as UNEXPECTED
  in the load report — safe to ignore; we use the encoder pooler_output only.
────────────────────────────────────────────────────────────────────────────
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, ViTForImageClassification

warnings.filterwarnings("ignore", category=UserWarning)

# ── 1. CONFIGURATION ─────────────────────────────────────────────────────────
IMAGE_ROOT     = Path("./Extracted_images")       # class folders live here
BATCH_SIZE     = 32
CACHE_EVERY_N  = 20                               # mps.empty_cache() interval
NUM_WORKERS    = 0                                # must be 0 for MPS safety
OUT_EMBEDDINGS = Path("facial_embeddings.npy")
OUT_METADATA   = Path("facial_metadata.csv")
HF_MODEL_ID    = "dima806/facial_emotions_image_detection"

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
mps_available = torch.backends.mps.is_available()
device = torch.device("mps" if mps_available else "cpu")
print(ok(f"Device         : {BOLD}{device}{RESET}"))
print(sub(f"mps.is_built() : {torch.backends.mps.is_built()}"))
print(sub(f"mps.is_avail() : {mps_available}"))

# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAD MODEL & FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────
print(hdr("2 · Loading HuggingFace ViT Model"))
print(sub(f"Model ID : {HF_MODEL_ID}"))
print(sub("Downloading / loading from cache …"))

t0 = time.time()
processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID)
model = ViTForImageClassification.from_pretrained(HF_MODEL_ID)
model = model.to(device)
model.eval()

print(ok(f"Model loaded & moved to '{device}' in {time.time()-t0:.1f}s"))
print(sub(f"Parameters     : {sum(p.numel() for p in model.parameters()):,}"))
print(sub(f"Image size     : {processor.size}"))

# Determine embedding dim from CLS token (pooler_output is None for this model)
with torch.no_grad():
    _dummy = torch.zeros(1, 3, 224, 224, device=device)
    _emb_dim = model.vit(_dummy).last_hidden_state[:, 0, :].shape[-1]
print(sub(f"Embedding dim  : {_emb_dim}  (CLS token — last_hidden_state[:, 0, :])"))
del _dummy

# ─────────────────────────────────────────────────────────────────────────────
# 4. DATASET
# ─────────────────────────────────────────────────────────────────────────────
print(hdr("3 · Building Dataset"))

if not IMAGE_ROOT.exists():
    print(err(f"Image root not found: {IMAGE_ROOT.resolve()}"))
    sys.exit(1)

# Collect all image paths + class labels
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
class_dirs = sorted([p for p in IMAGE_ROOT.iterdir() if p.is_dir()])
class_names = [d.name for d in class_dirs]
class_to_idx = {name: i for i, name in enumerate(class_names)}

all_image_paths: list[Path] = []
all_class_labels: list[str] = []
all_class_indices: list[int] = []

for cls_dir in class_dirs:
    imgs = sorted([f for f in cls_dir.iterdir() if f.suffix.lower() in IMG_EXTS])
    all_image_paths.extend(imgs)
    all_class_labels.extend([cls_dir.name] * len(imgs))
    all_class_indices.extend([class_to_idx[cls_dir.name]] * len(imgs))

print(ok(f"Total images found   : {len(all_image_paths):,}"))
print(f"\n{BOLD}  Class distribution:{RESET}")
for cls in class_names:
    cnt = all_class_labels.count(cls)
    bar = "█" * int(cnt / len(all_class_labels) * 28)
    pct = cnt / len(all_class_labels) * 100
    print(f"    {cls:<12}  {cnt:>5}  {pct:5.1f}%  {CYAN}{bar}{RESET}")


class FacialImageDataset(Dataset):
    """
    Loads 48×48 grayscale PNGs, converts to 3-channel RGB,
    and applies AutoImageProcessor preprocessing.
    """

    def __init__(
        self,
        image_paths: list[Path],
        class_labels: list[str],
        class_indices: list[int],
        processor: AutoImageProcessor,
    ):
        self.paths    = image_paths
        self.labels   = class_labels
        self.indices  = class_indices
        self.processor = processor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]
        try:
            img = Image.open(path)
            # Grayscale (L) or grayscale+alpha → RGB
            if img.mode != "RGB":
                img = img.convert("RGB")

            inputs = self.processor(images=img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)   # (3, H, W)

        except Exception as exc:
            pixel_values = torch.zeros(3, 224, 224)
            print(f"\n{warn(f'Failed to load {path.name}: {exc}')}")

        return {
            "pixel_values": pixel_values,
            "path":         str(path),
            "label":        self.labels[idx],
            "class_idx":    self.indices[idx],
        }


dataset = FacialImageDataset(
    all_image_paths, all_class_labels, all_class_indices, processor
)

def collate_fn(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "paths":        [b["path"] for b in batch],
        "labels":       [b["label"] for b in batch],
        "class_indices":[b["class_idx"] for b in batch],
    }

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,   # 0 = main process only (required for MPS)
    collate_fn=collate_fn,
    pin_memory=False,          # pin_memory unsupported on MPS
)

print(ok(f"DataLoader ready     : {len(dataloader)} batches of {BATCH_SIZE}"))

# ─────────────────────────────────────────────────────────────────────────────
# 5. EXTRACTION LOOP
# ─────────────────────────────────────────────────────────────────────────────
print(hdr("4 · Extracting Embeddings"))
print(sub(f"Batch size    : {BATCH_SIZE}"))
print(sub(f"Cache flush   : every {CACHE_EVERY_N} batches"))
print()

from tqdm import tqdm

all_embeddings: list[np.ndarray] = []
all_metadata:   list[dict]       = []
failed_count = 0

pbar = tqdm(
    enumerate(dataloader),
    total=len(dataloader),
    desc="Embedding",
    unit="batch",
    dynamic_ncols=True,
    colour="cyan",
)

for batch_idx, batch in pbar:
    pixel_values = batch["pixel_values"].to(device)   # (B, 3, H, W)

    try:
        with torch.no_grad():
            outputs = model.vit(pixel_values=pixel_values)
            # CLS token (index 0) — pooler_output is None for this model
            # last_hidden_state: (B, seq_len=197, 768)
            cls_emb = outputs.last_hidden_state[:, 0, :]   # (B, 768)

        embeddings_np = cls_emb.cpu().float().numpy()
        all_embeddings.append(embeddings_np)

        for i in range(len(batch["paths"])):
            all_metadata.append({
                "File_Path":   batch["paths"][i],
                "Class_Label": batch["labels"][i],
                "Class_Index": batch["class_indices"][i],
            })

    except Exception as exc:
        failed_count += len(batch["paths"])
        print(f"\n{err(f'Batch {batch_idx} failed: {exc}')}")
        continue

    # ── MPS memory management ─────────────────────────────────────────────
    if device.type == "mps" and (batch_idx + 1) % CACHE_EVERY_N == 0:
        torch.mps.empty_cache()

    pbar.set_postfix(
        embedded=len(all_metadata),
        failed=failed_count,
        refresh=True,
    )

pbar.close()

if device.type == "mps":
    torch.mps.empty_cache()

# ─────────────────────────────────────────────────────────────────────────────
# 6. SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
print(hdr("5 · Saving Outputs"))

if not all_embeddings:
    print(err("No embeddings extracted — check image files and model"))
    sys.exit(1)

embedding_matrix = np.vstack(all_embeddings).astype(np.float32)
metadata_df      = pd.DataFrame(all_metadata)

print(sub(f"Embedding matrix shape : {embedding_matrix.shape}"))
print(sub(f"Metadata rows          : {len(metadata_df)}"))

np.save(OUT_EMBEDDINGS, embedding_matrix)
size_mb = OUT_EMBEDDINGS.stat().st_size / (1024 ** 2)
print(ok(f"Saved → {OUT_EMBEDDINGS}  ({size_mb:.1f} MB)"))

metadata_df.to_csv(OUT_METADATA, index=False)
print(ok(f"Saved → {OUT_METADATA}  ({len(metadata_df)} rows)"))

# ─────────────────────────────────────────────────────────────────────────────
# 7. SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────
print(hdr("6 · Summary"))
print(sub(f"Total images found    : {len(all_image_paths):,}"))
print(sub(f"Successfully embedded : {len(metadata_df):,}"))
print(sub(f"Failed / skipped      : {failed_count}"))

if "Class_Label" in metadata_df.columns:
    print(f"\n{BOLD}  Class distribution in output:{RESET}")
    counts = metadata_df["Class_Label"].value_counts()
    total  = counts.sum()
    for label, cnt in counts.items():
        bar = "█" * int(cnt / total * 28)
        pct = cnt / total * 100
        print(f"    {label:<12}  {cnt:>5}  {pct:5.1f}%  {GREEN}{bar}{RESET}")

print(f"\n{BOLD}  Embedding stats:{RESET}")
print(sub(f"dtype     : {embedding_matrix.dtype}"))
print(sub(f"shape     : {embedding_matrix.shape}"))
print(sub(f"min / max : {embedding_matrix.min():.4f}  /  {embedding_matrix.max():.4f}"))
print(sub(f"mean      : {embedding_matrix.mean():.4f}"))
print(sub(f"std       : {embedding_matrix.std():.4f}"))

print(f"\n{BOLD}{GREEN}{'═'*62}")
print("  Extraction complete.")
print(f"{'═'*62}{RESET}\n")
