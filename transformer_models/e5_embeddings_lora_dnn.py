"""
E5-base Embeddings + DNN Classifier for LLM Attribution
=========================================================
Pipeline:
  1. Load Dataset parquet files
  2. Encode texts with intfloat/e5-base (768-dim, frozen)
  3. Train DNN on top of frozen embeddings
  4. Checkpoint every epoch + sync stats to Drive so a crashed runtime loses nothing

Why E5-base?
  Trained with a contrastive (query, passage) objective — embeddings are more
  discriminative than plain BERT mean-pool. The "passage: " prefix is required
  by the model to signal document-mode encoding (vs query-mode).

Colab setup:
  1. Upload Dataset_cleaned/ to Google Drive
  2. Set DATA_ROOT and DRIVE_OUT below
  3. Runtime > Change runtime type > T4 GPU
  4. Run all cells (or just run this file directly)

Install (paste in a Colab cell and run first):
  !pip install -q sentence-transformers pandas pyarrow scikit-learn
"""

# ── Colab: mount Drive ────────────────────────────────────────────────────────
# Uncomment when running on Colab:
#
# from google.colab import drive
# drive.mount('/content/drive')

# ── Imports ───────────────────────────────────────────────────────────────────
import os, json, time, shutil
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# =============================================================================
# CONFIG — edit these two paths for Colab
# =============================================================================
DATA_ROOT = Path("../Dataset_cleaned")
# DATA_ROOT = Path("/content/drive/MyDrive/Dataset_cleaned")  # Colab

# All checkpoints + stats written here AND mirrored to DRIVE_OUT after each epoch
OUTPUT_DIR = Path("./output/e5_dnn")
DRIVE_OUT  = None   # set to e.g. Path("/content/drive/MyDrive/llm_classifier/e5_dnn")
# DRIVE_OUT  = Path("/content/drive/MyDrive/llm_classifier/e5_dnn")  # Colab

DATASET_SIZE = "small"    # "small" | "medium"
E5_MODEL     = "intfloat/e5-base"   # 768-dim frozen encoder

BATCH_SIZE   = 256
EPOCHS       = 60
LR           = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 10
DROPOUT      = 0.3
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if DRIVE_OUT:
    DRIVE_OUT.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def sync_to_drive(src: Path, filename: str):
    """Copy a file from local OUTPUT_DIR to DRIVE_OUT (if configured)."""
    if DRIVE_OUT is None:
        return
    dst = DRIVE_OUT / filename
    shutil.copy2(src, dst)


def append_epoch_stats(stats: dict):
    """
    Append one epoch's stats to stats.jsonl (one JSON object per line).
    Also syncs the file to Drive immediately so a runtime crash loses at
    most the current (incomplete) epoch.
    """
    stats_path = OUTPUT_DIR / "stats.jsonl"
    with open(stats_path, "a") as f:
        f.write(json.dumps(stats) + "\n")
    sync_to_drive(stats_path, "stats.jsonl")


# ── 1. Load data ──────────────────────────────────────────────────────────────

print(f"\nLoading {DATASET_SIZE} dataset...")
splits = {}
for s in ["train", "val", "test"]:
    path = DATA_ROOT / f"Dataset_{DATASET_SIZE}" / s / f"{s}.parquet"
    splits[s] = pd.read_parquet(path)
    print(f"  {s}: {len(splits[s])}")

le = LabelEncoder()
le.fit(sorted(pd.concat(splits.values())["generated_by"].unique()))
NUM_CLASSES = len(le.classes_)
print(f"  Classes ({NUM_CLASSES}): {list(le.classes_)}")


# ── 2. Generate E5 embeddings (one-time, then free GPU) ───────────────────────

def embed_texts(model, texts, batch_size=64):
    # E5 needs "passage: " prefix for document encoding
    return model.encode(
        ["passage: " + t for t in texts],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-norm so cosine sim == dot product
    )

print(f"\nLoading E5: {E5_MODEL}")
e5 = SentenceTransformer(E5_MODEL, device=DEVICE)

print("Generating embeddings...")
t0 = time.time()
train_emb = embed_texts(e5, splits["train"]["text"].tolist())
val_emb   = embed_texts(e5, splits["val"]["text"].tolist())
test_emb  = embed_texts(e5, splits["test"]["text"].tolist())
print(f"Done in {time.time()-t0:.1f}s | shape: {train_emb.shape}")

del e5
if DEVICE == "cuda":
    torch.cuda.empty_cache()

train_y = le.transform(splits["train"]["generated_by"])
val_y   = le.transform(splits["val"]["generated_by"])
test_y  = le.transform(splits["test"]["generated_by"])


# ── 3. Normalise embeddings ───────────────────────────────────────────────────
# Re-centres each of the 768 dims to zero mean / unit variance on train set.
# Prevents any single embedding dimension from dominating the first linear layer.

scaler    = StandardScaler()
train_emb = scaler.fit_transform(train_emb)
val_emb   = scaler.transform(val_emb)
test_emb  = scaler.transform(test_emb)


# ── 4. Tensor datasets ────────────────────────────────────────────────────────

def make_loader(emb, labels, shuffle):
    x = torch.tensor(emb,    dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    return DataLoader(TensorDataset(x, y),
                      batch_size=BATCH_SIZE if shuffle else BATCH_SIZE * 2,
                      shuffle=shuffle, num_workers=0)

train_dl = make_loader(train_emb, train_y, shuffle=True)
val_dl   = make_loader(val_emb,   val_y,   shuffle=False)
test_dl  = make_loader(test_emb,  test_y,  shuffle=False)


# ── 5. DNN ────────────────────────────────────────────────────────────────────
# 768 -> 1024 -> 512 -> 256 -> 128 -> num_classes
# First layer expands to 1024 — gives the model room to rotate the L2-normalised
# E5 vectors before compressing them into class logits.
# BatchNorm after each linear speeds convergence and reduces LR sensitivity.

class DNN(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 512),       nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),        nn.BatchNorm1d(256),  nn.ReLU(), nn.Dropout(dropout * 0.7),
            nn.Linear(256, 128),        nn.BatchNorm1d(128),  nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


model = DNN(train_emb.shape[1], NUM_CLASSES, DROPOUT).to(DEVICE)
print(f"\nDNN params: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# ── 6. Training loop ──────────────────────────────────────────────────────────

# Clear stats file at the start of a fresh run
(OUTPUT_DIR / "stats.jsonl").unlink(missing_ok=True)

best_val, patience_count = 0.0, 0
t0 = time.time()

print(f"\nTraining (max {EPOCHS} ep, patience={PATIENCE})...")
print(f"{'Ep':>4}  {'TrainLoss':>10}  {'TrainAcc':>9}  {'ValLoss':>8}  {'ValAcc':>8}  {'Best':>8}")
print("-" * 62)

for epoch in range(1, EPOCHS + 1):
    t_ep = time.time()

    # --- train ---
    model.train()
    train_loss, train_preds, train_trues = 0.0, [], []
    for x, y in train_dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss  += loss.item()
        train_preds += out.argmax(-1).cpu().tolist()
        train_trues += y.cpu().tolist()
    scheduler.step()

    train_loss /= len(train_dl)
    train_acc   = accuracy_score(train_trues, train_preds)

    # --- validate ---
    model.eval()
    val_loss, val_preds, val_trues = 0.0, [], []
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out      = model(x)
            val_loss += criterion(out, y).item()
            val_preds += out.argmax(-1).cpu().tolist()
            val_trues += y.cpu().tolist()
    val_loss /= len(val_dl)
    val_acc   = accuracy_score(val_trues, val_preds)

    is_best = val_acc > best_val
    if is_best:
        best_val       = val_acc
        patience_count = 0
    else:
        patience_count += 1

    ep_time = time.time() - t_ep

    # --- checkpoint every epoch ---
    # Saves full training state so training can be resumed after a crash
    ckpt = {
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val":        best_val,
        "patience_count":  patience_count,
    }
    ckpt_path = OUTPUT_DIR / "checkpoint_latest.pt"
    torch.save(ckpt, ckpt_path)
    sync_to_drive(ckpt_path, "checkpoint_latest.pt")

    # Separately track the best model
    if is_best:
        best_path = OUTPUT_DIR / "best_model.pt"
        torch.save(model.state_dict(), best_path)
        sync_to_drive(best_path, "best_model.pt")

    # --- per-epoch stats (append to .jsonl, sync to Drive) ---
    append_epoch_stats({
        "epoch":      epoch,
        "train_loss": round(train_loss, 5),
        "train_acc":  round(train_acc,  5),
        "val_loss":   round(val_loss,   5),
        "val_acc":    round(val_acc,    5),
        "best_val":   round(best_val,   5),
        "lr":         round(scheduler.get_last_lr()[0], 7),
        "epoch_time": round(ep_time, 2),
        "elapsed":    round(time.time() - t0, 2),
        "is_best":    is_best,
    })

    marker = " <-- best" if is_best else ""
    print(f"{epoch:>4}  {train_loss:>10.4f}  {train_acc:>9.4f}"
          f"  {val_loss:>8.4f}  {val_acc:>8.4f}  {best_val:>8.4f}{marker}")

    if patience_count >= PATIENCE:
        print(f"Early stop.")
        break


# ── 7. Test ───────────────────────────────────────────────────────────────────

print("\nLoading best model for test evaluation...")
model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pt", map_location=DEVICE))
model.eval()

test_preds, test_trues = [], []
with torch.no_grad():
    for x, y in test_dl:
        test_preds += model(x.to(DEVICE)).argmax(-1).cpu().tolist()
        test_trues += y.tolist()

test_acc = accuracy_score(test_trues, test_preds)
report   = classification_report(test_trues, test_preds,
                                  target_names=le.classes_, digits=3)

print(f"\n{'='*52}")
print(f"  TEST ACCURACY: {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"{'='*52}")
print(report)


# ── 8. Save final results ─────────────────────────────────────────────────────

results = {
    "model":      E5_MODEL,
    "dataset":    DATASET_SIZE,
    "test_acc":   round(test_acc, 4),
    "best_val":   round(best_val, 4),
    "classes":    list(le.classes_),
    "report":     classification_report(test_trues, test_preds,
                      target_names=le.classes_, output_dict=True),
}
results_path = OUTPUT_DIR / "results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
sync_to_drive(results_path, "results.json")

print(f"\nAll files saved to:  {OUTPUT_DIR}/")
if DRIVE_OUT:
    print(f"Mirrored to Drive:   {DRIVE_OUT}/")
print("Files: best_model.pt  checkpoint_latest.pt  stats.jsonl  results.json")
