"""
BERT fine-tune + DNN — unfreezes last 2 BERT layers + trains a DNN head.
Pipeline: raw text → tokenizer → partially-frozen BERT → mean-pool → DNN → output
Mixed precision (fp16) + gradient accumulation for speed + memory efficiency.
Dual LR: 2e-5 for BERT layers, 1e-3 for DNN head.
Checkpoints every epoch, appends stats to stats.jsonl after each epoch.
"""
import os
import json
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report
from tqdm import tqdm


# ── Config ───────────────────────────────────────────────────────────────────

CLEAN_DIR    = "./../Dataset_cleaned"
OUTPUT_DIR   = "./output"
SIZES        = ["medium"]
NUM_CLASSES  = 12
MAX_LEN      = 256
BATCH        = 8
ACCUM_STEPS  = 4        # effective batch = 32 * 4 = 128
EPOCHS       = 100
LR_HEAD      = 1e-3
LR_BERT      = 2e-5
WEIGHT_DECAY = 1e-4
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS  = 0  # Windows + Python 3.14 can't pickle closures for multiprocessing
USE_AMP      = DEVICE == "cuda"  # mixed precision only on GPU

print(f"Device: {DEVICE}")
print(f"Mixed precision: {USE_AMP}")
print(f"Effective batch size: {BATCH * ACCUM_STEPS}")
print(f"DataLoader workers: {NUM_WORKERS}")


# ── Dataset ──────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, texts, labels, label2id):
        self.texts  = texts
        self.labels = [label2id[l] for l in labels]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def make_collate(tokenizer):
    def collate_fn(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(list(texts), truncation=True, padding="max_length",
                        max_length=MAX_LEN, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"], torch.tensor(labels, dtype=torch.long)
    return collate_fn


# ── Model ────────────────────────────────────────────────────────────────────

class BertDNN(nn.Module):
    """BERT (last 2 layers unfrozen) → mean-pool → DNN head."""
    def __init__(self, num_classes, bert_name="google-bert/bert-base-uncased", dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)

        # Freeze all, then unfreeze last 2 encoder layers
        for p in self.bert.parameters():
            p.requires_grad = False
        for layer in self.bert.encoder.layer[-2:]:
            for p in layer.parameters():
                p.requires_grad = True

        self.head = nn.Sequential(
            nn.Linear(768, 1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        mask   = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        return self.head(pooled)


# ── Training ─────────────────────────────────────────────────────────────────

def train_one_size(size):
    run_name = f"{size}_bert_dnn_finetune"
    out_dir  = f"{OUTPUT_DIR}/{run_name}"
    os.makedirs(f"{out_dir}/checkpoints", exist_ok=True)

    stats_path = f"{out_dir}/stats.jsonl"  # append-mode stats file

    print(f"\n{'='*60}")
    print(f"  {run_name}")
    print(f"{'='*60}")

    # Load data
    print("  Loading data...")
    train_df = pd.read_parquet(f"{CLEAN_DIR}/Dataset_{size}/train/train.parquet")
    val_df   = pd.read_parquet(f"{CLEAN_DIR}/Dataset_{size}/val/val.parquet")
    test_df  = pd.read_parquet(f"{CLEAN_DIR}/Dataset_{size}/test/test.parquet")

    label2id = {l: i for i, l in enumerate(sorted(train_df["model"].unique()))}
    id2label = {i: l for l, i in label2id.items()}
    print(f"  Classes: {len(label2id)} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    tokenizer  = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    collate_fn = make_collate(tokenizer)

    train_ds = TextDataset(train_df["generation"].tolist(), train_df["model"].tolist(), label2id)
    val_ds   = TextDataset(val_df["generation"].tolist(),   val_df["model"].tolist(),   label2id)
    test_ds  = TextDataset(test_df["generation"].tolist(),  test_df["model"].tolist(),  label2id)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                              collate_fn=collate_fn, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False,
                              collate_fn=collate_fn, pin_memory=True)

    # Model + optimizer + scheduler + scaler
    model = BertDNN(num_classes=NUM_CLASSES).to(DEVICE)

    bert_params = [p for n, p in model.named_parameters() if p.requires_grad and "bert" in n]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "bert" not in n]

    optimizer = torch.optim.AdamW([
        {"params": bert_params, "lr": LR_BERT},
        {"params": head_params, "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    scaler    = GradScaler(enabled=USE_AMP)

    print(f"  Params: {sum(p.numel() for p in model.parameters()):,} total | "
          f"BERT unfrozen: {sum(p.numel() for p in bert_params):,} | "
          f"Head: {sum(p.numel() for p in head_params):,}")

    best_val_acc = 0.0

    # Clear stats file at start
    with open(stats_path, "w") as f:
        pass

    # ── Epoch loop ───────────────────────────────────────────────────────
    epoch_bar = tqdm(range(1, EPOCHS + 1), desc="  Epochs", unit="ep")
    for epoch in epoch_bar:
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        optimizer.zero_grad()

        batch_bar = tqdm(train_loader, desc=f"    ep{epoch:02d}", unit="batch", leave=False)
        for step, (input_ids, attn_mask, labels) in enumerate(batch_bar):
            input_ids = input_ids.to(DEVICE, non_blocking=True)
            attn_mask = attn_mask.to(DEVICE, non_blocking=True)
            labels    = labels.to(DEVICE, non_blocking=True)

            # Mixed precision forward
            with autocast("cuda", enabled=USE_AMP):
                out  = model(input_ids, attn_mask)
                loss = criterion(out, labels) / ACCUM_STEPS

            scaler.scale(loss).backward()

            total_loss += loss.item() * ACCUM_STEPS * len(labels)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += len(labels)

            # Accumulate then step
            if (step + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            batch_bar.set_postfix(loss=f"{loss.item() * ACCUM_STEPS:.4f}")

        # Leftover gradients
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        # ── Validation ───────────────────────────────────────────────────
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad(), autocast("cuda", enabled=USE_AMP):
            for input_ids, attn_mask, labels in val_loader:
                out = model(input_ids.to(DEVICE, non_blocking=True),
                            attn_mask.to(DEVICE, non_blocking=True))
                val_correct += (out.argmax(1) == labels.to(DEVICE, non_blocking=True)).sum().item()
                val_total   += len(labels)

        train_acc = correct / total
        val_acc   = val_correct / val_total
        avg_loss  = total_loss / total
        cur_lr    = scheduler.get_last_lr()[0]

        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", train=f"{train_acc:.3f}",
                              val=f"{val_acc:.3f}", lr=f"{cur_lr:.2e}")

        # ── Append stats to jsonl ────────────────────────────────────────
        epoch_stats = {
            "epoch": epoch, "loss": round(avg_loss, 4),
            "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4),
            "lr": round(cur_lr, 6), "best_val_acc": round(best_val_acc, 4),
        }
        with open(stats_path, "a") as f:
            f.write(json.dumps(epoch_stats) + "\n")

        # ── Best model ───────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(),
                         "best_val_acc": best_val_acc, "label2id": label2id},
                        f"{out_dir}/best_model.pt")
            tqdm.write(f"    [new best val_acc: {val_acc:.4f}]")

        # ── Checkpoint every epoch (overwrite) ───────────────────────────
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_acc": best_val_acc,
            "label2id": label2id,
        }, f"{out_dir}/checkpoints/checkpoint.pt")

    print("  Training complete.")

    # ── Test ─────────────────────────────────────────────────────────────
    print("\n--- Test Set Report ---")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad(), autocast("cuda", enabled=USE_AMP):
        for input_ids, attn_mask, labels in tqdm(test_loader, desc="  Testing", unit="batch", leave=False):
            out = model(input_ids.to(DEVICE, non_blocking=True),
                        attn_mask.to(DEVICE, non_blocking=True))
            all_preds.append(out.argmax(1).cpu())
            all_labels.append(labels)

    preds  = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    names  = [id2label[i] for i in range(len(id2label))]

    report_dict = classification_report(labels, preds, target_names=names, digits=3, output_dict=True)
    report_str  = classification_report(labels, preds, target_names=names, digits=3)
    print(report_str)

    # ── Save final ───────────────────────────────────────────────────────
    torch.save({"model_state_dict": model.state_dict(), "label2id": label2id},
               f"{out_dir}/latest_model.pt")

    # Append test results to stats file
    with open(stats_path, "a") as f:
        f.write(json.dumps({"test_accuracy": round(report_dict["accuracy"], 4),
                             "report": report_str}) + "\n")

    print(f"  Saved: latest_model.pt, best_model.pt, stats.jsonl → {out_dir}/")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for size in SIZES:
        train_one_size(size)
    print(f"\nDone.")
