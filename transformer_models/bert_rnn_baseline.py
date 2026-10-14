"""
BERT + RNN classifier — tokenizes text on-the-fly, no pre-stored embeddings.
Pipeline: raw text → BERT tokenizer → frozen BERT → GRU → Linear → ReLU → output
Memory-friendly: processes one batch at a time through BERT, never stores full embeddings.
Includes checkpointing every 10 epochs + saves best/latest model + stats.
"""
import os
import json
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report
from tqdm import tqdm


# ── Config ───────────────────────────────────────────────────────────────────

CLEAN_DIR   = "./../Dataset_cleaned"
OUTPUT_DIR  = "./output"
SIZES       = ["small"]
NUM_CLASSES = 12
MAX_LEN     = 256
BATCH       = 16
EPOCHS      = 50
LR          = 1e-3
CKPT_EVERY  = 10
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")


# ── Dataset ──────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """Holds raw texts + labels. Tokenization happens in collate_fn."""
    def __init__(self, texts: list, labels: list, label2id: dict):
        self.texts  = texts
        self.labels = [label2id[l] for l in labels]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def make_collate(tokenizer):
    """Returns a collate_fn that tokenizes a batch of texts on-the-fly."""
    def collate_fn(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts), truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"], torch.tensor(labels, dtype=torch.long)
    return collate_fn


# ── Model: frozen BERT + GRU head ───────────────────────────────────────────

class BertRNN(nn.Module):
    """
    Frozen BERT encoder → Bidirectional GRU → Linear → ReLU → output.
    BERT weights are NOT trained — only the GRU + FC head learn.
    """
    def __init__(self, num_classes: int, bert_name: str = "google-bert/bert-base-uncased",
                 gru_hidden: int = 256, gru_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        # Frozen BERT — extracts token embeddings without training
        self.bert = AutoModel.from_pretrained(bert_name)
        for p in self.bert.parameters():
            p.requires_grad = False

        # GRU reads BERT's (B, seq_len, 768) output
        self.gru = nn.GRU(
            input_size=768, hidden_size=gru_hidden,
            num_layers=gru_layers, batch_first=True,
            bidirectional=True, dropout=dropout if gru_layers > 1 else 0,
        )

        # Classification head
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden * 2, 256),  # bidirectional → *2
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        # BERT forward (no grad since frozen)
        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = bert_out.last_hidden_state  # (B, seq_len, 768)

        # GRU over token sequence
        gru_out, _ = self.gru(hidden)  # (B, seq_len, gru_hidden*2)
        last = gru_out[:, -1, :]       # last timestep

        return self.fc(last)


# ── Training loop ────────────────────────────────────────────────────────────

def train_one_size(size: str):
    run_name = f"{size}_bert_rnn"
    out_dir  = f"{OUTPUT_DIR}/{run_name}"
    os.makedirs(f"{out_dir}/checkpoints", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {run_name}")
    print(f"{'='*60}")

    # Load raw text data
    print("  Loading data...")
    train_df = pd.read_parquet(f"{CLEAN_DIR}/Dataset_{size}/train/train.parquet")
    val_df   = pd.read_parquet(f"{CLEAN_DIR}/Dataset_{size}/val/val.parquet")
    test_df  = pd.read_parquet(f"{CLEAN_DIR}/Dataset_{size}/test/test.parquet")

    # Build label vocab
    label2id = {l: i for i, l in enumerate(sorted(train_df["generated_by"].unique()))}
    id2label = {i: l for l, i in label2id.items()}
    print(f"  Classes: {len(label2id)}  |  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    # Tokenizer + collate
    tokenizer  = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    collate_fn = make_collate(tokenizer)

    # Datasets + loaders
    train_ds = TextDataset(train_df["text"].tolist(), train_df["generated_by"].tolist(), label2id)
    val_ds   = TextDataset(val_df["text"].tolist(),   val_df["generated_by"].tolist(),   label2id)
    test_ds  = TextDataset(test_df["text"].tolist(),  test_df["generated_by"].tolist(),  label2id)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, collate_fn=collate_fn)

    # Model
    model = BertRNN(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.CrossEntropyLoss()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_p:,} total | {trainable:,} trainable (BERT frozen)")

    best_val_acc = 0.0
    history = []

    # ── Epoch loop ───────────────────────────────────────────────────────
    epoch_bar = tqdm(range(1, EPOCHS + 1), desc="  Epochs", unit="ep")
    for epoch in epoch_bar:
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        batch_bar = tqdm(train_loader, desc=f"    ep{epoch:02d}", unit="batch", leave=False)
        for input_ids, attn_mask, labels in batch_bar:
            input_ids = input_ids.to(DEVICE)
            attn_mask = attn_mask.to(DEVICE)
            labels    = labels.to(DEVICE)

            optimizer.zero_grad()
            out  = model(input_ids, attn_mask)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += len(labels)
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for input_ids, attn_mask, labels in val_loader:
                out = model(input_ids.to(DEVICE), attn_mask.to(DEVICE))
                val_correct += (out.argmax(1) == labels.to(DEVICE)).sum().item()
                val_total   += len(labels)

        train_acc = correct / total
        val_acc   = val_correct / val_total
        avg_loss  = total_loss / total

        history.append({"epoch": epoch, "loss": round(avg_loss, 4),
                        "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4)})

        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", train_acc=f"{train_acc:.3f}", val_acc=f"{val_acc:.3f}")

        # Best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(),
                         "best_val_acc": best_val_acc, "label2id": label2id},
                        f"{out_dir}/best_model.pt")

        # Checkpoint (overwrite)
        if epoch % CKPT_EVERY == 0:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                         "optimizer_state_dict": optimizer.state_dict(),
                         "best_val_acc": best_val_acc, "label2id": label2id},
                        f"{out_dir}/checkpoints/checkpoint.pt")
            tqdm.write(f"    [checkpoint saved: epoch {epoch}]")

    print("  Training complete.")

    # ── Test evaluation ──────────────────────────────────────────────────
    print("\n--- Test Set Report ---")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, attn_mask, labels in tqdm(test_loader, desc="  Testing", unit="batch", leave=False):
            out = model(input_ids.to(DEVICE), attn_mask.to(DEVICE))
            all_preds.append(out.argmax(1).cpu())
            all_labels.append(labels)

    preds  = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    names  = [id2label[i] for i in range(len(id2label))]

    report_dict = classification_report(labels, preds, target_names=names, digits=3, output_dict=True)
    report_str  = classification_report(labels, preds, target_names=names, digits=3)
    print(report_str)

    # ── Save everything ──────────────────────────────────────────────────
    torch.save({"model_state_dict": model.state_dict(), "label2id": label2id},
               f"{out_dir}/latest_model.pt")

    stats = {
        "best_val_acc": round(best_val_acc, 4),
        "test_stats": {
            "test_accuracy": round(report_dict["accuracy"], 4),
            "per_class": {k: {m: round(v, 4) for m, v in report_dict[k].items()} for k in names},
            "report_text": report_str,
        },
        "training_history": history,
        "label2id": label2id,
    }
    with open(f"{out_dir}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Saved: latest_model.pt, best_model.pt, stats.json → {out_dir}/")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for size in SIZES:
        train_one_size(size)
    print(f"\nAll BERT+RNN runs complete.")
