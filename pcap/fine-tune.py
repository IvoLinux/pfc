import argparse
import csv
import datetime
import glob
import math
import os
import select
import sys
import random

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup)

# ---------------------------------------------------------------------------- #
#                                    ARGS                                      #
# ---------------------------------------------------------------------------- #
# To force training to resume from a checkpoint
# py {file}.py --resume_from {checkpoint_name}
parser = argparse.ArgumentParser()
parser.add_argument(
    "--resume_from", type=str, default=None,
    help="path to checkpoint to resume from"
)
args = parser.parse_args()

# ---------------------------------------------------------------------------- #
#                                  SETTINGS                                    #
# ---------------------------------------------------------------------------- #
MODEL_NAME       = "distilbert-base-uncased"
CANDIDATE_LABELS = ["Benign", "Anomaly"]  # binary: anything not "Benign" => "Anomaly"
CSV_PATH         = "train_small.csv"
LABEL_KEY        = "Label"
BINARY_BENIGN    = "BENIGN"               # case-insensitive match
TEXT_KEY         = "Text"                 # new text column
BATCH_SIZE       = 8
NUM_EPOCHS       = 3
LEARNING_RATE    = 3e-5
WEIGHT_DECAY     = 0.01
WARMUP_RATIO     = 0.05    # 5% of dataset rows used as warmup
MAX_LENGTH       = 512
CHECKPOINT_DIR   = "./checkpoints"
LOG_DIR          = os.path.join(CHECKPOINT_DIR, "logs")
SEED             = 21023 + 21041

# Create directories and seed the RNG
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "checkpoint_log.txt")

# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)

# ---------------------------------------------------------------------------- #
#                             MODEL & TOKENIZER                                #
# ---------------------------------------------------------------------------- #
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(CANDIDATE_LABELS)
).to(device)

# ---------------------------------------------------------------------------- #
#                             LOAD CSV RECORDS (ROW=1 SAMPLE)                  #
# ---------------------------------------------------------------------------- #
# We parse CSV rows. Each row's quoted Text cell may span lines.
# This keeps one sample per CSV record, regardless of how many newlines it has.
import csv as _csv

_csv.field_size_limit(sys.maxsize)  # allow very large cells

def load_csv_records(csv_path, text_col=TEXT_KEY, label_col=LABEL_KEY, encoding="utf-8"):
    texts, labels = [], []
    with open(csv_path, "r", encoding=encoding, newline="") as f:
        reader = _csv.DictReader(f)
        # normalize header names once
        header_map = {c.strip().lower(): c for c in (reader.fieldnames or [])}
        tcol = header_map.get(text_col.strip().lower(), text_col)
        lcol = header_map.get(label_col.strip().lower(), label_col)

        for row in reader:
            text  = row.get(tcol, "") or ""
            label = row.get(lcol, "") or ""
            texts.append(text)
            labels.append(label)
    return texts, labels

# Load once, then treat each CSV row as one training sample
texts, labels = load_csv_records(CSV_PATH, TEXT_KEY, LABEL_KEY)

# Row-count now means sample-count
total_samples     = len(texts)
batches_per_epoch = math.ceil(total_samples / BATCH_SIZE)
total_steps       = batches_per_epoch * NUM_EPOCHS
warmup_steps      = int(total_steps * WARMUP_RATIO)

# ---------------------------------------------------------------------------- #
#                        DETERMINE RESUME POINT (if any)                       #
# ---------------------------------------------------------------------------- #
# Find latest checkpoint in CHECKPOINT_DIR
ckpt_paths = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint-*.pt"))
latest_ckpt = args.resume_from or (max(ckpt_paths, key=os.path.getmtime) if ckpt_paths else None)

start_epoch = 0
start_batch = None
resumed_epoch_loss = 0.0
resumed_batches    = 0

# Setup variables to resume from checkpoint
if latest_ckpt:
    print(f"[resuming] loading {latest_ckpt}")
    ckpt = torch.load(latest_ckpt, map_location=device)
    epoch_ckpt = ckpt["epoch"]
    batch_ckpt = ckpt.get("batch_idx")
    total_steps  = ckpt.get("total_steps", total_steps)
    warmup_steps = ckpt.get("warmup_steps", warmup_steps)
    if batch_ckpt is None:
        start_epoch = epoch_ckpt
    else:
        start_epoch = epoch_ckpt - 1
        start_batch = batch_ckpt
    resumed_epoch_loss = ckpt.get("epoch_loss", 0.0)
    resumed_batches    = ckpt.get("batches_done", 0)

# ---------------------------------------------------------------------------- #
#                            OPTIMIZER & SCHEDULER                             #
# ---------------------------------------------------------------------------- #
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)
scaler = GradScaler()

if latest_ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    print(f"[resumed] epoch={start_epoch}, batch={start_batch}")
else:
    print(f"[started] with no checkpoints loaded")

# ---------------------------------------------------------------------------- #
#                          MAP-STYLE DATASET DEFINITION                        #
# ---------------------------------------------------------------------------- #
class InMemoryCSVDataset(Dataset):
    """
    Each item is one CSV record: the whole 'Text' field (possibly multi-line)
    and its label. No file seeking; parsing happened once up-front.
    """
    def __init__(self, texts, labels, tokenizer, max_length, indices):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length= max_length
        self.indices   = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        t = self.texts[i]
        y_raw = self.labels[i]
        # Binary mapping: 0 if label contains "BENIGN" (case-insensitive), else 1
        y = 0 if BINARY_BENIGN in str(y_raw).strip().upper() else 1

        tokens = self.tokenizer(
            t,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels":         torch.tensor(y, dtype=torch.long),
        }

# ---------------------------------------------------------------------------- #
#                            CHECKPOINT SAVER                                  #
# ---------------------------------------------------------------------------- #
def save_checkpoint(epoch, batch_idx=None, loss_val=None, epoch_loss=None, batches_done=None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"checkpoint-epoch{epoch+1}"
    if batch_idx is not None:
        fname += f"-batch{batch_idx+1}"
    fname += f"-{now}.pt"
    path = os.path.join(CHECKPOINT_DIR, fname)

    try:
        torch.save({
            "epoch":                 epoch+1,
            "batch_idx":             batch_idx,
            "model_state_dict":      model.state_dict(),
            "optimizer_state_dict":  optimizer.state_dict(),
            "scheduler_state_dict":  scheduler.state_dict(),
            "scaler_state_dict":     scaler.state_dict(),
            "epoch_loss":            epoch_loss,
            "batches_done":          batches_done,
            "total_steps":           total_steps,
            "warmup_steps":          warmup_steps,
        }, path)

        with open(LOG_FILE, "a") as log:
            line = f"{datetime.datetime.now().isoformat()} | epoch {epoch+1}"
            if batch_idx is not None:
                line += f" | batch {batch_idx+1}"
            if loss_val is not None:
                line += f" | loss {loss_val:.4f}"
            log.write(line + "\n")

        print(f"[checkpoint saved] {path}")

    except Exception as e:
        print(f"[⚠️ checkpoint failed] {e}", file=sys.stderr)

# ---------------------------------------------------------------------------- #
#                              TRAINING LOOP                                   #
# ---------------------------------------------------------------------------- #
for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()

    # Deterministically shuffle rows for this epoch
    random.seed(SEED + epoch)
    indices = list(range(total_samples))
    random.shuffle(indices)

    # If we're resuming mid‐epoch, drop the already‐processed examples
    if epoch == start_epoch and start_batch is not None:
        skip = (start_batch + 1) * BATCH_SIZE
        indices = indices[skip:]
        epoch_loss     = resumed_epoch_loss
        batches_offset = resumed_batches
    else:
        epoch_loss     = 0.0
        batches_offset = 0
    # Build DataLoader with our offset‐indexed CSV
    dataset = InMemoryCSVDataset(
        texts      = texts,
        labels     = labels,
        tokenizer  = tokenizer,
        max_length = MAX_LENGTH,
        indices    = indices,
    )
    loader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,   # already shuffled via `indices`
        num_workers = 4,
        pin_memory  = True,
        persistent_workers = True,
    )
    steps_in_epoch = math.ceil(len(indices) / BATCH_SIZE)
    pbar = tqdm(loader, total=steps_in_epoch, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    batches_since = 0
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        inputs = {k: v.to(device) for k, v in batch.items()}
        with autocast(device_type=device.type):
            outputs = model(**inputs)
            loss    = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_val      = loss.item()
        epoch_loss   += loss_val
        batches_since = batch_idx + 1
        total_batches = batches_offset + batches_since
        avg_loss      = epoch_loss / total_batches
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        # On-demand save from typing 's' or 'c' + 'Enter' in the terminal while training
        # 's' just saves the checkpoint, 'c' saves and exits
        if select.select([sys.stdin], [], [], 0)[0]:
            cmd = sys.stdin.readline().strip().lower()
            current_batch = (start_batch + batch_idx) if (epoch == start_epoch and start_batch is not None) else batch_idx
            if cmd == "s":
                save_checkpoint(epoch, current_batch, avg_loss, epoch_loss, total_batches)
            elif cmd == "c":
                print("Saving and Exiting.")
                save_checkpoint(epoch, current_batch, avg_loss, epoch_loss, total_batches)
                sys.exit(0)

    # End‐of‐epoch
    if batches_since:
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
    save_checkpoint(epoch, None, avg_loss, epoch_loss, total_batches)

print("Training complete.")