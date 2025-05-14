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
CANDIDATE_LABELS = ["BENIGN", "MALICIOUS"]
CSV_PATH         = "../dataset/subset_cicids2017.csv"
LABEL_KEY        = "Label"
BATCH_SIZE       = 8
NUM_EPOCHS       = 3
LEARNING_RATE    = 3e-5
WEIGHT_DECAY     = 0.01
WARMUP_RATIO     = 0.05    # 5% of dataset rows used as warmup
MAX_LENGTH       = 256
CHECKPOINT_DIR   = "./checkpoints"
LOG_DIR          = os.path.join(CHECKPOINT_DIR, "logs")

# Create directories and seed the RNG
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "checkpoint_log.txt")

SEED = 21023 + 21041
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ---------------------------------------------------------------------------- #
#                               MODEL & TOKENIZER                              #
# ---------------------------------------------------------------------------- #
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(CANDIDATE_LABELS)
).to(device)

# ---------------------------------------------------------------------------- #
#                           COUNT ROWS IN DATASET                              #
# ---------------------------------------------------------------------------- #
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    rows = list(reader)
total_rows       = len(rows)
batches_per_epoch = math.ceil(total_rows / BATCH_SIZE)
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
class CICIDSDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length, indices, fieldnames):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.fieldnames = fieldnames
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.rows[self.indices[idx]]
        # Build a dictionary {ATRBIUTE:VALUE} for each row
        feats = {k: v for k, v in row.items() if k.strip().upper() != LABEL_KEY.strip().upper()}
        text = "; ".join(f"\"{k.strip()}\":{v}" for k, v in feats.items())
        
        candidates = [
            h for h in self.fieldnames
            if LABEL_KEY.lower() in h.strip().lower()
        ]
        if not candidates:
            raise ValueError(f"No column matching '{LABEL_KEY}' in {self.fieldnames}")
        label_col = candidates[0]

        lab  = row[label_col].strip().upper()
        label = 0 if lab == "BENIGN" else 1

        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels":         torch.tensor(label, dtype=torch.long),
        }

# ---------------------------------------------------------------------------- #
#                            CHECKPOINT SAVER                                  #
# ---------------------------------------------------------------------------- #
def save_checkpoint(epoch, batch_idx=None, loss_val=None, epoch_loss=None, batches_done=None):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"checkpoint-epoch{epoch+1}"
    if batch_idx is not None:
        fname += f"-batch{batch_idx+1}"
    fname += f"-{now}.pt"
    path = os.path.join(CHECKPOINT_DIR, fname)

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

    # Log the checkpoint save
    with open(LOG_FILE, "a") as log:
        line = f"{datetime.datetime.now().isoformat()} | epoch {epoch+1}"
        if batch_idx is not None:
            line += f" | batch {batch_idx+1}"
        if loss_val is not None:
            line += f" | loss {loss_val:.4f}"
        log.write(line + "\n")

    print(f"[checkpoint saved] {path}")

# ---------------------------------------------------------------------------- #
#                            TRAINING LOOP                                     #
# ---------------------------------------------------------------------------- #
for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()

    # Decide indices and starting loss/batches
    if epoch == start_epoch and start_batch is not None:
        # Resume mid-epoch
        skip_items = (start_batch + 1) * BATCH_SIZE
        indices = list(range(skip_items, total_rows))
        shuffle_flag = False
        epoch_loss = resumed_epoch_loss
        batches_offset = resumed_batches
    else:
        # Fresh epoch
        indices = list(range(total_rows))
        shuffle_flag = (epoch > start_epoch)
        if shuffle_flag:
            random.shuffle(indices)
        epoch_loss = 0.0
        batches_offset = 0

    dataset = CICIDSDataset(rows, tokenizer, MAX_LENGTH, indices, reader.fieldnames)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    steps_in_epoch = math.ceil(len(indices) / BATCH_SIZE)
    batches_since = 0
    pbar = tqdm(loader, total=steps_in_epoch, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

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

        loss_val = loss.item()
        epoch_loss += loss_val
        batches_since = batch_idx + 1
        total_batches = batches_offset + batches_since
        avg_loss = epoch_loss / total_batches
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
