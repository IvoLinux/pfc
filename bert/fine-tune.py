import argparse
import csv
import datetime
import glob
import math
import os
import select
import sys

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ---------------------------------------------------------------------------- #
#                                    ARGS                                      #
# ---------------------------------------------------------------------------- #
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
NUM_EPOCHS       = 5
LEARNING_RATE    = 5e-5
WEIGHT_DECAY     = 0.01
WARMUP_RATIO     = 0.05    # 5% of steps as warmup
MAX_LENGTH       = 256
CHECKPOINT_DIR   = "./checkpoints"
LOG_DIR          = os.path.join(CHECKPOINT_DIR, "logs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "checkpoint_log.txt")

# ---------------------------------------------------------------------------- #
#                               MODEL & TOKENIZER                              #
# ---------------------------------------------------------------------------- #
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(CANDIDATE_LABELS)
).to(device)

# ---------------------------------------------------------------------------- #
#                        DETERMINE RESUME POINT (if any)                       #
# ---------------------------------------------------------------------------- #
# Find latest checkpoint
ckpt_paths = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint-*.pt"))
latest_ckpt = args.resume_from or (max(ckpt_paths, key=os.path.getmtime) if ckpt_paths else None)

start_epoch = 0
start_batch = None

if latest_ckpt:
    print(f"[resuming] loading {latest_ckpt}")
    ckpt = torch.load(latest_ckpt, map_location=device)
    epoch_ckpt = ckpt["epoch"]
    batch_ckpt = ckpt.get("batch_idx")
    if batch_ckpt is None:
        start_epoch = epoch_ckpt
    else:
        start_epoch = epoch_ckpt - 1
    start_batch = batch_ckpt

# ---------------------------------------------------------------------------- #
#                             COUNT TOTAL ROWS                                 #
# ---------------------------------------------------------------------------- #
with open(CSV_PATH) as f:
    total_rows = sum(1 for _ in f) - 1    # minus header
batches_per_epoch = math.ceil(total_rows / BATCH_SIZE)
total_steps       = batches_per_epoch * NUM_EPOCHS
warmup_steps      = int(total_steps * WARMUP_RATIO)

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

# ---------------------------------------------------------------------------- #
#                        ITERABLE DATASET DEFINITION                           #
# ---------------------------------------------------------------------------- #
class CICIDSIterable(IterableDataset):
    def __init__(self, path, tokenizer, max_length, skiprows=0):
        self.path       = path
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.skiprows   = skiprows

    def __iter__(self):
        with open(self.path) as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if idx < self.skiprows:
                    continue
                # Build a pseudo-JSON string
                feats = {k: v for k, v in row.items() if k != LABEL_KEY}
                text = "; ".join(f"\"{k.strip()}\":{v}" for k, v in feats.items())
                lab  = row[LABEL_KEY].strip().upper()
                label = 0 if lab == "BENIGN" else 1

                tokens = self.tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                yield {
                    "input_ids":      tokens["input_ids"].squeeze(0),
                    "attention_mask": tokens["attention_mask"].squeeze(0),
                    "labels":         torch.tensor(label, dtype=torch.long),
                }

# ---------------------------------------------------------------------------- #
#                            CHECKPOINT SAVER                                  #
# ---------------------------------------------------------------------------- #
def save_checkpoint(epoch, batch_idx=None, loss_val=None):
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
    }, path)

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
    epoch_loss = 0.0

    # figure out how many rows/batches we’ve already done in this epoch
    if epoch == start_epoch and start_batch is not None:
        initial_batch = start_batch + 1
    else:
        initial_batch = 0

    skip = initial_batch * BATCH_SIZE


    ds = CICIDSIterable(CSV_PATH, tokenizer, MAX_LENGTH, skiprows=skip)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    pbar = tqdm(loader, total=batches_per_epoch, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", initial=initial_batch, leave=False)
    for batch_idx, batch in enumerate(pbar, start=initial_batch):
        optimizer.zero_grad()
        inputs = {k: v.to(device) for k, v in batch.items()}
        with autocast(device_type=device.type):
            outputs = model(**inputs)
            loss    = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        epoch_loss += loss.item()
        batches_this_run = batch_idx - initial_batch + 1
        avg_loss         = epoch_loss / batches_this_run
        global_step = epoch * batches_per_epoch + (batch_idx + 1)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        # On‐demand save
        if select.select([sys.stdin], [], [], 0)[0]:
            cmd = sys.stdin.readline().strip().lower()
            if cmd == "s":
                save_checkpoint(epoch, batch_idx, avg_loss)
            elif cmd == "c":
                print("Saving and Exiting.")
                save_checkpoint(epoch, batch_idx, avg_loss)
                sys.exit(0)

    # End‐of‐epoch
    print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
    save_checkpoint(epoch, batch_idx=None, loss_val=avg_loss)

print("Training complete.")
