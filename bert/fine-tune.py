import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from tqdm.auto import tqdm
import glob
import csv
import os

# 1. CONFIGURATION
MODEL_NAME       = "distilbert-base-uncased"
CANDIDATE_LABELS = ["BENIGN", "MALICIOUS"]
CSV_PATH         = "../dataset/subset_cicids2017.csv"
LABEL_KEY        = " Label"
BATCH_SIZE       = 16
NUM_EPOCHS       = 5
LEARNING_RATE    = 5e-5
WEIGHT_DECAY     = 0.01
WARMUP_STEPS     = 100
MAX_LENGTH       = 256
CHECKPOINT_DIR   = "./checkpoints"

# 2. LOAD MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(CANDIDATE_LABELS),
).to(device)

# OPTIONAL: Freeze some of the DistilBERT layers to speed up training
# for name, param in model.distilbert.transformer.layer[:4].named_parameters():
#     param.requires_grad = False

# 3. DATASET + DATALOADER
class CICIDSDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=256):
        self.rows = list(csv.DictReader(open(csv_path)))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        # build text from all fields except label
        features = {k: v for k, v in row.items() if k != LABEL_KEY}
        text = "; ".join(f"\"{k.strip()}\":{v}" for k, v in features.items())
        raw = row[LABEL_KEY].strip().upper()
        label = 0 if raw == "BENIGN" else 1

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

train_dataset = CICIDSDataset(CSV_PATH, tokenizer, MAX_LENGTH)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 4. OPTIMIZER & SCHEDULER
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)
total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
)

# 5. TRAINING LOOP
scaler = GradScaler()
best_loss = float("inf")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

latest_ckpt = sorted(glob.glob("./checkpoints/checkpoint-epoch*.pt"))[-1] if os.path.isdir("./checkpoints") else None

start_epoch = 0
if latest_ckpt:
    print(f"Loading checkpoint {latest_ckpt}")
    ckpt = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt["epoch"]
    print(f"Resuming from epoch {start_epoch}")

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    for batch in pbar:
        optimizer.zero_grad()

        # Move inputs to device
        inputs = {k: v.to(device) for k, v in batch.items()}

        # Mixed precision forward
        with autocast(device_type=device.type):
            outputs = model(**inputs)
            loss = outputs.loss

        # backprop
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=epoch_loss / (pbar.n + 1))

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    # 6. CHECKPOINT
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint-epoch{epoch+1}.pt")
    torch.save({
        "epoch":     epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    # (Optional) run a quick validation here to decide early stopping
    # val_loss = evaluate_on_validation_set(...)
    # if val_loss < best_loss:
    #     best_loss = val_loss
    # else:
    #     print("Validation loss did not improve; consider early stopping.")
