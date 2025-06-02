import torch
import csv
import os
import glob

from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. CONFIG
MODEL_NAME       = "distilbert-base-uncased"
CHECKPOINT_DIR   = "./checkpoints"
CHECKPOINT_PATH  = None
CANDIDATE_LABELS = ["BENIGN", "MALICIOUS"]
# CSV_PATH         = "../dataset/preprocessed-data/test_split.csv"
CSV_PATH         = "/home/bibber/Downloads/CIC-IDS 2018/02-22-2018.csv"
OUTPUT_DIR       = "./inference-result"
LABEL_KEY        = "Label"
BATCH_SIZE       = 8

# 2. LOAD MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    num_labels=len(CANDIDATE_LABELS)
).to(device)

if CHECKPOINT_PATH is None and os.path.isdir(CHECKPOINT_DIR):
    all_ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint-epoch*.pt")))
    if all_ckpts:
        CHECKPOINT_PATH = all_ckpts[-1]

CHECKPOINT_PATH = './checkpoints/checkpoint-epoch1-20250515_060552.pt'
if CHECKPOINT_PATH:
    print(f"Loading from checkpoint {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
else:
    print("No checkpoint found, using base model")

model.eval()
torch.backends.cudnn.benchmark = True

# 3. LOAD DATA
def load_dataset(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

# 4. INFERENCE
def classify_batch(texts: list[str]) -> list[str]:
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    preds = torch.argmax(logits, dim=1).cpu().tolist()
    return [CANDIDATE_LABELS[p] for p in preds]

def evaluate(batch_size: int = 16):
    data = load_dataset(CSV_PATH)
    true_labels = []
    pred_labels = []
    
    # Inference loop
    for start_idx in tqdm(range(0, len(data), batch_size), desc=f"Batches (size={batch_size})"):
        batch = data[start_idx : start_idx + batch_size]

        texts, labels = [], []
        for row in batch:
            features = {k: v for k, v in row.items() if k.strip().upper() != LABEL_KEY.strip().upper()}
            text = "; ".join(f"\"{k.strip()}\":{v}" for k, v in features.items())
            texts.append(text)
            raw_label = row[LABEL_KEY].strip().upper()
            mapped_label = raw_label if raw_label == "BENIGN" else "MALICIOUS"
            labels.append(mapped_label)

        preds = classify_batch(texts)
        true_labels.extend(labels)
        pred_labels.extend([p.upper() for p in preds])

    # Prepare output directory
    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = os.path.join(OUTPUT_DIR, today)
    os.makedirs(out_dir, exist_ok=True)

    # 1) Raw predictions
    raw_path = os.path.join(out_dir, "raw_predictions.txt")
    with open(raw_path, "w") as fout:
        for i, (t, p) in enumerate(zip(true_labels, pred_labels)):
            fout.write(f"{i}: {t}-{p}\n")

    # 2) Summary statistics
    cm = confusion_matrix(true_labels, pred_labels, labels=CANDIDATE_LABELS)
    report = classification_report(true_labels, pred_labels, labels=CANDIDATE_LABELS, digits=4)
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as fout:
        fout.write("Confusion Matrix:\n")
        fout.write(str(cm))
        fout.write("\n\nClassification Report:\n")
        fout.write(report)

    # 3) Info file
    info_path = os.path.join(out_dir, "info.txt")
    with open(info_path, "w") as fout:
        fout.write(f"Date: {today}\n")
        fout.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
        fout.write(f"Model: {MODEL_NAME}\n")
        fout.write(f"Dataset: {CSV_PATH}\n")
        fout.write(f"Batch Size: {batch_size}\n")
        fout.write(f"Total Samples: {len(data)}\n")

    print(f"\nLogs saved to {out_dir}:")
    print(f" • Raw predictions  → {raw_path}")
    print(f" • Summary stats    → {summary_path}")
    print(f" • Run info         → {info_path}")

# 5. MAIN
if __name__ == "__main__":
    evaluate(batch_size=BATCH_SIZE)
