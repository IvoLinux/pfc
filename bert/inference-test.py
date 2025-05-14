import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import csv
import os
import glob

# 1. CONFIG
MODEL_NAME       = "distilbert-base-uncased"
CHECKPOINT_DIR   = "./checkpoints"
CHECKPOINT_PATH  = None
CANDIDATE_LABELS = ["BENIGN", "MALICIOUS"]
CSV_PATH         = "../dataset/Monday-2017.csv"
LABEL_KEY        = "Label"
BATCH_SIZE       = 16

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
    
    # Detecting malicious -> Benign is a True Negative
    trueNegative = truePositive = falsePositive = falseNegative = total = 0

    # Iterate over data in slices of `BATCH_SIZE`
    for start_idx in tqdm(range(0, len(data), batch_size), desc=f"Batches (size={batch_size})"):
        batch = data[start_idx : start_idx + batch_size]

        # List of row data and the corresponding true labels (per batch)
        texts = []
        labels = []
        for row in batch:
            features = {k: v for k, v in row.items() if k.strip().upper() != LABEL_KEY.strip().upper()}
            text = "; ".join(f"\"{k}\":{v}" for k, v in features.items())
            texts.append(text)
            labels.append(row[LABEL_KEY].strip().upper())

        # Classify and measure the batch
        preds = classify_batch(texts)
        for pred, label in zip(preds, labels):
            pred = pred.upper()
            if (pred == "BENIGN" and label == "BENIGN"):
                trueNegative += 1
            if (pred == "MALICIOUS" and label != "BENIGN"):
                truePositive += 1
            if (pred == "MALICIOUS" and label == "BENIGN"):
                falsePositive += 1
            if (pred == "BENIGN" and label != "BENIGN"):
                falseNegative += 1
            total += 1
            
        if total % 6400 == 0:
            print(f"Partial Results: {total} samples")
            print(f"True Negative: {trueNegative} ({trueNegative/total:.2%})")
            print(f"True Positive: {truePositive} ({truePositive/total:.2%})")
            print(f"False Positive: {falsePositive} ({falsePositive/total:.2%})")
            print(f"False Negative: {falseNegative} ({falseNegative/total:.2%})")

    # Print final results
    print(f"Final Results: {total} samples")
    print(f"True Negative: {trueNegative} ({trueNegative/total:.2%})")
    print(f"True Positive: {truePositive} ({truePositive/total:.2%})")
    print(f"False Positive: {falsePositive} ({falsePositive/total:.2%})")
    print(f"False Negative: {falseNegative} ({falseNegative/total:.2%})")
    print(f"Accuracy: {(truePositive + trueNegative) / total:.2%}")
    print(f"Precision: {truePositive / (truePositive + falsePositive):.2%}")
    print(f"Recall: {truePositive / (truePositive + falseNegative):.2%}")
    print(f"F1 Score: {2 * (truePositive / (truePositive + falsePositive)) * (truePositive / (truePositive + falseNegative)) / ((truePositive / (truePositive + falsePositive)) + (truePositive / (truePositive + falseNegative))):.2%}")

# 5. MAIN
if __name__ == "__main__":
    evaluate(batch_size=BATCH_SIZE)
