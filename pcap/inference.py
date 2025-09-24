#!/usr/bin/env python3
import argparse
import csv
import glob
import os
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------------------------- CLI / CONFIG --------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Run inference on Text/Label CSV using latest checkpoint."
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to input CSV with columns: Text, Label",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Optional explicit checkpoint .pt file. If omitted, picks most recent from --checkpoint-dir.",
    )
    p.add_argument(
        "--checkpoint-dir",
        default="./checkpoints",
        help="Directory to search for most recent checkpoint when --checkpoint is not given.",
    )
    p.add_argument(
        "--output-dir",
        default="./inference-result",
        help="Base directory for outputs.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size.",
    )
    p.add_argument(
        "--model-name",
        default="distilbert-base-uncased",
        help="HF model name used at train time (tokenizer + base weights).",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Tokenizer max_length used at train time.",
    )
    p.add_argument(
        "--strict-benign",
        action="store_true",
        help='If set, uses strict equality for "Benign"; otherwise case-insensitive equality (default).',
    )
    return p.parse_args()

# --------------------------- HELPERS --------------------------- #
def pick_latest_checkpoint(ckpt_dir: str) -> str | None:
    pattern = os.path.join(ckpt_dir, "checkpoint-*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    # Sort by modified time (newest last)
    files.sort(key=lambda f: os.path.getmtime(f))
    return files[-1]

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def binarize_label(raw: str, strict: bool = False) -> str:
    """Return 'Benign' or 'Anomaly'."""
    if raw is None:
        return "Anomaly"
    s = str(raw).strip()
    if strict:
        return "Benign" if s == "Benign" else "Anomaly"
    # case-insensitive equality
    return "Benign" if s.lower() == "benign" else "Anomaly"

# --------------------------- DATA --------------------------- #
def load_rows(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))

def batchify(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

# --------------------------- MAIN --------------------------- #
def main():
    args = parse_args()

    # Device / model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    ).to(device)

    # Resolve checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = pick_latest_checkpoint(args.checkpoint_dir)

    if ckpt_path and os.path.isfile(ckpt_path):
        print(f"[info] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        # Support both plain state_dict or dict with 'model_state_dict'
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
    else:
        print("[warn] No checkpoint found; running with base model weights.")

    model.eval()
    torch.backends.cudnn.benchmark = True

    # Load data
    rows = load_rows(args.input)
    if not rows:
        raise RuntimeError(f"No rows found in: {args.input}")

    # Expect columns Text / Label (case-insensitive).
    # Find them robustly:
    header = rows[0].keys()
    text_col = next((h for h in header if h.strip().lower() == "text"), None)
    label_col = next((h for h in header if h.strip().lower() == "label"), None)
    if text_col is None or label_col is None:
        raise ValueError(
            "Input CSV must contain 'Text' and 'Label' columns (case-insensitive)."
        )

    true_labels = []
    pred_labels = []
    prob_anomaly = []  # probability for the 'Anomaly' class (index 1)
    id_list = []

    # Inference loop
    for batch in tqdm(batchify(rows, args.batch_size), total=(len(rows) + args.batch_size - 1) // args.batch_size, desc=f"Batches(size={args.batch_size})"):
        texts = [str(r.get(text_col, "") or "") for r in batch]
        labels = [binarize_label(r.get(label_col, ""), strict=args.strict_benign) for r in batch]

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits  # [B, 2]
            probs = torch.softmax(logits, dim=1)  # [B, 2]
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            p_anom = probs[:, 1].detach().cpu().numpy()

        # Map indices to labels (0=Benign, 1=Anomaly)
        pred_batch = ["Benign" if p == 0 else "Anomaly" for p in preds]

        true_labels.extend(labels)
        pred_labels.extend(pred_batch)
        prob_anomaly.extend(p_anom.tolist())
        id_list.extend(range(len(id_list), len(id_list) + len(batch)))

    # Outputs
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = Path(args.output_dir) / stamp
    ensure_dir(out_dir)

    # 1) Raw predictions CSV
    # Include index, true, pred, prob_anomaly for easy filtering later
    raw_csv = out_dir / "predictions.csv"
    with open(raw_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "true_label", "pred_label", "prob_anomaly"])
        for i, t, p, pr in zip(id_list, true_labels, pred_labels, prob_anomaly):
            w.writerow([i, t, p, f"{pr:.6f}"])

    # 2) Confusion matrices + report
    labels_order = ["Benign", "Anomaly"]
    cm = confusion_matrix(true_labels, pred_labels, labels=labels_order)
    cm_norm = confusion_matrix(true_labels, pred_labels, labels=labels_order, normalize="true")
    report = classification_report(true_labels, pred_labels, labels=labels_order, digits=4)

    # Save text summary
    summary_txt = out_dir / "summary.txt"
    with open(summary_txt, "w") as f:
        f.write("Labels order: " + ", ".join(labels_order) + "\n\n")
        f.write("Confusion Matrix (counts):\n")
        f.write(str(cm))
        f.write("\n\nConfusion Matrix (row-normalized):\n")
        f.write(np.array2string(cm_norm, precision=4))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    # Save CM as CSVs for quick loading elsewhere
    np.savetxt(out_dir / "confusion_matrix_counts.csv", cm, fmt="%d", delimiter=",")
    np.savetxt(out_dir / "confusion_matrix_normalized.csv", cm_norm, fmt="%.6f", delimiter=",")

    # 3) ROC-AUC (binary)
    # Convert true labels to 0/1 with 1 = Anomaly
    y_true = np.array([1 if t == "Anomaly" else 0 for t in true_labels])
    y_score = np.array(prob_anomaly)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")

    # 4) Optional: save a simple confusion-matrix heatmap (no fancy styling)
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(4, 4))
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix (counts)")
        plt.xticks([0, 1], labels_order, rotation=45, ha="right")
        plt.yticks([0, 1], labels_order)
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center")
        plt.tight_layout()
        fig.savefig(out_dir / "confusion_matrix_counts.png", dpi=160)
        plt.close(fig)
    except Exception as e:
        print(f"[warn] Could not save confusion-matrix image: {e}")

    # 5) Run info
    info_txt = out_dir / "info.txt"
    with open(info_txt, "w") as f:
        f.write(f"Date/time:     {stamp}\n")
        f.write(f"Input CSV:     {args.input}\n")
        f.write(f"Checkpoint:    {ckpt_path or '(base model, none found)'}\n")
        f.write(f"Model name:    {args.model_name}\n")
        f.write(f"Batch size:    {args.batch_size}\n")
        f.write(f"Max length:    {args.max_length}\n")
        f.write(f"Total samples: {len(rows)}\n")
        f.write(f"ROC-AUC:       {auc:.6f}\n")

    print("\nSaved:")
    print(f" • Predictions CSV        → {raw_csv}")
    print(f" • Summary (report/mats)  → {summary_txt}")
    print(f" • CM (counts CSV)        → {out_dir / 'confusion_matrix_counts.csv'}")
    print(f" • CM (normalized CSV)    → {out_dir / 'confusion_matrix_normalized.csv'}")
    print(f" • CM (counts PNG)        → {out_dir / 'confusion_matrix_counts.png'}")
    print(f" • Info                   → {info_txt}")

if __name__ == "__main__":
    main()
