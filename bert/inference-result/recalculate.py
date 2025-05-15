#!/usr/bin/env python3
import sys
from sklearn.metrics import confusion_matrix, classification_report

def main(raw_path: str):
    true_labels = []
    pred_labels = []

    with open(raw_path, "r") as f:
        for line in f:
            # Each line: "{idx}: {TRUE}-{PRED}\n"
            _, rest = line.strip().split(": ", 1)
            true, pred = rest.split("-", 1)
            true = true.upper()
            pred = pred.upper()
            # Collapse all non-BENIGN into MALICIOUS
            mapped_true = true if true == "BENIGN" else "MALICIOUS"
            true_labels.append(mapped_true)
            pred_labels.append(pred)

    labels = ["BENIGN", "MALICIOUS"]
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm, "\n")
    print("Classification Report:")
    print(classification_report(true_labels, pred_labels, labels=labels, digits=4))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: recalculate.py path/to/raw_predictions.txt")
    else:
        main(sys.argv[1])
