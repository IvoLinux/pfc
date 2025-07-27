import os, csv, math, datetime, random, sys, time, json, warnings
from typing import Tuple, Dict, List, Callable
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.exceptions import UndefinedMetricWarning


# hard‑coded for now
CANDIDATE_LABELS = [
    "Benign", "Botnet", "Brute_Force_Attack",
    "DoS_Attack", "Port_Scan_Infiltration",
    "Web_Attack", "Other"
]


class FineTuner:
    def __init__(
        self,
        csv_path: str,
        model_name: str,
        output_root: str,
        update_progress_cb: Callable[[Dict], None],
        num_epochs: int,
        max_length: int,
    ):
        # settings (keep your old constants here or parameterize later)
        self.csv_path      = csv_path
        self.model_name    = model_name
        self.output_root   = output_root
        self.num_epochs    = num_epochs
        self.max_length    = max_length
        self.batch_size    = 8
        self.learning_rate = 3e-5
        self.weight_decay  = 0.01
        self.warmup_ratio  = 0.05
        self.seed          = 21023 + 21041

        # checkpoint & log dirs
        self.ckpt_dir = output_root
        self.log_dir  = os.path.join(self.ckpt_dir, "logs")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir,  exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "checkpoint_log.txt")

        # progress callback
        self.update_progress_cb = update_progress_cb

        # control flags
        self._save_request = False
        self._stop_request = False

        # device, tokenizer, model
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model     = (
            AutoModelForSequenceClassification
            .from_pretrained(self.model_name, num_labels=len(CANDIDATE_LABELS))
            .to(self.device)
        )

        # build CSV offset index
        with open(self.csv_path, "r", newline="") as f:
            header_line = f.readline()
            self.header = next(csv.reader([header_line]))
            self.offsets = []
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                self.offsets.append(pos)

        self.total_rows        = len(self.offsets)
        self.batches_per_epoch = math.ceil(self.total_rows / self.batch_size)
        self.total_steps       = self.batches_per_epoch * self.num_epochs
        self.warmup_steps      = int(self.total_steps * self.warmup_ratio)

    def train(self) -> Tuple[str, Dict[str, float], List[List[int]]]:
        # prepare optimizer, scheduler, scaler
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scaler = GradScaler()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler    = scaler

        history: List[List[int]] = []
        metrics: Dict[str, float] = {}

        self.start_time = None
        for epoch in range(self.num_epochs):
            self.model.train()
            # deterministic shuffle
            random.seed(self.seed + epoch)
            indices = list(range(self.total_rows))
            random.shuffle(indices)

            epoch_loss = 0.0
            batches_done = 0

            # DataLoader
            dataset = IndexedCSV(
                csv_path   = self.csv_path,
                offsets    = self.offsets,
                header     = self.header,
                tokenizer  = self.tokenizer,
                max_length = self.max_length,
                indices    = indices,
            )
            loader = DataLoader(
                dataset,
                batch_size  = self.batch_size,
                shuffle     = False,
                num_workers = 4,
                pin_memory  = True,
            )

            if self.start_time is None:
                self.start_time = time.perf_counter()
            pbar = tqdm(loader, total=len(loader), desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for batch_idx, batch in enumerate(pbar):    
                optimizer.zero_grad()
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                with autocast(device_type=self.device.type):
                    outputs = self.model(**inputs)
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                loss_val    = loss.item()
                epoch_loss += loss_val
                batches_done += 1
                avg_loss    = epoch_loss / batches_done

                # progress callback
                step     = epoch * self.batches_per_epoch + (batch_idx + 1)
                progress = step / self.total_steps * 100
                elapsed  = time.perf_counter() - self.start_time
                eta      = (elapsed / step) * (self.total_steps - step) if step > 0 else None

                if self.update_progress_cb:
                    self.update_progress_cb({
                        "progress":     progress,
                        "loss":         avg_loss,
                        "epoch":        epoch+1,
                        "batch":        batch_idx+1,
                        "elapsed":      elapsed,
                        "eta":          eta
                    })

                # history.append([epoch+1, batch_idx+1, avg_loss])

                if self._save_request:
                    self._do_checkpoint(epoch, batch_idx, avg_loss, epoch_loss, batches_done)
                    self._save_request = False

                if self._stop_request:
                    self._do_checkpoint(epoch, batch_idx, avg_loss, epoch_loss, batches_done)
                    return (self._last_ckpt_path, metrics, history)

                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

            self._do_checkpoint(epoch, None, avg_loss, epoch_loss, batches_done)

        metrics["final_avg_loss"] = avg_loss
        return (self._last_ckpt_path, metrics, history)

    def request_save(self):
        """Called by save_checkpoint_llm()"""
        self._save_request = True

    def request_stop(self):
        """Called by stop_training_llm()"""
        self._stop_request = True

    def _do_checkpoint(
        self,
        epoch: int,
        batch_idx: int = None,
        loss_val: float = None,
        epoch_loss: float = None,
        batches_done: int = None
    ):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"checkpoint-epoch{epoch+1}"
        if batch_idx is not None:
            fname += f"-batch{batch_idx+1}"
        fname += f"-{now}.pt"
        path = os.path.join(self.ckpt_dir, fname)

        try:
            torch.save({
                "epoch":                 epoch+1,
                "batch_idx":             batch_idx,
                "model_state_dict":      self.model.state_dict(),
                "labels": CANDIDATE_LABELS,
                "optimizer_state_dict":  self.optimizer.state_dict(),
                "scheduler_state_dict":  self.scheduler.state_dict(),
                "scaler_state_dict":     self.scaler.state_dict(),
                "epoch_loss":            epoch_loss,
                "batches_done":          batches_done,
                "total_steps":           self.total_steps,
                "warmup_steps":          self.warmup_steps,
            }, path)

            with open(self.log_file, "a") as log:
                line = f"{datetime.now().isoformat()} | epoch {epoch+1}"
                if batch_idx is not None:
                    line += f" | batch {batch_idx+1}"
                if loss_val is not None:
                    line += f" | loss {loss_val:.4f}"
                log.write(line + "\n")

            print(f"[checkpoint saved] {path}")
            self._last_ckpt_path = path

        except Exception as e:
            print(f"[⚠️ checkpoint failed] {e}", file=sys.stderr)

_active_tuners: dict[str, FineTuner] = {}


def fine_tune_llm(
    job_id: str,
    dataset_path: str,
    model_name: str,
    output_root: str,
    update_progress_cb: Callable[[Dict], None] = None,
    num_epochs: int = 1,
    max_length: int = 256,
) -> Tuple[str, Dict[str, float], List[List[int]]]:
    """
    Trains a HuggingFace model on a CSV
    Returns: (last_checkpoint_path, metrics_dict, history_batches)
    """
    llm_root = os.path.join(output_root, "llm_checkpoints", job_id)
    os.makedirs(llm_root, exist_ok=True)
    tuner = FineTuner(
        csv_path=dataset_path,
        model_name=model_name,
        output_root=llm_root,
        update_progress_cb=update_progress_cb,
        num_epochs=num_epochs,
        max_length=max_length,
    )
    _active_tuners[job_id] = tuner

    try:
        return tuner.train()
    finally:
        _active_tuners.pop(job_id, None)


def save_checkpoint_llm(job_id: str):
    """
    External entry-point: request an immediate checkpoint.
    """
    tuner = _active_tuners.get(job_id)
    if tuner is None:
        raise RuntimeError(f"No active training session for job {job_id}")
    tuner.request_save()


def stop_training_llm(job_id: str):
    """
    External entry-point: request checkpoint + graceful stop.
    """
    tuner = _active_tuners.get(job_id)
    if tuner is None:
        raise RuntimeError(f"No active training session for job {job_id}")
    tuner.request_stop()


def infer_llm(
    ckpt_path: str,
    dataset_path: str,
    output_root: str,
    update_progress_cb=None,
) -> tuple[dict, list[list[int]], str]:
    """
    Load the finetuned model from `ckpt_path`, run inference on
    `dataset_path` (a CSV), compute metrics, confusion matrix, and save:
      • raw preds
      • summary.txt (classification report)
      • info.json (run info)
      • confusion_matrix.png
    Returns (metrics_dict, confusion_matrix, path_to_result_json).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # assume MODEL_NAME is recoverable or fixed per your project; you can hard‑code or pass it in
    MODEL_NAME = "distilbert-base-uncased"
    LABELS = ["Benign", "Botnet", "Brute_Force_Attack",
              "DoS_Attack", "Port_Scan_Infiltration",
              "Web_Attack", "Other"]

    # 1) load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    LABELS = ckpt.get("labels", CANDIDATE_LABELS)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 2) read CSV & normalize to canonical LABELS
    rows = list(csv.DictReader(open(dataset_path)))
    texts, true = [], []
    # build mapping from uppercase → canonical label
    label_map = {lbl.upper(): lbl for lbl in LABELS}

    for r in rows:
        feat = {k: v for k, v in r.items() if k.strip().lower() != "label"}
        texts.append("; ".join(f"\"{k}\":{v}\"" for k, v in feat.items()))
        raw = r["Label"].strip().upper()
        # map to canonical case if possible, else keep raw
        true.append(label_map.get(raw, raw))

    # 3) batched inference
    preds = []
    BS = 8
    for i in range(0, len(texts), BS):
        batch = texts[i : i + BS]
        toks = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**toks).logits
        idxs = torch.argmax(logits, 1).cpu().tolist()
        preds.extend([LABELS[i] for i in idxs])

    # 4) metrics + CM
    y_true = true
    y_pred = preds
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=LABELS,
            zero_division=0,
        )
        report = classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            digits=4,
            zero_division=0,
        )

    metrics = {
        "precision": float(round(prec[1], 4)),
        "recall":    float(round(rec[1], 4)),
        "f1_score":  float(round(f1[1], 4)),
    }

    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
 
    # 5) prepare output dir
    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = os.path.join(output_root, "inference", today)
    os.makedirs(out_dir, exist_ok=True)

    # save raw preds
    with open(os.path.join(out_dir, "raw_predictions.txt"), "w") as f:
        for i,(t,p) in enumerate(zip(y_true,y_pred)):
            f.write(f"{i}: {t} → {p}\n")

    # save summary
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("Classification Report:\n")
        f.write(report)

    # save info.json
    info = {
        "date": today,
        "model_name": MODEL_NAME,
        "checkpoint": os.path.basename(ckpt_path),
        "dataset": os.path.basename(dataset_path),
        "total_samples": len(texts),
        **metrics,
    }
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # plot confusion matrix with class names + cell counts
    plot_confusion_matrix(
        np.array(cm),
        LABELS,
        os.path.join(out_dir, "confusion_matrix.png"),
    )
    
    # save a top‑level result.json for API
    result = {
        "metrics": metrics,
        "confusion_matrix": cm,
        "info": info,
        "plot_filename": "confusion_matrix.png",
    }
    result_path = os.path.join(out_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    return metrics, cm, result_path


class IndexedCSV(Dataset):
    def __init__(self, csv_path, offsets, header, tokenizer, max_length, indices):
        self.csv_path   = csv_path
        self.offsets    = offsets
        self.header     = header
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.indices    = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        off = self.offsets[self.indices[idx]]
        with open(self.csv_path, "r", newline="") as f:
            f.seek(off)
            reader = csv.DictReader(f, fieldnames=self.header)
            row = next(reader)

        # reconstruct the text
        feats = {
            k: v
            for k, v in row.items()
            if k.strip().upper() != "LABEL"
        }
        text = "; ".join(f"\"{k.strip()}\":{v}" for k, v in feats.items())

        # map to binary label (your old logic)
        lab = row[next(h for h in self.header
                       if "label" in h.strip().lower())].strip().upper()
        label = 0 if "BENIGN" in lab else 1

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

import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_path: str):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.YlGnBu)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Count", rotation=-90, va="bottom")

    # set ticks
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # rotate x‑labels
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=10,
    )
    plt.setp(ax.get_yticklabels(), fontsize=10)

    # annotate each cell with its count
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]:d}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )

    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

    # make room on the bottom and left for long labels
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.25, left=0.25)

    plt.savefig(out_path, dpi=150)
    plt.close(fig)
