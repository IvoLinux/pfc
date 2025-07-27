"""
High-level ML pipeline for network-traffic anomaly detection.

* tabular_model.train_model(...)   → train, evaluate, save, return metrics + ckpt
* tabular_model.infer_model(...)   → load checkpoint, run inference, return metrics

The code purposefully avoids GPU-only libraries so it will run in the
slim Python image already used in the Dockerfile.  If you want RAPIDS/
CuPy acceleration just swap the sklearn blocks for your RAPIDS code.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
#                                CONSTANTS                                    #
# --------------------------------------------------------------------------- #
DEFAULT_LABEL_ALIASES = {"benign", "normal", "legitimate"}
RANDOM_SEED = 42


# --------------------------------------------------------------------------- #
#                            HELPER  UTILITIES                                #
# --------------------------------------------------------------------------- #
def _detect_label_column(df: pd.DataFrame) -> str:
    """Try to guess the label column name."""
    for cand in ["label", "Label", "attack", "Attack", "class", "Class"]:
        if cand in df.columns:
            return cand
    # fallback – last column
    return df.columns[-1]


def _binary_label(x: str) -> int:
    return 0 if str(x).strip().lower() in DEFAULT_LABEL_ALIASES else 1


def _split_data(
    df: pd.DataFrame, label_col: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X = df.drop(columns=[label_col])
    y = df[label_col]

    # keep only numeric features
    X_num = X.select_dtypes(include=["number"]).fillna(0)

    X_train, X_val, y_train, y_val = train_test_split(
        X_num, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    return X_train, y_train, X_val, y_val


def _balance_smote(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Moderate class-imbalance fix – increase minority with SMOTE."""
    if y.value_counts().min() / y.value_counts().max() >= 0.3:
        # not that imbalanced
        return X, y

    smote = SMOTE(random_state=RANDOM_SEED)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res


def _make_pipeline(n_features: int) -> Pipeline:
    scaler = StandardScaler()
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        class_weight="balanced_subsample",
    )
    return Pipeline(steps=[("scaler", scaler), ("model", rf)])


def _metric_dict(
    y_true: pd.Series, y_pred: np.ndarray
) -> Tuple[Dict[str, float], List[List[int]]]:
    acc, prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
    }
    cm = confusion_matrix(y_true, y_pred).tolist()
    return metrics, cm


# --------------------------------------------------------------------------- #
#                       PUBLIC  PIPELINE  FUNCTIONS                           #
# --------------------------------------------------------------------------- #
def train_model(
    dataset_path: str,
    model_name: str,
    output_root: str,
    update_progress_cb=None,
) -> Tuple[str, Dict[str, float], List[List[int]]]:
    """
    Main training entry-point.

    Returns
    -------
    ckpt_path : str
    metrics   : Dict[str, float]
    conf_mat  : List[List[int]]
    """
    t0 = time.time()

    df = pd.read_csv(dataset_path, low_memory=False)
    label_col = _detect_label_column(df)
    df[label_col] = df[label_col].apply(_binary_label)

    X_train, y_train, X_val, y_val = _split_data(df, label_col)
    X_train, y_train = _balance_smote(X_train, y_train)

    pipe = _make_pipeline(X_train.shape[1])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_val)
    metrics, cm = _metric_dict(y_val, y_pred)

    # save checkpoint (pipeline inc. scaler + model) & metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_name = f"{model_name}__{timestamp}.joblib"
    ckpt_dir = Path(output_root).joinpath("tabular_checkpoints", model_name)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / ckpt_name
    joblib.dump(pipe, ckpt_path)

    # save metrics JSON right next to checkpoint
    meta_path = ckpt_path.with_suffix(".json")
    with open(meta_path, "w") as fp:
        json.dump({"metrics": metrics, "confusion_matrix": cm}, fp, indent=2)

    print(
        f"[TRAIN] {dataset_path} → {ckpt_path}  "
        f"(took {time.time()-t0:.1f}s, accuracy={metrics['accuracy']:.3f})"
    )
    return str(ckpt_path), metrics, cm


def infer_model(
    ckpt_path: str,
    dataset_path: str,
    output_root: str,
    update_progress_cb=None,
) -> Tuple[Dict[str, float], List[List[int]], str]:
    """
    Load a saved pipeline & run inference on *dataset_path*.

    If the dataset has the ground-truth label column we compute metrics,
    otherwise we only output predictions.

    Returns
    -------
    metrics       : Dict[str, float]   (empty dict if labels absent)
    confusion_mat : List[List[int]] | []
    result_json   : str  (path saved)
    """
    from sklearn.exceptions import NotFittedError

    pipe: Pipeline = joblib.load(ckpt_path)
    df = pd.read_csv(dataset_path, low_memory=False)
    label_col = _detect_label_column(df)
    has_labels = label_col in df.columns

    if has_labels:
        df[label_col] = df[label_col].apply(_binary_label)

    X = df.select_dtypes(include=["number"]).fillna(0)

    if update_progress_cb:
        update_progress_cb(30)

    try:
        preds = pipe.predict(X)
    except NotFittedError as e:
        raise RuntimeError(f"Model at {ckpt_path} not fitted: {e}") from e

    if update_progress_cb:
        update_progress_cb(70)

    metrics, cm = ({}, []) if not has_labels else _metric_dict(df[label_col], preds)

    # save predictions + metrics
    res_dir = Path(output_root).joinpath("inference_results")
    res_dir.mkdir(parents=True, exist_ok=True)
    res_name = f"result__{Path(dataset_path).stem}__{Path(ckpt_path).stem}.json"
    res_path = res_dir / res_name

    with open(res_path, "w") as fp:
        json.dump(
            {
                "metrics": metrics,
                "confusion_matrix": cm,
                "predictions": preds.tolist(),
            },
            fp,
            indent=2,
        )

    if update_progress_cb:
        update_progress_cb(100)

    return metrics, cm, str(res_path)
