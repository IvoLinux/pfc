import numpy as np
import os
import pandas as pd
import time
import warnings

import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns

from dask_ml.preprocessing import StandardScaler as DaskStandardScaler
from dask_ml.preprocessing import LabelEncoder as DaskLabelEncoder
from dask_ml.model_selection import train_test_split as dask_train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

RESULTS_SAVE_DIR = "./preprocessed-data"
RAW_DATASET_DIR = "/home/bibber/Downloads/CIC-IDS 2017"
os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
# True  = encode your grouped labels into integers
# False = keep the raw label strings (from the original CSV)
ENCODE_LABELS = False
# ────────────────────────────────────────────────────────────────────────────────

def group_attack_labels(label):
    l = str(label).lower().strip()
    if any(x in l for x in ['dos', 'ddos', 'slowloris', 'hulk', 'goldeneye']):
        return 'DoS_Attack'
    if any(x in l for x in ['bruteforce', 'ftp-patator', 'ssh-patator']):
        return 'Brute_Force_Attack'
    if any(x in l for x in ['web', 'sql', 'xss', 'injection']):
        return 'Web_Attack'
    if any(x in l for x in ['port', 'scan', 'nmap', 'portsweep', 'infiltration']):
        return 'Port_Scan_Infiltration'
    if 'bot' in l:
        return 'Botnet'
    if 'heartbleed' in l:
        return 'Heartbleed'
    if l in ('benign', 'normal', 'legitimate'):
        return 'Benign'
    return 'Other_Attack'

def plot_class_distribution(y, title, filename=None):
    y_train_arr = y.compute()
    counts = pd.Series(y_train_arr).value_counts()

    plt.figure(figsize=(12,6))
    ax = sns.barplot(x=counts.index, y=counts.values)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.1, str(v), ha='center')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

def fmt_float(x):
    if pd.isna(x):
        return ""
    s = f"{x:.5f}".rstrip("0").rstrip(".")
    # ensure at least one decimal place
    if "." not in s:
        s += ".0"
    return s

if __name__ == "__main__":
    start_time = time.time()
    print("Reading CSVs with Dask...")
    df = dd.read_csv(
        os.path.join(RAW_DATASET_DIR, "*.csv"),
        assume_missing=True,
        skipinitialspace=True,
    ).rename(columns=lambda s: s.strip())

    # 1) Identify the label column
    candidates = [
        c for c in df.columns
        if 'label' in c.strip().lower() or 'attack' in c.strip().lower() or 'category' in c.strip().lower()
    ]
    label_col = candidates[0] if candidates else df.columns[-1]
    print("Target column:", label_col)

    # 2) Keep a copy of the raw CSV label
    df['original_label'] = df[label_col]

    # 3) Map to your grouped categories
    df[label_col] = df[label_col].map(
        group_attack_labels,
        meta=(label_col, 'object')
    )

    # 4) Feature engineering
    if {'Flow Bytes/s', 'Flow Duration'}.issubset(df.columns):
        df['flow_bytes_per_sec'] = df['Flow Bytes/s'] / (df['Flow Duration'] + 1)
    if {'Total Fwd Packets', 'Total Backward Packets', 'Flow Duration'}.issubset(df.columns):
        df['packet_rate'] = (
            df['Total Fwd Packets'] + df['Total Backward Packets']
        ) / (df['Flow Duration'] + 1)

    # 5) Convert all other object columns to numeric (exclude your labels!)
    obj_cols = [
        c for c in df.select_dtypes(include=['object']).columns
        if c not in (label_col, 'original_label')
    ]
    for col in obj_cols:
        df[col] = dd.to_numeric(df[col], errors='coerce')

    # 6) Replace infinities & fill NaNs with means
    df = df.replace([np.inf, -np.inf], np.nan)
    num_cols = df._meta.select_dtypes(include=['number']).columns.tolist()
    means = pd.Series(
        {col: df[col].mean().compute() for col in num_cols},
        index=num_cols
    )
    df[num_cols] = df[num_cols].fillna(means)

    # 7) Choose labels: encoded ints or raw strings
    if ENCODE_LABELS:
        print("→ Encoding grouped labels to integers")
        le = DaskLabelEncoder()
        y = le.fit_transform(df[label_col])
    else:
        print("→ Using raw labels from original CSV")
        y = df['original_label']

    # 8) Drop label columns from features
    X = df.drop([label_col, 'original_label'], axis=1)

    # 9) Split train/test
    X_train, X_test, y_train, y_test = dask_train_test_split(
        X, y,
        test_size=0.2,
        random_state=21023 + 21041,
        convert_mixed_types=True,
    )

    # ─── Write out the TRAIN split with trimmed decimals ─────────────────────────────
    print("Saving train_split.csv…")
    Xtr = X_train.compute()
    ytr = y_train.compute()
    df_tr = pd.DataFrame(Xtr, columns=X.columns)
    df_tr[label_col] = ytr

    # convert float columns to formatted strings
    float_cols = df_tr.select_dtypes(include=["float"]).columns
    df_tr[float_cols] = df_tr[float_cols].applymap(fmt_float)

    df_tr.to_csv(
        os.path.join(RESULTS_SAVE_DIR, "train_split.csv"),
        index=False
    )

    # ─── Write out the TEST split with trimmed decimals ──────────────────────────────
    print("Saving test_split.csv…")
    Xte = X_test.compute()
    yte = y_test.compute()
    df_te = pd.DataFrame(Xte, columns=X.columns)
    df_te[label_col] = yte

    # reuse the same formatter
    float_cols = df_te.select_dtypes(include=["float"]).columns
    df_te[float_cols] = df_te[float_cols].applymap(fmt_float)

    df_te.to_csv(
        os.path.join(RESULTS_SAVE_DIR, "test_split.csv"),
        index=False
)

    # 10) Quick plot of your train-set class distribution
    plot_class_distribution(
        y_train,
        "Class Distribution Before Resampling",
        os.path.join(RESULTS_SAVE_DIR, 'before_resample.png')
    )

    # 11) Scale features
    scaler = DaskStandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Preprocessing complete in {(time.time() - start_time):.1f}s")
