import os
import time
import warnings

import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns

from dask_ml.preprocessing import StandardScaler, LabelEncoder
from dask_ml.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ——— CONFIG ———
RAW_DATASET_DIR = "/home/bibber/Downloads/CIC-IDS 2018"
RESULTS_SAVE_DIR = "./CIC-IDS 2018"
ENCODE_LABELS   = False                  # switch to True if you want integer labels

os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)

# ——— 0) Helpers ———
def group_attack_labels(label):
    l = str(label).lower().strip()
    if any(x in l for x in ['dos','slowloris','hulk','goldeneye']):
        return 'DoS_Attack'
    if any(x in l for x in ['web','sql','xss','injection']):
        return 'Web_Attack'
    if any(x in l for x in ['brute','force','ftp','ssh']):
        return 'Brute_Force_Attack'
    if any(x in l for x in ['port','scan','nmap','portsweep','infiltration', 'infilteration']):
        return 'Port_Scan_Infiltration'
    if 'bot' in l:
        return 'Botnet'
    if 'heartbleed' in l:
        return 'Heartbleed'
    if l in ('benign','normal'):
        return 'Benign'
    return 'Other_Attack'

def fmt_float(x):
    if pd.isna(x):
        return ""
    s = f"{x:.5f}".rstrip("0").rstrip(".")
    return s if "." in s else s + ".0"

def plot_class_distribution(y_dd, title, filename=None):
    arr = y_dd.compute()
    counts = pd.Series(arr).value_counts().sort_index()
    plt.figure(figsize=(10,5))
    ax = sns.barplot(x=counts.index, y=counts.values)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.02*v, str(v), ha='center')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()


def save_split(X_dd, y_dd, name):
    print(f"   → Writing {name}_split.csv…")
    X_df = X_dd.compute()
    y_arr = y_dd.compute() if hasattr(y_dd, "compute") else y_dd.compute()
    out = X_df.copy()
    out[label_col] = y_arr

    float_cols = out.select_dtypes(include=['float']).columns
    path = os.path.join(RESULTS_SAVE_DIR, f"{name}_split.csv")
    with open(path, "w", newline="") as f:
        out.head(0).to_csv(f, index=False)
        for i in range(0, len(out), 100_000):
            chunk = out.iloc[i : i + 100_000].copy()
            chunk[float_cols] = chunk[float_cols].applymap(fmt_float)
            chunk.to_csv(f, index=False, header=False)

# ——— MAIN ———
if __name__ == "__main__":
    t0 = time.time()
    print("1) Reading CSVs from", RAW_DATASET_DIR)
    df = dd.read_csv(
        os.path.join(RAW_DATASET_DIR, "*.csv"),
        assume_missing=True,
        skipinitialspace=True,
        dtype=str                    # ← force all columns to string
    ).rename(columns=lambda s: s.strip())

    # 2) Drop any “Timestamp” column to avoid time‑based leakage
    if 'Timestamp' in df.columns:
        df = df.drop('Timestamp', axis=1)

    # 3) Identify label column
    label_candidates = [
        c for c in df.columns
        if 'label' in c.lower() or 'attack' in c.lower() or 'category' in c.lower()
    ]
    label_col = label_candidates[0] if label_candidates else df.columns[-1]
    print("   → using label column:", label_col)

    # 4) Keep a raw copy, then group
    # df['original_label'] = df[label_col]
    df[label_col] = df[label_col].map(
        group_attack_labels,
        meta=(label_col, 'object')
    )

    # 5) Feature engineering
    if {'Flow Bytes/s','Flow Duration'}.issubset(df.columns):
        df['flow_bytes_per_sec'] = df['Flow Bytes/s'] / (df['Flow Duration'] + 1)
    if {'Total Fwd Pkts','Tot Bwd Pkts','Flow Duration'}.issubset(df.columns):
        df['packet_rate'] = (
            df['Total Fwd Pkts'] + df['Tot Bwd Pkts']
        ) / (df['Flow Duration'] + 1)

    # 6) Coerce remaining object columns → numeric (leave labels alone)
    obj_cols = [
        c for c in df.select_dtypes(include=['object']).columns
        if c != label_col
    ]
    for c in obj_cols:
        df[c] = dd.to_numeric(df[c], errors='coerce')

    # 7) Quick peek at full distribution
    plot_class_distribution(
        df[label_col],
        "Full Dataset — Grouped Class Distribution",
        os.path.join(RESULTS_SAVE_DIR, "full_class_dist.png")
    )

    # 8) Split rows into train/test
    print("2) Splitting into train/test (80/20)…")
    X = df.drop(label_col, axis=1)
    if ENCODE_LABELS:
        le = LabelEncoder()
        y = le.fit_transform(df[label_col])
    else:
        y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=42,
        convert_mixed_types=True
    )

    # 9) Impute missing & infinite values with TRAIN means only
    print("3) Computing training set means for imputation…")
    numeric_cols = X_train._meta.select_dtypes(include=['number']).columns.tolist()
    train_means = {
        col: X_train[col].mean().compute()
        for col in numeric_cols
    }

    def impute_with_means(ddf, means):
        ddf = ddf.replace([np.inf, -np.inf], np.nan)
        return ddf.fillna(means)

    X_train = impute_with_means(X_train, train_means)
    X_test  = impute_with_means(X_test,  train_means)

    # ─── 10) SAVE YOUR TRAIN/TEST SPLITS *NOW*, BEFORE SCALING ───
    save_split(X_train, y_train, "train")
    save_split(X_test,  y_test,  "test")

    # ─── 11) SCALE ───
    print("4) Scaling features…")

    # take a numeric‐only slice for scaling, but do NOT overwrite X_train
    num_cols = X_train._meta.select_dtypes(include=['number']).columns.tolist()
    X_train_num = X_train[num_cols]
    X_test_num  = X_test[num_cols]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_num)
    X_test_scaled  = scaler.transform(X_test_num)

    # ─── 12) FINAL PLOTS ───
    plot_class_distribution(
        y_train,
        "Train Set Distribution After Preprocessing",
        os.path.join(RESULTS_SAVE_DIR, "train_dist_after.png")
    )
    plot_class_distribution(
        y_test,
        "Test Set Distribution After Preprocessing",
        os.path.join(RESULTS_SAVE_DIR, "test_dist_after.png")
    )
    print(f"✅ Done in {(time.time() - t0):.1f}s")