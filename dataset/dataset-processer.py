import os
import glob
import random
import re
import pandas as pd

# 1) Configure these:
DATA_DIR = "/home/bibber/Downloads/CIC-IDS 2017"
LABEL_COLUMN = " Label"
OUTPUT_CSV = "subset_cicids2017.csv"

category_patterns = {
    "Web Attack":    ["Web Attack"],
    "Brute Force":   ["FTP-Patator", "SSH-Patator"],
    "Bot":           ["Bot", "Botnet", "Infiltration", "Heartbleed"],
    "PortScan":      ["PortScan"],
    "DoS/DDoS":      ["DoS"],
    "BENIGN":        ["BENIGN"]
}

desired_counts = {
    "Web Attack": 2000,
    "Brute Force": 4000,
    "Bot": 1900,
    "PortScan": 8000,
    "DoS/DDoS": 10000,
    "BENIGN": 50000,
}

category_regex = {
    cat: re.compile("|".join(re.escape(s) for s in subs), re.IGNORECASE)
    for cat, subs in category_patterns.items()
}


# 2) Initialize reservoirs and counters for reservoir sampling
reservoirs = {lbl: [] for lbl in desired_counts}
seen_counts = {lbl: 0 for lbl in desired_counts}


# 3) Iterate through every CSV in the directory, chunk by chunk
for csv_file in glob.glob(os.path.join(DATA_DIR, "*.csv")):
    for chunk in pd.read_csv(csv_file, chunksize=100_000):
        # for each category, pick matching rows
        for cat, regex in category_regex.items():
            match_idx = chunk[LABEL_COLUMN].str.contains(regex, na=False)
            for _, row in chunk.loc[match_idx].iterrows():
                seen_counts[cat] += 1
                k = desired_counts[cat]
                if len(reservoirs[cat]) < k:
                    reservoirs[cat].append(row)
                else:
                    j = random.randint(0, seen_counts[cat] - 1)
                    if j < k:
                        reservoirs[cat][j] = row



# 4) Combine, shuffle, and write out
all_samples = []
for cat, rows in reservoirs.items():
    found = len(rows)
    want  = desired_counts[cat]
    if found < want:
        print(f"Warning: only found {found} rows for '{cat}', wanted {want}")
    all_samples.extend(rows)


df = pd.DataFrame(all_samples)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Wrote {len(df)} rows to {OUTPUT_CSV}")

