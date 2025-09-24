#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Reduce dataset size for testing and inference"
    )
    parser.add_argument(
        "--input", required=True, help="Path to the input CSV (with Text,Label)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    df = df.sample(frac=1.0, random_state=21023).reset_index(drop=True)

    n_train = max(1, len(df) // 2)
    df_train_small = df.iloc[:n_train]

    n_infer = max(1, n_train // 5)
    df_infer_small = df.iloc[n_train:n_train + n_infer]

    out_train = input_path.parent / "train_small.csv"
    out_infer = input_path.parent / "inference_small.csv"

    df_train_small.to_csv(out_train, index=False)
    df_infer_small.to_csv(out_infer, index=False)

    print(f"Full dataset: {len(df)} rows")
    print(f"train_small: {len(df_train_small)} rows saved to {out_train}")
    print(f"inference_small: {len(df_infer_small)} rows saved to {out_infer}")

if __name__ == "__main__":
    main()
