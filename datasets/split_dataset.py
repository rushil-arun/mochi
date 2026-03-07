"""
Stratified train/val/test split for a binary classification CSV dataset.

Usage:
    python split_dataset.py --input data.csv [--output-dir .] [--seed 42]

Arguments:
    --input       Path to the input CSV file (must contain a "class" column)
    --output-dir  Directory to write train.csv, val.csv, test.csv (default: same as input)
    --seed        Random seed for reproducibility (default: 42)
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(input_path: str, output_dir: str, seed: int) -> None:
    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(input_path)

    if "class" not in df.columns:
        raise ValueError("Input CSV must contain a 'class' column.")

    print(f"Loaded {len(df):,} rows from '{input_path}'")
    print(f"Class distribution:\n{df['class'].value_counts().to_string()}\n")

    # ── Stratified split: 80 / 10 / 10 ───────────────────────────────────────
    # Step 1: split off 20 % (val + test) while preserving class ratios
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["class"],
        random_state=seed,
    )

    # Step 2: split the 20 % evenly into val and test (each 10 % of original)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["class"],
        random_state=seed,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    splits = {"train": train_df, "val": val_df, "test": test_df}
    for name, split_df in splits.items():
        out_path = os.path.join(output_dir, f"{name}.csv")
        split_df.to_csv(out_path, index=False)

        dist = split_df["class"].value_counts(normalize=True).sort_index()
        print(
            f"{name:>5}.csv  →  {len(split_df):>6,} rows  |  "
            + "  ".join(f"class {k}: {v:.1%}" for k, v in dist.items())
        )

    print(f"\nFiles written to: {os.path.abspath(output_dir)}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified train/val/test CSV splitter.")
    parser.add_argument("--input",      required=True,  help="Path to input CSV file")
    parser.add_argument("--output-dir", default=None,   help="Output directory (default: same as input file)")
    parser.add_argument("--seed",       type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    split_dataset(args.input, output_dir, args.seed)