#!/usr/bin/env python3
"""
Filter a CSV file to only keep rows where the 'prompt' column is under a
specified token length threshold.

Usage:
    python filter_by_token_length.py input.csv output.csv
    python filter_by_token_length.py input.csv output.csv --max_tokens 200
    python filter_by_token_length.py input.csv output.csv --max_tokens 200 --tokenizer Qwen/Qwen2.5-0.5B
"""

import argparse
import sys
import pandas as pd


def load_tokenizer(tokenizer_name: str):
    try:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer: {tokenizer_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print("Tokenizer loaded.")
        return tokenizer
    except ImportError:
        print("Warning: `transformers` not installed. Falling back to whitespace tokenizer.")
        return None
    except Exception as e:
        print(f"Warning: Could not load tokenizer '{tokenizer_name}': {e}")
        print("Falling back to whitespace tokenizer.")
        return None


def count_tokens(text: str, tokenizer) -> int:
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer.encode(text, add_special_tokens=False))


def main():
    parser = argparse.ArgumentParser(description="Filter CSV rows by prompt token length.")
    parser.add_argument("--input_csv", help="Path to the input CSV file")
    parser.add_argument("--output_csv", help="Path to write the filtered CSV file")
    parser.add_argument("--column", default="prompt", help="Column name to tokenize (default: 'prompt')")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum token length (exclusive, default: 200)")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B", help="HuggingFace tokenizer to use")
    args = parser.parse_args()

    # Load CSV
    print(f"Loading {args.input_csv} ...")
    try:
        df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"Error: File '{args.input_csv}' not found.")
        sys.exit(1)

    if args.column not in df.columns:
        print(f"Error: Column '{args.column}' not found. Available columns: {list(df.columns)}")
        sys.exit(1)

    original_count = len(df)
    df = df.dropna(subset=[args.column])
    if len(df) < original_count:
        print(f"Dropped {original_count - len(df)} rows with null '{args.column}' values.")

    # Tokenize
    tokenizer = load_tokenizer(args.tokenizer)
    print("Counting tokens...")
    token_lengths = df[args.column].apply(lambda x: count_tokens(str(x), tokenizer))

    # Filter
    mask = token_lengths < args.max_tokens
    filtered_df = df[mask].reset_index(drop=True)

    removed = len(df) - len(filtered_df)
    pct_removed = 100 * removed / len(df)

    print(f"\n--- Filter Results ---")
    print(f"  Original rows : {len(df):,}")
    print(f"  Removed rows  : {removed:,}  ({pct_removed:.1f}%)")
    print(f"  Remaining rows: {len(filtered_df):,}")

    # Save
    filtered_df.to_csv(args.output_csv, index=False)
    print(f"\nFiltered CSV saved to: {args.output_csv}")


if __name__ == "__main__":
    main()