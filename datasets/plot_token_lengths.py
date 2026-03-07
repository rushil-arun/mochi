#!/usr/bin/env python3
"""
Plot a histogram of token lengths for the 'prompt' column in a CSV file.
Uses the Qwen2.5 tokenizer by default, but falls back to a simple whitespace
tokenizer if transformers is not installed.

Usage:
    python plot_token_lengths.py data.csv
    python plot_token_lengths.py data.csv --tokenizer Qwen/Qwen2.5-0.5B
    python plot_token_lengths.py data.csv --max_length 200 --output histogram.png
"""

import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


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
    parser = argparse.ArgumentParser(description="Plot token length histogram for a CSV prompt column.")
    parser.add_argument("--csv_file", help="Path to the CSV file")
    parser.add_argument("--column", default="prompt", help="Column name to tokenize (default: 'prompt')")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B", help="HuggingFace tokenizer to use")
    parser.add_argument("--max_length", type=int, default=None, help="Draw a vertical cutoff line at this token length")
    parser.add_argument("--bins", type=int, default=40, help="Number of histogram bins (default: 40)")
    parser.add_argument("--output", default="token_length_histogram.png", help="Output image filename")
    args = parser.parse_args()

    # Load CSV
    print(f"Loading {args.csv_file} ...")
    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"Error: File '{args.csv_file}' not found.")
        sys.exit(1)

    if args.column not in df.columns:
        print(f"Error: Column '{args.column}' not found in CSV. Available columns: {list(df.columns)}")
        sys.exit(1)

    df = df.dropna(subset=[args.column])
    print(f"Found {len(df)} rows with non-null '{args.column}' values.")

    # Tokenize
    tokenizer = load_tokenizer(args.tokenizer)
    print("Counting tokens...")
    token_lengths = df[args.column].apply(lambda x: count_tokens(str(x), tokenizer))

    # Stats
    p50  = int(np.percentile(token_lengths, 50))
    p90  = int(np.percentile(token_lengths, 90))
    p95  = int(np.percentile(token_lengths, 95))
    p99  = int(np.percentile(token_lengths, 99))
    mean = int(token_lengths.mean())
    max_ = int(token_lengths.max())

    print(f"\n--- Token Length Stats ---")
    print(f"  Mean:  {mean}")
    print(f"  p50:   {p50}")
    print(f"  p90:   {p90}")
    print(f"  p95:   {p95}")
    print(f"  p99:   {p99}")
    print(f"  Max:   {max_}")
    if args.max_length:
        pct_over = 100 * (token_lengths > args.max_length).mean()
        print(f"  Over {args.max_length} tokens: {pct_over:.1f}% of prompts")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    n, bins_out, patches = ax.hist(
        token_lengths, bins=args.bins,
        color="#4f8ef7", edgecolor="#0f1117", linewidth=0.5, alpha=0.9
    )

    # Colour bars beyond max_length red
    if args.max_length:
        for patch, left in zip(patches, bins_out[:-1]):
            if left >= args.max_length:
                patch.set_facecolor("#e05c5c")
        ax.axvline(args.max_length, color="#e05c5c", linewidth=1.8,
                   linestyle="--", label=f"Cutoff: {args.max_length} tokens")

    # Percentile lines
    for val, label, color in [
        (p50, "p50", "#a0c4ff"),
        (p90, "p90", "#ffd166"),
        (p95, "p95", "#ef9b20"),
    ]:
        ax.axvline(val, color=color, linewidth=1.2, linestyle=":", alpha=0.85, label=f"{label}: {val}")

    # Styling
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d3a")
    ax.tick_params(colors="#9ca3af", labelsize=10)
    ax.xaxis.label.set_color("#d1d5db")
    ax.yaxis.label.set_color("#d1d5db")
    ax.set_xlabel("Token Length", fontsize=12, labelpad=8)
    ax.set_ylabel("Number of Prompts", fontsize=12, labelpad=8)
    ax.set_title(
        f"Token Length Distribution — '{args.column}' column  (n={len(df):,})",
        color="#f9fafb", fontsize=13, pad=14
    )
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(framealpha=0.2, labelcolor="white", fontsize=10)

    # Stats box
    stats_text = f"mean={mean}  p50={p50}  p90={p90}  p95={p95}  max={max_}"
    ax.text(0.99, 0.97, stats_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="#9ca3af",
            bbox=dict(facecolor="#1e2130", edgecolor="none", alpha=0.7, pad=4))

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nHistogram saved to: {args.output}")


if __name__ == "__main__":
    main()