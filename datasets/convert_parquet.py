#!/usr/bin/env python3
"""
Convert a Parquet file to a CSV file.

Usage:
    python parquet_to_csv.py <input.parquet> [output.csv]

Requirements:
    pip install pandas pyarrow
"""

import sys
import os
import pandas as pd


def parquet_to_csv(input_path: str, output_path: str = None) -> str:
    """
    Convert a Parquet file to CSV format.

    Args:
        input_path:  Path to the input .parquet file.
        output_path: Path for the output .csv file.
                     Defaults to the same name/location as the input file.

    Returns:
        The path to the written CSV file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}.csv"

    print(f"Reading Parquet file: {input_path}")
    df = pd.read_parquet(input_path)

    print(f"  Rows   : {len(df):,}")
    print(f"  Columns: {len(df.columns):,}  ({', '.join(df.columns.tolist()[:5])}"
          f"{'...' if len(df.columns) > 5 else ''})")

    print(f"Writing CSV file  : {output_path}")
    df.to_csv(output_path, index=False)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done — {size_mb:.2f} MB written to {output_path}")

    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python parquet_to_csv.py <input.parquet> [output.csv]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else None

    try:
        parquet_to_csv(input_path, output_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()