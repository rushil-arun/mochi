import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove rows from a CSV where label == 0."
    )
    parser.add_argument(
        "--input",
        default="datasets/Mochi - Full Dataset.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default="datasets/Mochi - Full Dataset_label1_only.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Name of the label column.",
    )
    parser.add_argument(
        "--remove-value",
        default="0",
        help="Label value to remove (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    removed_rows = 0
    kept_rows = 0

    with input_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("Input CSV appears to have no header.")
        if args.label_column not in fieldnames:
            raise ValueError(
                f"Column '{args.label_column}' not found. Found: {fieldnames}"
            )

        with output_path.open("w", encoding="utf-8", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                total_rows += 1
                value = str(row.get(args.label_column, "")).strip()
                if value == str(args.remove_value):
                    removed_rows += 1
                    continue
                writer.writerow(row)
                kept_rows += 1

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Total rows read: {total_rows}")
    print(f"Rows removed ({args.label_column} == {args.remove_value}): {removed_rows}")
    print(f"Rows kept: {kept_rows}")


if __name__ == "__main__":
    main()
