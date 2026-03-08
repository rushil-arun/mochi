import argparse
import csv
import json
import random
from pathlib import Path


REFUSAL_TEMPLATES = [
    "I'm not able to help with that. Is there something else I can assist you with?",
    "That's not something I'm able to assist with. Feel free to ask me something else.",
    "I'm unable to fulfill that request, but I'm happy to help with other questions.",
    "I can't assist with that request, but I can help with a safer alternative.",
    "I’m sorry, but I can’t help with that. I can help with something else instead.",
    "I can't provide help for that request. If you'd like, I can help with a different topic.",
    "I’m not able to support that request. I can still help with general, safe information.",
    "I can't help with that, but I can offer guidance on a safer direction.",
    "I’m unable to assist with that request. Please ask about something I can safely support.",
    "I can’t help with that request, but I’m here to help with other questions.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a combined dataset from MalicousPromptsDataset.csv and "
            "a sampled subset of databricks-dolly-15k.jsonl."
        )
    )
    parser.add_argument(
        "--malicious-csv",
        default="datasets/MalicousPromptsDataset.csv",
        help="Path to MalicousPromptsDataset CSV.",
    )
    parser.add_argument(
        "--dolly-jsonl",
        default="datasets/databricks-dolly-15k.jsonl",
        help="Path to Dolly JSONL file.",
    )
    parser.add_argument(
        "--output-csv",
        default="datasets/combined_safety_dataset.csv",
        help="Path for combined output CSV.",
    )
    parser.add_argument(
        "--dolly-fraction",
        type=float,
        default=1.0 / 3.0,
        help="Fraction of Dolly rows to sample (default: one-third).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling/template assignment.",
    )
    return parser.parse_args()


def build_dolly_prompt(instruction: str, context: str) -> str:
    instruction = (instruction or "").strip()
    context = (context or "").strip()
    if not context:
        return instruction
    return f"{instruction}\n\nContext:\n{context}"


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    malicious_path = Path(args.malicious_csv)
    dolly_path = Path(args.dolly_jsonl)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with malicious_path.open("r", encoding="utf-8", newline="") as f:
        malicious_reader = csv.DictReader(f)
        malicious_fieldnames = malicious_reader.fieldnames or []
        malicious_rows = list(malicious_reader)

    required_malicious_cols = {
        "prompt",
        "label",
        "category",
        "source",
        "dataset",
        "variant_group_id",
        "variant_index",
    }
    missing = required_malicious_cols.difference(malicious_fieldnames)
    if missing:
        raise ValueError(
            f"Malicious CSV missing required columns: {sorted(missing)}"
        )

    # Keep same columns as MalicousPromptsDataset, plus response.
    output_fieldnames = list(malicious_fieldnames)
    if "response" not in output_fieldnames:
        output_fieldnames.append("response")

    output_rows = []

    # Add all malicious rows with generated refusal responses.
    for row in malicious_rows:
        out = {k: row.get(k, "") for k in malicious_fieldnames}
        out["response"] = random.choice(REFUSAL_TEMPLATES)
        output_rows.append(out)

    # Load all Dolly rows.
    dolly_rows = []
    with dolly_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            dolly_rows.append(json.loads(line))

    if not dolly_rows:
        raise ValueError("No rows found in Dolly JSONL.")
    if not (0 < args.dolly_fraction <= 1):
        raise ValueError("--dolly-fraction must be in (0, 1].")

    sample_size = int(len(dolly_rows) * args.dolly_fraction)
    sample_size = max(1, sample_size)
    sampled = random.sample(dolly_rows, sample_size)

    # Add sampled Dolly rows.
    for row in sampled:
        instruction = row.get("instruction", "")
        context = row.get("context", "")
        response = row.get("response", "")
        category = row.get("category", "")

        out = {k: "" for k in malicious_fieldnames}
        out["prompt"] = build_dolly_prompt(instruction, context)
        out["response"] = response
        out["label"] = "0"
        out["category"] = category
        out["source"] = "Dolly"
        out["dataset"] = "Dolly"
        out["variant_group_id"] = ""
        out["variant_index"] = ""
        output_rows.append(out)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote: {output_path}")
    print(f"Malicious rows added: {len(malicious_rows)}")
    print(f"Dolly total rows: {len(dolly_rows)}")
    print(f"Dolly sampled rows: {sample_size}")
    print(f"Total output rows: {len(output_rows)}")
    print("\nRefusal templates used:")
    for i, t in enumerate(REFUSAL_TEMPLATES, start=1):
        print(f"{i}. {t}")


if __name__ == "__main__":
    main()
