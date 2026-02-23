import csv
import sys

def remove_phrase_from_csv(input_file, output_file, column, phrase):
    """
    Remove a phrase from the end of values in a specified column of a CSV file.
    
    Args:
        input_file:  Path to the input CSV file
        output_file: Path to save the modified CSV file
        column:      Column name (or 0-based index if no header)
        phrase:      The phrase to remove from the end of values
    """
    rows = []
    modified_count = 0

    with open(input_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        if column not in fieldnames:
            print(f"Error: Column '{column}' not found. Available columns: {fieldnames}")
            sys.exit(1)

        for row in reader:
            value = row[column]
            if value.endswith(phrase):
                row[column] = value[: -len(phrase)]
                modified_count += 1
            rows.append(row)

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done! Modified {modified_count} row(s). Saved to '{output_file}'.")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python remove_phrase.py <input.csv> <output.csv> <column_name> <phrase>")
        print('Example: python remove_phrase.py data.csv cleaned.csv "Description" " (draft)"')
        sys.exit(1)

    _, input_file, output_file, column, phrase = sys.argv
    remove_phrase_from_csv(input_file, output_file, column, phrase)