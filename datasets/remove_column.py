import csv
import sys

def remove_column(input_file, output_file, column):
    rows = []

    with open(input_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        if column not in fieldnames:
            print(f"Error: Column '{column}' not found. Available columns: {fieldnames}")
            sys.exit(1)

        new_fieldnames = [col for col in fieldnames if col != column]

        for row in reader:
            del row[column]
            rows.append(row)

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done! Removed column '{column}'. Saved to '{output_file}'.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python remove_column.py <input.csv> <output.csv> <column_name>")
        print('Example: python remove_column.py data.csv cleaned.csv "Notes"')
        sys.exit(1)

    _, input_file, output_file, column = sys.argv
    remove_column(input_file, output_file, column)