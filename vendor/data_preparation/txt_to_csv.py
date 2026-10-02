"""Convert the public trajectory text files to CSV files."""

import argparse
import csv
import re
from pathlib import Path


EXPECTED_COLUMNS = [
    "Frame", "Timer",
    "HeadX", "HeadY", "HeadZ", "HeadRX", "HeadRY", "HeadRZ",
    "LHandX", "LHandY", "LHandZ", "LHandRX", "LHandRY", "LHandRZ",
    "RHandX", "RHandY", "RHandZ", "RHandRX", "RHandRY", "RHandRZ",
    "LEyeRX", "LEyeRY", "LEyeRZ", "REyeRX", "REyeRY", "REyeRZ",
    "LEyeConf", "REyeConf",
]
EXPECTED_COLUMN_COUNT = len(EXPECTED_COLUMNS)
SPLIT_PATTERN = re.compile(r"[,\s]+")


def tokenize(line: str):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    return SPLIT_PATTERN.split(line)


def validate_and_read_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line_number, raw_line in enumerate(file, start=1):
            tokens = tokenize(raw_line)
            if tokens is None:
                continue
            if len(tokens) != EXPECTED_COLUMN_COUNT:
                reason = (
                    f"line {line_number}: expected {EXPECTED_COLUMN_COUNT} "
                    f"columns, got {len(tokens)}"
                )
                return False, reason
            try:
                [float(value) for value in tokens]
            except ValueError:
                return False, f"line {line_number}: non-numeric token found"
            rows.append(tokens)

    if not rows:
        return False, "no data rows"
    return True, rows


def write_csv(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(EXPECTED_COLUMNS)
        writer.writerows(rows)


def convert_folder(input_dir: Path, output_dir: Path):
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    input_files = sorted(input_dir.glob("*.txt"))
    if not input_files:
        raise FileNotFoundError(f"No TXT files found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    skipped = []
    converted = 0

    for input_path in input_files:
        is_valid, payload = validate_and_read_rows(input_path)
        if not is_valid:
            skipped.append((input_path.name, payload))
            print(f"[SKIP] {input_path.name}: {payload}")
            continue

        output_path = output_dir / f"{input_path.stem}.csv"
        write_csv(payload, output_path)
        converted += 1
        print(f"[OK] {input_path.name} -> {output_path.name} ({len(payload)} rows)")

    report_path = output_dir / "skipped_files.txt"
    if skipped:
        with report_path.open("w", encoding="utf-8") as file:
            for file_name, reason in skipped:
                file.write(f"{file_name}\t{reason}\n")
        print(f"[INFO] Skipped-file report: {report_path}")
    elif report_path.exists():
        report_path.unlink()

    if converted == 0:
        raise RuntimeError("No valid trajectory files were converted.")
    print(f"[DONE] Converted={converted}, Skipped={len(skipped)}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    convert_folder(arguments.input_dir.expanduser(), arguments.output_dir.expanduser())
