"""Generate an audit-friendly run passport for a CSV dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import cleanup_temp_files, generate_reproducibility_passport, run_pipeline


def generate_passport(csv_path: str | Path, output_path: str | Path) -> Path:
    """Run the workflow and write its run passport as Markdown."""
    result = run_pipeline(csv_path)
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.write_text(generate_reproducibility_passport(result), encoding="utf-8")
    finally:
        cleanup_temp_files(result)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an AutoWrangler reproducibility passport.")
    parser.add_argument("csv_path", nargs="?", default=str(PROJECT_ROOT / "data" / "messy_data.csv"))
    parser.add_argument("--output", default="reports/reproducibility_passport.md")
    args = parser.parse_args()
    print(generate_passport(args.csv_path, args.output))


if __name__ == "__main__":
    main()
