"""Command-line interface for CSV analysis or a reproducible synthetic demo."""
import argparse
import pandas as pd
from .analysis import run_analysis
from .sample import generate_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Path to an observations CSV")
    source.add_argument("--demo", action="store_true", help="Use clearly labeled synthetic observations")
    parser.add_argument("--target", default="pm25_ug_m3")
    parser.add_argument("--date-column", help="Use a chronological holdout and exclude this column")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        frame = generate_sample(seed=args.seed) if args.demo else pd.read_csv(args.input)
        result = run_analysis(frame, args.target, args.output, seed=args.seed,
                              date_column=args.date_column or ("date" if args.demo else None),
                              synthetic=args.demo)
    except (ValueError, OSError, pd.errors.ParserError) as error:
        parser.exit(2, f"Analysis failed: {error}\n")
    print(f"Analysis complete: {args.output}/report.md ({result['train_rows']} train / {result['test_rows']} test)")


if __name__ == "__main__":
    main()
