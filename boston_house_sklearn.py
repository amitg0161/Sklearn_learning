"""Command-line entry point for the Boston housing analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis import dataset_summary
from data import build_frame, load_housing_data
from modeling import evaluate_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--plots", action="store_true", help="Save EDA plots; requires matplotlib."
    )
    args = parser.parse_args()

    features, target = load_housing_data()
    frame = build_frame(features, target)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_summary(frame).to_csv(args.output_dir / "dataset_summary.csv")
    evaluate_models(features, target).to_csv(args.output_dir / "model_results.csv", index=False)

    if args.plots:
        from eda import save_plots

        save_plots(frame, args.output_dir / "plots")

    print(f"Loaded {len(frame)} rows and {features.shape[1]} features.")
    print(f"Results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()