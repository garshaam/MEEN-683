from pathlib import Path

import pandas as pd


def expanded_bounds(series, expansion_fraction=0.10):
    min_value = float(series.min())
    max_value = float(series.max())

    lower_bound = min_value - abs(min_value) * expansion_fraction
    upper_bound = max_value + abs(max_value) * expansion_fraction

    return min_value, max_value, lower_bound, upper_bound


def main():
    downloads_path = Path("./MultiObjective")
    pareto_points_path = downloads_path / "pareto_points.csv"
    bounds_output_path = downloads_path / "pareto_bounds.csv"

    design_columns = [
        "Motor",
        "Battery",
        "Diameter",
        "Root Chord",
        "Tip Chord",
        "Root Pitch",
        "Tip Pitch",
        "Airfoil",
    ]

    if not pareto_points_path.exists():
        raise FileNotFoundError(
            f"Pareto points file not found: {pareto_points_path}. Run pareto.py first."
        )

    pareto_df = pd.read_csv(pareto_points_path)

    missing_columns = [col for col in design_columns if col not in pareto_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in pareto_points.csv: {missing_columns}")

    clean_df = pareto_df.dropna(subset=design_columns)
    if clean_df.empty:
        raise ValueError("No valid Pareto rows remain after dropping NaNs in design columns.")

    records = []
    for column in design_columns:
        min_value, max_value, lower_bound, upper_bound = expanded_bounds(clean_df[column])
        records.append(
            {
                "Variable": column,
                "MinPareto": min_value,
                "MaxPareto": max_value,
                "LowerBound": lower_bound,
                "UpperBound": upper_bound,
            }
        )

    bounds_df = pd.DataFrame(records)
    bounds_df.to_csv(bounds_output_path, index=False)
    print(f"Saved Pareto-based bounds to: {bounds_output_path}")


if __name__ == "__main__":
    main()
