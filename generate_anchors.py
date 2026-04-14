from pathlib import Path

import pandas as pd


def main():
    downloads_path = Path("./MultiObjective")
    lawnmower_path = downloads_path / "lawnmower_search.csv"
    anchors_path = downloads_path / "anchor_points.csv"

    if not lawnmower_path.exists():
        raise FileNotFoundError(
            f"Lawnmower data file not found: {lawnmower_path}. Run optimization first."
        )

    required_columns = ["Flight Time [hr]", "Thrust to Weight"]
    df = pd.read_csv(lawnmower_path)

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column in lawnmower_search.csv: {column}")

    numeric_df = df.copy()
    numeric_df["Flight Time [hr]"] = pd.to_numeric(numeric_df["Flight Time [hr]"], errors="coerce")
    numeric_df["Thrust to Weight"] = pd.to_numeric(numeric_df["Thrust to Weight"], errors="coerce")
    numeric_df = numeric_df.dropna(subset=required_columns)

    if numeric_df.empty:
        raise ValueError("No valid lawnmower rows with numeric Flight Time and Thrust to Weight.")

    ft_min = float(numeric_df["Flight Time [hr]"].min())
    ft_max = float(numeric_df["Flight Time [hr]"].max())
    twr_min = float(numeric_df["Thrust to Weight"].min())
    twr_max = float(numeric_df["Thrust to Weight"].max())

    if ft_max <= ft_min:
        raise ValueError(f"Invalid anchor span for Flight Time: min={ft_min}, max={ft_max}")
    if twr_max <= twr_min:
        raise ValueError(f"Invalid anchor span for Thrust to Weight: min={twr_min}, max={twr_max}")

    anchors_df = pd.DataFrame([
        {
            "FT_min": ft_min,
            "FT_max": ft_max,
            "TWR_min": twr_min,
            "TWR_max": twr_max,
        }
    ])
    anchors_df.to_csv(anchors_path, index=False)

    print(f"Saved anchor points to: {anchors_path}")
    print(
        f"FT_min={ft_min:.6f}, FT_max={ft_max:.6f}, "
        f"TWR_min={twr_min:.6f}, TWR_max={twr_max:.6f}"
    )


if __name__ == "__main__":
    main()
