#from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def main():
    # Run this after running Heuristic_Gradient_MOO.py
    # It will plot the results and attempt to create a pareto front.
    downloads_path = Path("./MultiObjective")
    ga_csv_path = downloads_path / "prelim_GA_lambda_sweep_results.csv"
    gradient_csv_path = downloads_path / "post_gradient_lambda_sweep_results.csv"
    lawnmower_csv_path = downloads_path / "lawnmower_search.csv"

    required_columns = ["Flight Time [hr]", "Thrust to Weight"]

    fig, ax = plt.subplots(figsize=(8, 6))

    if ga_csv_path.exists():
        ga_df = pd.read_csv(ga_csv_path)
        ga_df = ga_df.dropna(subset=required_columns)
        ax.plot(
            ga_df["Flight Time [hr]"],
            ga_df["Thrust to Weight"],
            marker="*",
            linestyle="-",
            linewidth=1.5,
            color="darkred",
            markersize=10,
            label="Heuristic (Genetic) method"
        )

    if lawnmower_csv_path.exists():
        lawnmower_df = pd.read_csv(lawnmower_csv_path)
        lawnmower_plot_df = lawnmower_df.dropna(subset=required_columns)
        ax.plot(
            lawnmower_plot_df["Flight Time [hr]"],
            lawnmower_plot_df["Thrust to Weight"],
            marker="*",
            linestyle="None",
            color="black",
            markersize=7,
            label="Lawnmower Search"
        )

    if gradient_csv_path.exists():
        gradient_df = pd.read_csv(gradient_csv_path)
        gradient_plot_df = gradient_df.dropna(subset=required_columns).sort_values("Flight Time [hr]")
        ax.plot(
            gradient_plot_df["Flight Time [hr]"],
            gradient_plot_df["Thrust to Weight"],
            marker="*",
            linestyle="-",
            linewidth=1.5,
            color="green",
            markersize=10,
            label="Gradient (SQP) method"
        )

    if "Lambda" in ga_df.columns:
        for _, row in ga_df.iterrows():
            ax.annotate(
                f"{row['Lambda']:.2f}",
                (row["Flight Time [hr]"], row["Thrust to Weight"]),
                textcoords="offset points",
                xytext=(5, 5)
            )

    if "Lambda" in gradient_df.columns:
        for _, row in gradient_df.iterrows():
            ax.annotate(
                f"{row['Lambda']:.2f}",
                (row["Flight Time [hr]"], row["Thrust to Weight"]),
                textcoords="offset points",
                xytext=(5, 5)
            )

    ax.set_xlabel("Flight Time [hr]")
    ax.set_ylabel("Thrust to Weight")
    ax.set_title("Pareto Front Estimation")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    ax.legend()

    plot_path = downloads_path / "pareto_front_comparison.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved Pareto front plot to: {plot_path}")

    plt.show()


if __name__ == "__main__":
    main()
