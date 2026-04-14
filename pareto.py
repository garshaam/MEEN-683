import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def pareto_survivor_mask(points_2d: np.ndarray) -> np.ndarray:
    """Return mask of non-dominated points when maximizing both objectives."""
    n_points = len(points_2d)
    survivors = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        if not survivors[i]:
            continue
        for j in range(n_points):
            if i == j:
                continue
            if (
                (points_2d[j, 0] >= points_2d[i, 0])
                and (points_2d[j, 1] >= points_2d[i, 1])
                and (
                    (points_2d[j, 0] > points_2d[i, 0])
                    or (points_2d[j, 1] > points_2d[i, 1])
                )
            ):
                survivors[i] = False
                break

    return survivors


def plot_points(ax, data_frames, required_columns):
    plot_styles = {
        "Lawnmower Search": {
            "color": "black",
            "marker": "*",
            "linestyle": "None",
            "linewidth": 1,
            "markersize": 7,
            "annotation_col": None,
        },
        "Heuristic (Genetic) method": {
            "color": "darkred",
            "marker": "*",
            "linestyle": "-",
            "linewidth": 1,
            "markersize": 10,
            "annotation_col": "Lambda",
        },
        "Gradient (SQP) method": {
            "color": "green",
            "marker": "*",
            "linestyle": "-",
            "linewidth": 1,
            "markersize": 10,
            "annotation_col": "Lambda",
        },
        "NBI (also SQP) method": {
            "color": "blue",
            "marker": "*",
            "linestyle": "-",
            "linewidth": 2,
            "markersize": 15,
            "annotation_col": "Beta",
        },
    }

    plotted_any = False

    for method_name, df in data_frames.items():
        if df.empty:
            continue

        style = plot_styles[method_name]
        plot_df = df.dropna(subset=required_columns)
        if plot_df.empty:
            continue

        if style["linestyle"] != "None":
            plot_df = plot_df.sort_values("Flight Time [hr]")

        ax.plot(
            plot_df["Flight Time [hr]"],
            plot_df["Thrust to Weight"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            color=style["color"],
            markersize=style["markersize"],
            label=method_name,
        )
        plotted_any = True

        annotation_col = style["annotation_col"]
        if annotation_col and annotation_col in plot_df.columns:
            for _, row in plot_df.iterrows():
                ax.annotate(
                    f"{row[annotation_col]:.2f}",
                    (row["Flight Time [hr]"], row["Thrust to Weight"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                )

    return plotted_any


def main():
    # Run this after running Heuristic_Gradient_MOO.py
    # It will plot the results and attempt to create a pareto front.
    downloads_path = Path("./MultiObjective")
    ga_csv_path = downloads_path / "prelim_GA_lambda_sweep_results.csv"
    gradient_csv_path = downloads_path / "post_gradient_lambda_sweep_results.csv"
    NBI_csv_path = downloads_path / "normal_boundary_intersection_results.csv"
    lawnmower_csv_path = downloads_path / "lawnmower_search.csv"

    required_columns = ["Flight Time [hr]", "Thrust to Weight"]

    csv_sources = {
        "Lawnmower Search": lawnmower_csv_path,
        "Heuristic (Genetic) method": ga_csv_path,
        "Gradient (SQP) method": gradient_csv_path,
        "NBI (also SQP) method": NBI_csv_path,
    }

    method_frames = {}
    all_points = []

    for method_name, csv_path in csv_sources.items():
        if not csv_path.exists():
            method_frames[method_name] = pd.DataFrame(columns=required_columns)
            continue

        df = pd.read_csv(csv_path).dropna(subset=required_columns)
        method_frames[method_name] = df

        if not df.empty:
            tagged = df.copy()
            tagged["Method"] = method_name
            all_points.append(tagged)

    fig, ax = plt.subplots(figsize=(8, 6))
    plotted_any = plot_points(ax, method_frames, required_columns)
    ax.set_xlabel("Flight Time [hr]")
    ax.set_ylabel("Thrust to Weight")
    ax.set_title("All points")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if plotted_any:
        ax.legend()

    plot_path = downloads_path / "pareto_front_comparison.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved Pareto front plot to: {plot_path}")

    if all_points:
        all_points_df = pd.concat(all_points, ignore_index=True)
        objective_points = all_points_df[required_columns].to_numpy(dtype=float)
        survivors_mask = pareto_survivor_mask(objective_points)
        pareto_df = all_points_df.loc[survivors_mask].copy()
    else:
        pareto_df = pd.DataFrame(columns=required_columns + ["Method"])

    pareto_method_frames = {
        method_name: pareto_df[pareto_df["Method"] == method_name]
        for method_name in csv_sources.keys()
    }

    fig_pareto, ax_pareto = plt.subplots(figsize=(8, 6))
    pareto_plotted_any = plot_points(ax_pareto, pareto_method_frames, required_columns)
    ax_pareto.set_xlabel("Flight Time [hr]")
    ax_pareto.set_ylabel("Thrust to Weight")
    ax_pareto.set_title("Pareto-Surviving Points Only")
    ax_pareto.grid(True, alpha=0.3)
    fig_pareto.tight_layout()
    if pareto_plotted_any:
        ax_pareto.legend()

    pareto_plot_path = downloads_path / "pareto_surviving_points.png"
    fig_pareto.savefig(pareto_plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved Pareto-surviving points plot to: {pareto_plot_path}")

    plt.show()


if __name__ == "__main__":
    main()
