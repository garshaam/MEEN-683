import pandas as pd

def collect_ga_best_from_file(filepath):
    ga_seed_columns = [
        "Motor", "Battery", "Diameter", "Root Chord", "Tip Chord",
        "Root Pitch", "Tip Pitch", "Airfoil"
    ]

    ga_results_df = pd.read_csv(filepath).sort_values("Lambda").reset_index(drop=True)
    ga_best_by_lambda = [
        ga_results_df.loc[i, ga_seed_columns].to_numpy(dtype=float)
        for i in range(len(ga_results_df))
    ]

    return ga_best_by_lambda

def collect_gradient_best_from_file(filepath):
    gradient_seed_columns = [
        "Motor", "Battery", "Diameter", "Root Chord", "Tip Chord",
        "Root Pitch", "Tip Pitch", "Airfoil"
    ]

    gradient_results_df = pd.read_csv(filepath).sort_values("Lambda").reset_index(drop=True)
    gradient_best_by_lambda = [
        gradient_results_df.loc[i, gradient_seed_columns].to_numpy(dtype=float)
        for i in range(len(gradient_results_df))
    ]

    return gradient_best_by_lambda