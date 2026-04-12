import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.lhs import LHS
from pymoo.termination import get_termination
from Project_simulation import full_simulation  
from Project_simulation import get_motor_from_kv
from Project_simulation import battery_calcs
from Project_simulation import full_simulationWT
from pathlib import Path
from scipy.optimize import minimize as scipy_minimize

plt.ion()

downloads_path = Path("./MultiObjective")
downloads_path.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD DOE DATA
# =========================
df = pd.read_csv(r".\feasible_points.csv")
def map_airfoil_to_number(airfoil_name):
    airfoil_list = [
        "xfoild_cl_cd_Re5000",
        "NACA_4412",
        "NACA_2412",
        "NACA_0012",
        "E63",
        "Selig_S1223",
        "SD7037",
        "HN1033",
        "MH30_7_DOE",
        "NACA2412_DOE",
        "RG-15_8_DOE",
        "SD7037_DOE",
        "SD7062_DOE"
    ]
    try:
        return airfoil_list.index(airfoil_name)
    except ValueError:
        return -1  # Return -1 if airfoil name is not found





# Filter feasible
df = df[df["Thrust to Weight"] >= 1.5]

# Take top 3
#top3 = df.head(3)
top3 = df.sample(n=3)
avg_TWR = df["Thrust to Weight"].mean()
avg_FT = df["Flight Time [hr]"].mean()
print(avg_TWR)
print(avg_FT)
X_seed = top3[[
    "Motor", "Battery", "Diameter", "Root Chord", "Tip Chord",
    "Root Pitch", "Tip Pitch", "Airfoil"
]].values

print("Top 3 seed designs:")
print(X_seed)

for x in X_seed:
    try:
        # Convert airfoil string 
        x[7] = int(map_airfoil_to_number(x[7]))
    
    except Exception as e:
        print("Index failed:", e)
# =========================
# DEFINE BOUNDS (LOCAL SEARCH)
# =========================
# Create bounds around top 3 points
lower = np.min(X_seed, axis=0)
upper = np.max(X_seed, axis=0)
buffer = 0.1
xl = lower * (1 - buffer)
xu = upper * (1 + buffer)

# Fix discrete vars
xl[0], xu[0] = 2300, 3800   # kv range
xl[1], xu[1] = 1, 4         # Np
xl[7], xu[7] = 0, 12        # airfoil index

lawnmower_search_columns = [
    "Lambda", "Motor", "Battery", "Diameter", "Root Chord", "Tip Chord",
    "Root Pitch", "Tip Pitch", "Airfoil", "WeightedObj", "ConstraintG",
    "Flight Time [hr]", "Thrust to Weight"
]
lawnmower_search_buffer = []
lawnmower_search_path = downloads_path / "lawnmower_search.csv"
lawnmower_flush_size = 5


def flush_lawnmower_search(force=False):
    global lawnmower_search_buffer

    if not lawnmower_search_buffer:
        return

    if not force and len(lawnmower_search_buffer) < lawnmower_flush_size:
        return

    pd.DataFrame(lawnmower_search_buffer, columns=lawnmower_search_columns).to_csv(
        lawnmower_search_path,
        mode="a",
        header=not lawnmower_search_path.exists(),
        index=False
    )
    lawnmower_search_buffer = []

# =========================
# PROBLEM DEFINITION
# =========================
class DroneOptimization(Problem):

    def __init__(self,ld):
        self.ld = ld
        super().__init__(
            n_var=8,
            n_obj=1,
            n_constr=1,
            xl=xl,
            xu=xu
        )

    def evaluate_design(self, x):
        try:
            mass, flight_time, TWR = full_simulation(x)
            print("TWR:", TWR)

            norm_ft = flight_time / avg_FT
            norm_twr = TWR / avg_TWR

            evaluation = {
                "F": -((norm_ft * self.ld) + (norm_twr * (1 - self.ld))),
                "G": 1.5 - TWR,
                "FT": flight_time,
                "TWR": TWR
            }
        except Exception as e:
            print("FAILED x:", x)
            print("Error:", e)

            evaluation = {
                "F": 1e6,
                "G": 1e6,
                "FT": np.nan,
                "TWR": np.nan
            }

        global lawnmower_search_buffer
        lawnmower_row = {
            "Lambda": self.ld,
            "Motor": x[0] if len(x) > 0 else np.nan,
            "Battery": x[1] if len(x) > 1 else np.nan,
            "Diameter": x[2] if len(x) > 2 else np.nan,
            "Root Chord": x[3] if len(x) > 3 else np.nan,
            "Tip Chord": x[4] if len(x) > 4 else np.nan,
            "Root Pitch": x[5] if len(x) > 5 else np.nan,
            "Tip Pitch": x[6] if len(x) > 6 else np.nan,
            "Airfoil": x[7] if len(x) > 7 else np.nan,
            "WeightedObj": evaluation["F"],
            "ConstraintG": evaluation["G"],
            "Flight Time [hr]": evaluation["FT"],
            "Thrust to Weight": evaluation["TWR"]
        }
        lawnmower_search_buffer.append(lawnmower_row)
        flush_lawnmower_search()

        return evaluation

    def _evaluate(self, X, out, *args, **kwargs):
        
        f = []
        g = []
        ft_vals = []
        twr_vals = []
        for x in X:
            evaluation = self.evaluate_design(x)

            f.append(evaluation["F"])
            g.append(evaluation["G"])
            # storing individual objectives separately makes it easier to
            # draw the pareto front later
            ft_vals.append(evaluation["FT"])
            twr_vals.append(evaluation["TWR"])

        out["F"] = np.array(f)
        out["G"] = np.array(g)
        out["FT"] = np.array(ft_vals)
        out["TWR"] = np.array(twr_vals)


# =========================
# Sampling
# =========================
class SeedSampling(FloatRandomSampling):
    def _do(self, problem, n_samples, **kwargs):
        # Start with top 3
        samples = X_seed.copy()

        # Fill rest randomly
        remaining = n_samples - len(samples)
        if remaining > 0:
            rand = super()._do(problem, remaining, **kwargs)
            samples = np.vstack([samples, rand])

        return samples

from pymoo.core.callback import Callback

class LivePlotCallback(Callback):
    def __init__(self):
        super().__init__()
        self.best_history = []

        # Setup plot
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], marker='o')
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Best Weighted Objective Sum")
        self.ax.set_title("GA Convergence")

    def notify(self, algorithm):
        gen = algorithm.n_gen
        F = algorithm.pop.get("F")

        best = np.min(F)

        # candidate value
        candidate = -best

        # mask values that fall in valid range
        valid = F[( -F >= 0 ) & ( -F <= 50 )]
        print(F)
        print(valid)
        if 0 <= candidate <= 50:
            best_TWRFT = candidate
            print("OPTION 1")
        else:
            if len(valid) > 0:
                best_TWRFT = np.max(valid)
                print("OPTION 2")
            else:
                best_TWRFT = np.max(F)
                print("OPTION 3")  # fallback if nothing valid exists
        
        self.best_history.append(best_TWRFT)

        # Update plot data
        self.line.set_xdata(range(1, len(self.best_history) + 1))
        self.line.set_ydata(self.best_history)

        self.ax.relim()
        self.ax.autoscale_view()

        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        print(f"Gen {gen}: Best TWRFT = {best_TWRFT:.4f}")


def finalize_solution_metrics(ld, best_x, best_obj, metrics, use_topology_at_end):
    if use_topology_at_end:
        total_mass_kg, aux_mass, topo_flight_time_hr, topo_thrust_to_weight = full_simulationWT(best_x)
        # Only proceed with the topo results if the structures mass is < 250 grams
        use_topology_metrics = (total_mass_kg - aux_mass) < 0.25

        if use_topology_metrics:
            metrics["flight_time_hr"] = topo_flight_time_hr
            metrics["thrust_to_weight"] = topo_thrust_to_weight
            metrics["used_topology_metrics"] = True
            best_obj = -(((topo_flight_time_hr / avg_FT) * ld) + ((topo_thrust_to_weight / avg_TWR) * (1 - ld)))
            print("TOPO_BESTOBJ", best_obj)

    return best_obj, metrics


def run_ga_for_lambda(ld, use_topology_at_end=False):

    problem = DroneOptimization(ld)

    algorithm = GA(
        pop_size=8,   #GA PARAM 1
        n_offsprings=8,  #GA PARAM 2
        sampling=SeedSampling(),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", 5)   #GA PARAM 3

    callback = LivePlotCallback()

    res = pymoo_minimize(
        problem,
        algorithm,
        termination,
        seed=1,
        verbose=True,
        callback=callback
    )
    plot_path = downloads_path / f"ga_lambda_{ld:.2f}.png"
    callback.fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

    best_x = res.X
    best_obj = float(np.atleast_1d(res.F)[0])
    print("CurrentBESTOBJ", best_obj)

    opt_ft = np.atleast_1d(res.opt.get("FT"))
    opt_twr = np.atleast_1d(res.opt.get("TWR"))
    metrics = {
        "flight_time_hr": float(opt_ft[0]),
        "thrust_to_weight": float(opt_twr[0]),
        "used_topology_metrics": False
    }
    print("PRE_TOPO_BESTOBJ", best_obj)

    best_obj, metrics = finalize_solution_metrics(
        ld,
        best_x,
        best_obj,
        metrics,
        use_topology_at_end
    )

    return best_x, best_obj, callback.best_history, metrics

def run_gradient_method_for_lambda(ld, starting_x=None, use_topology_at_end=False):

    problem = DroneOptimization(ld)

    fixed_discrete_variables = {
        "motor": starting_x[0],
        "battery": starting_x[1],
        "airfoil": starting_x[7]
    }

    def assemble_full_x(x_continuous):
        return np.array([
            fixed_discrete_variables["motor"],
            fixed_discrete_variables["battery"],
            x_continuous[0],
            x_continuous[1],
            x_continuous[2],
            x_continuous[3],
            x_continuous[4],
            fixed_discrete_variables["airfoil"]
        ], dtype=float)

    evaluation_cache = {
        "x_continuous": None,
        "x_full": None,
        "evaluation": None
    }

    def evaluate_continuous_cached(x_cont):
        x_cont = np.array(x_cont, dtype=float)

        if (
            evaluation_cache["x_continuous"] is not None
            and np.array_equal(evaluation_cache["x_continuous"], x_cont)
        ):
            return evaluation_cache["evaluation"]

        x_full = assemble_full_x(x_cont)
        evaluation = problem.evaluate_design(x_full)
        evaluation_cache["x_continuous"] = x_cont.copy()
        evaluation_cache["x_full"] = x_full
        evaluation_cache["evaluation"] = evaluation

        return evaluation

    def objective_continuous(x_cont):
        return evaluate_continuous_cached(x_cont)["F"]

    def twr_constraint(x_cont):
        return -evaluate_continuous_cached(x_cont)["G"]

    bounds = list(zip(problem.xl[2:7], problem.xu[2:7]))

    res = scipy_minimize(
        objective_continuous,
        starting_x[2:7],
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": twr_constraint}],
        options={"maxiter": 3, "disp": True}
    )

    evaluation = evaluate_continuous_cached(res.x)
    best_x = evaluation_cache["x_full"].copy()
    best_obj = float(evaluation["F"])
    metrics = {
        "flight_time_hr": float(evaluation["FT"]),
        "thrust_to_weight": float(evaluation["TWR"]),
        "used_topology_metrics": False
    }
    history = [-best_obj]

    print("PRE_TOPO_BESTOBJ", best_obj)

    best_obj, metrics = finalize_solution_metrics(
        ld,
        best_x,
        best_obj,
        metrics,
        use_topology_at_end
    )

    return best_x, best_obj, history, metrics

num_lambdas = 20
lambda_min = 0.1
lambda_max = 1
lambda_values = np.linspace(lambda_min, lambda_max, num_lambdas)    # CHANGE 3rd VALUE FOR HIGHER WS DISCRETIZATION

all_results = []

columns = [
    "Motor", "Battery", "Diameter", "Root Chord", "Tip Chord",
    "Root Pitch", "Tip Pitch", "Airfoil",
    "Lambda", "WeightedObjSum", "Flight Time [hr]", "Thrust to Weight", "Runtime [min]"
]

# ga_best_by_lambda = []

# for ld in lambda_values:
#     print(f"\n===== Running GA for lambda = {ld:.2f} =====")
#     start_time = time.time()
#     best_x, best_obj, history, metrics = run_ga_for_lambda(ld, use_topology_at_end=False)
#     runtime = (time.time() - start_time) / 60
#     best_score = -best_obj  # convert back
#     result_row = list(best_x) + [
#         ld,
#         best_score,
#         metrics["flight_time_hr"],
#         metrics["thrust_to_weight"],
#         runtime
#     ]

#     all_results.append(result_row)
#     ga_best_by_lambda.append(best_x)

# GA_results_df = pd.DataFrame(all_results, columns=columns)
# GA_csv_path = downloads_path / "prelim_GA_lambda_sweep_results.csv"
# GA_results_df.to_csv(GA_csv_path, index=False)

# print(f"\nSaved preliminary GA lambda sweep results to: {GA_csv_path}")
print("Running SQP gradient method now...")

GA_csv_path = downloads_path / "prelim_GA_lambda_sweep_results.csv"
ga_seed_columns = [
    "Motor", "Battery", "Diameter", "Root Chord", "Tip Chord",
    "Root Pitch", "Tip Pitch", "Airfoil"
]

ga_results_df = pd.read_csv(GA_csv_path).sort_values("Lambda").reset_index(drop=True)
ga_best_by_lambda = [
    ga_results_df.loc[i, ga_seed_columns].to_numpy(dtype=float)
    for i in range(len(ga_results_df))
]

all_results = []
lambda_key = lambda value: round(float(value), 10)

# Temporary only because of long runtime duration:
# recover the best feasible result seen so far for each lambda from the lawnmower search log,
# seed the output with those rows, and skip rerunning those lambdas.
completed_lambda_keys = set()
if lawnmower_search_path.exists():
    recovered_df = pd.read_csv(lawnmower_search_path)
    recovered_df = recovered_df[recovered_df["ConstraintG"] <= 0].copy()

    if not recovered_df.empty:
        recovered_df["LambdaKey"] = recovered_df["Lambda"].apply(lambda_key)
        recovered_best_idx = recovered_df.groupby("LambdaKey")["WeightedObj"].idxmin()
        recovered_best_df = recovered_df.loc[recovered_best_idx].sort_values("Lambda")

        for _, row in recovered_best_df.iterrows():
            row_lambda = float(row["Lambda"])
            row_key = lambda_key(row_lambda)

            if row_key not in {lambda_key(ld) for ld in lambda_values}:
                continue

            completed_lambda_keys.add(row_key)
            all_results.append([
                float(row["Motor"]),
                float(row["Battery"]),
                float(row["Diameter"]),
                float(row["Root Chord"]),
                float(row["Tip Chord"]),
                float(row["Root Pitch"]),
                float(row["Tip Pitch"]),
                float(row["Airfoil"]),
                row_lambda,
                float(-row["WeightedObj"]),
                float(row["Flight Time [hr]"]),
                float(row["Thrust to Weight"]),
                0.0
            ])

for ld in lambda_values:
    if lambda_key(ld) in completed_lambda_keys:
        print(f"\n===== Skipping lambda = {ld:.2f}; recovered from lawnmower_search.csv =====")
        continue

    print(f"\n===== Running gradient method for lambda = {ld:.2f} =====")
    start_time = time.time()
    best_x, best_obj, history, metrics = run_gradient_method_for_lambda(ld, ga_best_by_lambda[round(ld/lambda_max*num_lambdas-1)], use_topology_at_end=False)
    runtime = (time.time() - start_time) / 60
    best_score = -best_obj  # convert back
    result_row = list(best_x) + [
        ld,
        best_score,
        metrics["flight_time_hr"],
        metrics["thrust_to_weight"],
        runtime
    ]

    all_results.append(result_row)

Gradient_results_df = pd.DataFrame(all_results, columns=columns).sort_values("Lambda").reset_index(drop=True)
Gradient_csv_path = downloads_path / "post_gradient_lambda_sweep_results.csv"
Gradient_results_df.to_csv(Gradient_csv_path, index=False)

print(f"\nSaved gradient method lambda sweep results to: {Gradient_csv_path}")
flush_lawnmower_search(force=True)
