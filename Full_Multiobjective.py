import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Choice, Integer
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination
from Project_simulation import full_simulation  
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

# =========================
# DEFINE BOUNDS (LOCAL SEARCH)
# =========================
PARETO_BOUNDS_PATH = downloads_path / "pareto_bounds.csv"
ANCHOR_POINTS_PATH = downloads_path / "anchor_points.csv"
DESIGN_VARIABLE_INDEX = {
    "Motor": 0,
    "Battery": 1,
    "Diameter": 2,
    "Root Chord": 3,
    "Tip Chord": 4,
    "Root Pitch": 5,
    "Tip Pitch": 6,
    "Airfoil": 7,
}


def load_bounds_from_pareto_csv(bounds_path):
    if not bounds_path.exists():
        raise FileNotFoundError(
            f"Bounds file not found: {bounds_path}. Run generate_pareto_bounds.py first."
        )

    bounds_df = pd.read_csv(bounds_path)
    required_columns = {"Variable", "LowerBound", "UpperBound"}
    missing_columns = required_columns.difference(bounds_df.columns)
    if missing_columns:
        raise ValueError(f"pareto_bounds.csv missing required columns: {sorted(missing_columns)}")

    xl_local = np.full(8, np.nan, dtype=float)
    xu_local = np.full(8, np.nan, dtype=float)
    missing_variables = []

    for variable_name, variable_idx in DESIGN_VARIABLE_INDEX.items():
        match = bounds_df[bounds_df["Variable"] == variable_name]
        if match.empty:
            missing_variables.append(variable_name)
            continue

        lower_bound = float(match.iloc[0]["LowerBound"])
        upper_bound = float(match.iloc[0]["UpperBound"])
        if lower_bound > upper_bound:
            raise ValueError(
                f"Invalid bounds for {variable_name}: LowerBound ({lower_bound}) > UpperBound ({upper_bound})."
            )

        xl_local[variable_idx] = lower_bound
        xu_local[variable_idx] = upper_bound

    if missing_variables:
        raise ValueError(f"pareto_bounds.csv missing variables: {missing_variables}")

    return xl_local, xu_local


xl, xu = load_bounds_from_pareto_csv(PARETO_BOUNDS_PATH)
print(f"Loaded design bounds from: {PARETO_BOUNDS_PATH}")


def load_anchor_points(anchor_path):
    if not anchor_path.exists():
        raise FileNotFoundError(
            f"Anchor file not found: {anchor_path}. Run generate_anchors.py first."
        )

    anchor_df = pd.read_csv(anchor_path)
    required_columns = {"FT_min", "FT_max", "TWR_min", "TWR_max"}
    missing_columns = required_columns.difference(anchor_df.columns)
    if missing_columns:
        raise ValueError(f"anchor_points.csv missing required columns: {sorted(missing_columns)}")
    if anchor_df.empty:
        raise ValueError("anchor_points.csv is empty.")

    row = anchor_df.iloc[0]
    ft_min = float(row["FT_min"])
    ft_max = float(row["FT_max"])
    twr_min = float(row["TWR_min"])
    twr_max = float(row["TWR_max"])

    if ft_max <= ft_min:
        raise ValueError(f"Invalid FT anchor range: FT_min={ft_min}, FT_max={ft_max}")
    if twr_max <= twr_min:
        raise ValueError(f"Invalid TWR anchor range: TWR_min={twr_min}, TWR_max={twr_max}")

    return ft_min, ft_max, twr_min, twr_max


def normalize_with_fixed_range(value, minimum, maximum):
    return (float(value) - minimum) / (maximum - minimum)


def weighted_objective_from_metrics(ld, flight_time_hr, thrust_to_weight):
    norm_ft = normalize_with_fixed_range(flight_time_hr, FT_MIN, FT_MAX)
    norm_twr = normalize_with_fixed_range(thrust_to_weight, TWR_MIN, TWR_MAX)
    return -((norm_ft * ld) + (norm_twr * (1 - ld)))


FT_MIN, FT_MAX, TWR_MIN, TWR_MAX = load_anchor_points(ANCHOR_POINTS_PATH)
print(
    f"Loaded anchors from: {ANCHOR_POINTS_PATH} | "
    f"FT:[{FT_MIN:.6f}, {FT_MAX:.6f}] TWR:[{TWR_MIN:.6f}, {TWR_MAX:.6f}]"
)

MOTOR_OPTIONS = [2300, 3600, 3800]
BATTERY_OPTIONS = [1, 2, 3, 4]
AIRFOIL_OPTIONS = list(range(13))
SEED_TYPE_DOE = "DOE"
SEED_TYPE_FLIGHT_TIME = "FLIGHT_TIME_SEEDS"
SEED_TYPE_THRUST_TO_WEIGHT = "THRUST_TO_WEIGHT_SEEDS"
VALID_GA_SEED_TYPES = {
    SEED_TYPE_DOE,
    SEED_TYPE_FLIGHT_TIME,
    SEED_TYPE_THRUST_TO_WEIGHT
}
SEED_COLUMNS = [
    "Motor", "Battery", "Diameter", "Root Chord", "Tip Chord",
    "Root Pitch", "Tip Pitch", "Airfoil"
]


def row_to_seed_design(row):
    try:
        motor = int(round(float(row["Motor"])))
        battery = int(round(float(row["Battery"])))

        airfoil_raw = row["Airfoil"]
        if isinstance(airfoil_raw, str):
            mapped_airfoil = map_airfoil_to_number(airfoil_raw)
            if mapped_airfoil != -1:
                airfoil = int(mapped_airfoil)
            else:
                airfoil = int(round(float(airfoil_raw)))
        else:
            airfoil = int(round(float(airfoil_raw)))

        if motor not in MOTOR_OPTIONS or battery not in BATTERY_OPTIONS or airfoil not in AIRFOIL_OPTIONS:
            return None

        return {
            "motor": motor,
            "battery": battery,
            "diameter": float(np.clip(float(row["Diameter"]), xl[2], xu[2])),
            "root_chord": float(np.clip(float(row["Root Chord"]), xl[3], xu[3])),
            "tip_chord": float(np.clip(float(row["Tip Chord"]), xl[4], xu[4])),
            "root_pitch": float(np.clip(float(row["Root Pitch"]), xl[5], xu[5])),
            "tip_pitch": float(np.clip(float(row["Tip Pitch"]), xl[6], xu[6])),
            "airfoil": airfoil,
        }
    except Exception:
        return None


def build_feasible_seed_designs(feasible_df):
    seed_designs = []
    required = SEED_COLUMNS + ["Thrust to Weight"]
    cleaned_df = feasible_df.dropna(subset=required)

    for _, row in cleaned_df.iterrows():
        seed_design = row_to_seed_design(row)
        if seed_design is not None:
            seed_designs.append(seed_design)

    return seed_designs


FEASIBLE_SEED_DESIGNS = build_feasible_seed_designs(df)
print(f"Prepared {len(FEASIBLE_SEED_DESIGNS)} feasible-point seed designs for GA initialization.")

lawnmower_search_columns = [
    "Lambda", "Motor", "Battery", "Diameter", "Root Chord", "Tip Chord",
    "Root Pitch", "Tip Pitch", "Airfoil", "WeightedObj", "ConstraintG",
    "Flight Time [hr]", "Thrust to Weight"
]
lawnmower_search_buffer = []
lawnmower_search_path = downloads_path / "lawnmower_search.csv"
# The lawnmower search is appended to file after "lawnmower_flush_size" number of runs
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


def build_lawnmower_seed_designs(sort_column):
    flush_lawnmower_search(force=True)

    if not lawnmower_search_path.exists():
        return []

    required = SEED_COLUMNS + [sort_column]
    lawnmower_df = pd.read_csv(lawnmower_search_path).dropna(subset=required)
    if lawnmower_df.empty:
        return []

    lawnmower_df = lawnmower_df.sort_values(sort_column, ascending=False)
    lawnmower_df = lawnmower_df.drop_duplicates(subset=SEED_COLUMNS, keep="first")

    seed_designs = []
    for _, row in lawnmower_df.iterrows():
        seed_design = row_to_seed_design(row)
        if seed_design is not None:
            seed_designs.append(seed_design)

    return seed_designs


def get_seed_designs_by_type(seed_type):
    if seed_type == SEED_TYPE_DOE:
        return FEASIBLE_SEED_DESIGNS
    if seed_type == SEED_TYPE_FLIGHT_TIME:
        return build_lawnmower_seed_designs("Flight Time [hr]")
    if seed_type == SEED_TYPE_THRUST_TO_WEIGHT:
        return build_lawnmower_seed_designs("Thrust to Weight")

    raise ValueError(f"Unknown seed_type '{seed_type}'. Valid options are: {sorted(VALID_GA_SEED_TYPES)}")

# =========================
# PROBLEM DEFINITION
# =========================
class DroneOptimization(ElementwiseProblem):

    def __init__(self,ld):
        self.ld = ld
        vars = {
            "motor": Choice(options=MOTOR_OPTIONS),
            "battery": Choice(options=BATTERY_OPTIONS),
            "diameter": Real(bounds=(float(xl[2]), float(xu[2]))),
            "root_chord": Real(bounds=(float(xl[3]), float(xu[3]))),
            "tip_chord": Real(bounds=(float(xl[4]), float(xu[4]))),
            "root_pitch": Real(bounds=(float(xl[5]), float(xu[5]))),
            "tip_pitch": Real(bounds=(float(xl[6]), float(xu[6]))),
            "airfoil": Choice(options=AIRFOIL_OPTIONS),
        }
        super().__init__(
            vars=vars,
            n_obj=1,
            n_ieq_constr=1
        )

    def to_design_vector(self, x):
        if isinstance(x, dict):
            return np.array([
                x["motor"],
                x["battery"],
                x["diameter"],
                x["root_chord"],
                x["tip_chord"],
                x["root_pitch"],
                x["tip_pitch"],
                x["airfoil"],
            ], dtype=float)

        return np.array(x, dtype=float)

    def evaluate_design(self, x):
        x = self.to_design_vector(x)

        try:
            mass, flight_time, TWR = full_simulation(x)
            print("TWR:", TWR)

            evaluation = {
                "F": weighted_objective_from_metrics(self.ld, flight_time, TWR),
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
        evaluation = self.evaluate_design(X)

        out["F"] = evaluation["F"]
        out["G"] = evaluation["G"]
        out["FT"] = evaluation["FT"]
        out["TWR"] = evaluation["TWR"]


class SeedSampling(Sampling):
    def __init__(self, seed_designs):
        super().__init__()
        self.seed_designs = list(seed_designs)

    @staticmethod
    def _to_scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _random_mixed_samples(self, problem, n_samples):
        random_samples = []
        for _ in range(n_samples):
            sample = {}
            for var_name, var in problem.vars.items():
                sampled = var.sample(1)
                if isinstance(sampled, (list, tuple, np.ndarray, pd.Series)):
                    value = sampled[0]
                else:
                    value = sampled
                sample[var_name] = self._to_scalar(value)
            random_samples.append(sample)
        return random_samples

    def _do(self, problem, n_samples, **kwargs):
        if not self.seed_designs:
            return self._random_mixed_samples(problem, n_samples)

        take_count = min(n_samples, len(self.seed_designs))
        seeded_samples = [self.seed_designs[idx].copy() for idx in range(take_count)]

        remaining = n_samples - take_count
        if remaining > 0:
            seeded_samples.extend(self._random_mixed_samples(problem, remaining))

        return seeded_samples

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
            best_obj = weighted_objective_from_metrics(ld, topo_flight_time_hr, topo_thrust_to_weight)
            print("TOPO_BESTOBJ", best_obj)

    return best_obj, metrics

def run_ga_for_lambda(ld, use_topology_at_end=False, seed_type=SEED_TYPE_DOE):
    seed_type = str(seed_type).upper()
    if seed_type not in VALID_GA_SEED_TYPES:
        raise ValueError(f"Unknown seed_type '{seed_type}'. Valid options are: {sorted(VALID_GA_SEED_TYPES)}")

    problem = DroneOptimization(ld)
    seed_designs = get_seed_designs_by_type(seed_type)
    print(f"GA seed type: {seed_type} | seed designs available: {len(seed_designs)}")

    algorithm = MixedVariableGA(
        pop_size=16, # GA PARAM 1
        n_offsprings=10, # GA PARAM 2
        sampling=SeedSampling(seed_designs),
    )

    termination = get_termination("n_gen", 6)   #GA PARAM 3

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

    best_x = problem.to_design_vector(res.X)
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

    bounds = [
        (float(xl[2]), float(xu[2])),
        (float(xl[3]), float(xu[3])),
        (float(xl[4]), float(xu[4])),
        (float(xl[5]), float(xu[5])),
        (float(xl[6]), float(xu[6])),
    ]

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

def run_weighted_sum(num_lambdas, lambda_min=0.1, lambda_max=1, skip_existing_lambdas=True, ga_seed_type=SEED_TYPE_DOE):
    lambda_values = np.linspace(lambda_min, lambda_max, num_lambdas)    # CHANGE 3rd VALUE FOR HIGHER WS DISCRETIZATION

    columns = [
        "Motor", "Battery", "Diameter", "Root Chord", "Tip Chord",
        "Root Pitch", "Tip Pitch", "Airfoil",
        "Lambda", "WeightedObjSum", "Flight Time [hr]", "Thrust to Weight", "Runtime [min]"
    ]

    GA_csv_path = downloads_path / "prelim_GA_lambda_sweep_results.csv"
    Gradient_csv_path = downloads_path / "post_gradient_lambda_sweep_results.csv"

    def lambda_key(value):
        return round(float(value), 10)

    ga_seed_columns = [
        "Motor", "Battery", "Diameter", "Root Chord", "Tip Chord",
        "Root Pitch", "Tip Pitch", "Airfoil"
    ]

    def merge_results(existing_path, new_results):
        frames = []

        if existing_path.exists():
            existing_df = pd.read_csv(existing_path)
            frames.append(existing_df[columns])

        if new_results:
            frames.append(pd.DataFrame(new_results, columns=columns))

        if not frames:
            return pd.DataFrame(columns=columns)

        merged_df = pd.concat(frames, ignore_index=True)
        merged_df["LambdaKey"] = merged_df["Lambda"].apply(lambda_key)
        merged_df = merged_df.drop_duplicates(subset="LambdaKey", keep="first")
        merged_df = merged_df[columns].sort_values("Lambda").reset_index(drop=True)

        return merged_df

    def build_ga_seed_map(results_df):
        ga_seed_map = {}

        for _, row in results_df.iterrows():
            ga_seed_map[lambda_key(row["Lambda"])] = row[ga_seed_columns].to_numpy(dtype=float)

        return ga_seed_map

    GA_results = []
    completed_ga_lambda_keys = set()

    if GA_csv_path.exists():
        ga_existing_df = pd.read_csv(GA_csv_path).sort_values("Lambda").reset_index(drop=True)
        completed_ga_lambda_keys = {lambda_key(value) for value in ga_existing_df["Lambda"]}

    for ld in lambda_values:
        row_key = lambda_key(ld)
        if skip_existing_lambdas and row_key in completed_ga_lambda_keys:
            print(f"\n===== Skipping GA for lambda = {ld:.2f}; already present in {GA_csv_path.name} =====")
            continue

        print(f"\n===== Running GA for lambda = {ld:.2f} =====")
        start_time = time.time()
        best_x, best_obj, history, metrics = run_ga_for_lambda(
            ld,
            use_topology_at_end=False,
            seed_type=ga_seed_type
        )
        runtime = (time.time() - start_time) / 60
        best_score = -best_obj  # convert back
        result_row = list(best_x) + [
            ld,
            best_score,
            metrics["flight_time_hr"],
            metrics["thrust_to_weight"],
            runtime
        ]

        GA_results.append(result_row)
        completed_ga_lambda_keys.add(row_key)

    GA_results_df = merge_results(GA_csv_path, GA_results)
    GA_results_df.to_csv(GA_csv_path, index=False)
    ga_best_by_lambda = build_ga_seed_map(GA_results_df)

    print(f"\nSaved preliminary GA lambda sweep results to: {GA_csv_path}")
    print("Running SQP gradient method now...")

    gradient_results = []
    completed_gradient_lambda_keys = set()

    if Gradient_csv_path.exists():
        gradient_existing_df = pd.read_csv(Gradient_csv_path).sort_values("Lambda").reset_index(drop=True)
        completed_gradient_lambda_keys = {lambda_key(value) for value in gradient_existing_df["Lambda"]}

    missing_ga_seed_keys = [lambda_key(ld) for ld in lambda_values if lambda_key(ld) not in ga_best_by_lambda]
    if missing_ga_seed_keys:
        raise ValueError(f"Missing GA seeds for lambdas: {missing_ga_seed_keys}")

    for ld in lambda_values:
        row_key = lambda_key(ld)
        if skip_existing_lambdas and row_key in completed_gradient_lambda_keys:
            print(f"\n===== Skipping gradient method for lambda = {ld:.2f}; already present in {Gradient_csv_path.name} =====")
            continue

        print(f"\n===== Running gradient method for lambda = {ld:.2f} =====")
        start_time = time.time()
        best_x, best_obj, history, metrics = run_gradient_method_for_lambda(
            ld,
            ga_best_by_lambda[lambda_key(ld)],
            use_topology_at_end=False
        )
        runtime = (time.time() - start_time) / 60
        best_score = -best_obj  # convert back
        result_row = list(best_x) + [
            ld,
            best_score,
            metrics["flight_time_hr"],
            metrics["thrust_to_weight"],
            runtime
        ]

        gradient_results.append(result_row)
        completed_gradient_lambda_keys.add(row_key)

    Gradient_results_df = merge_results(Gradient_csv_path, gradient_results)
    Gradient_results_df.to_csv(Gradient_csv_path, index=False)

    print(f"\nSaved gradient method lambda sweep results to: {Gradient_csv_path}")

def run_normal_boundary_intersection(num_runs, pull_utopia_from_file=True, skip_existing_points=True, use_topology_at_end=False):
    columns = [
        "Motor", "Battery", "Diameter", "Root Chord", "Tip Chord",
        "Root Pitch", "Tip Pitch", "Airfoil",
        "Beta", "NBI_t", "WeightedObjSum", "Flight Time [hr]", "Thrust to Weight", "Runtime [min]"
    ]

    GA_csv_path = downloads_path / "prelim_GA_lambda_sweep_results.csv"
    Gradient_csv_path = downloads_path / "post_gradient_lambda_sweep_results.csv"
    NBI_csv_path = downloads_path / "normal_boundary_intersection_results.csv"

    def beta_key(value):
        return round(float(value), 10)

    def lambda_key(value):
        return round(float(value), 10)

    if not pull_utopia_from_file:
        run_weighted_sum(2, 0, 1, skip_existing_lambdas=False)

    if not Gradient_csv_path.exists():
        raise FileNotFoundError(
            f"Gradient results file not found: {Gradient_csv_path}. "
            "Run weighted sum first, or call this function with pull_utopia_from_file=False."
        )

    gradient_df = pd.read_csv(Gradient_csv_path).sort_values("Lambda").reset_index(drop=True)
    gradient_df["LambdaKey"] = gradient_df["Lambda"].apply(lambda_key)

    ft_span = FT_MAX - FT_MIN
    twr_span = TWR_MAX - TWR_MIN
    if ft_span <= 0 or twr_span <= 0:
        raise ValueError(
            "Cannot form NBI line because anchor_points.csv has a non-positive objective span."
        )

    objective_scale = np.array([1.0 / ft_span, 1.0 / twr_span], dtype=float)
    objective_offset = np.array([FT_MIN, TWR_MIN], dtype=float)

    # In normalized objective space: twr-favored corner and ft-favored corner.
    scaled_twr_anchor = np.array([0.0, 1.0], dtype=float)
    scaled_ft_anchor = np.array([1.0, 0.0], dtype=float)

    line_direction = scaled_ft_anchor - scaled_twr_anchor
    normal = np.array([line_direction[1], -line_direction[0]], dtype=float)
    if np.sum(normal) < 0:
        normal = -normal
    normal = normal / np.linalg.norm(normal)

    seed_columns = [
        "Motor", "Battery", "Diameter", "Root Chord", "Tip Chord",
        "Root Pitch", "Tip Pitch", "Airfoil"
    ]
    nbi_multi_start_count = 5
    lawnmower_candidate_pool_size = 40

    def merge_results(existing_path, new_results):
        frames = []

        if existing_path.exists():
            existing_df = pd.read_csv(existing_path)
            frames.append(existing_df[columns])

        if new_results:
            frames.append(pd.DataFrame(new_results, columns=columns))

        if not frames:
            return pd.DataFrame(columns=columns)

        merged_df = pd.concat(frames, ignore_index=True)
        merged_df["BetaKey"] = merged_df["Beta"].apply(beta_key)
        merged_df = merged_df.drop_duplicates(subset="BetaKey", keep="first")
        merged_df = merged_df[columns].sort_values("Beta").reset_index(drop=True)

        return merged_df

    def nearest_gradient_seed(beta):
        if gradient_df.empty:
            raise ValueError("Gradient results CSV is empty; cannot generate NBI seeds.")
        nearest_idx = (gradient_df["Lambda"] - beta).abs().idxmin()
        return gradient_df.loc[nearest_idx, seed_columns].to_numpy(dtype=float)

    def load_lawnmower_seed_pool():
        flush_lawnmower_search(force=True)

        if not lawnmower_search_path.exists():
            return pd.DataFrame(columns=seed_columns + ["Flight Time [hr]", "Thrust to Weight", "FT_norm", "TWR_norm"])

        required_columns = seed_columns + ["Flight Time [hr]", "Thrust to Weight"]
        lawnmower_df = pd.read_csv(lawnmower_search_path)

        missing_cols = [col for col in required_columns if col not in lawnmower_df.columns]
        if missing_cols:
            return pd.DataFrame(columns=seed_columns + ["Flight Time [hr]", "Thrust to Weight", "FT_norm", "TWR_norm"])

        for col in required_columns:
            lawnmower_df[col] = pd.to_numeric(lawnmower_df[col], errors="coerce")

        lawnmower_df = lawnmower_df.dropna(subset=required_columns)
        lawnmower_df = lawnmower_df[lawnmower_df["Thrust to Weight"] >= 1.5]
        if lawnmower_df.empty:
            return pd.DataFrame(columns=seed_columns + ["Flight Time [hr]", "Thrust to Weight", "FT_norm", "TWR_norm"])

        lawnmower_df["FT_norm"] = (lawnmower_df["Flight Time [hr]"] - objective_offset[0]) * objective_scale[0]
        lawnmower_df["TWR_norm"] = (lawnmower_df["Thrust to Weight"] - objective_offset[1]) * objective_scale[1]
        lawnmower_df = lawnmower_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["FT_norm", "TWR_norm"])
        lawnmower_df = lawnmower_df.drop_duplicates(subset=seed_columns, keep="last").reset_index(drop=True)

        return lawnmower_df

    line_direction_unit = line_direction / np.linalg.norm(line_direction)

    def solve_nbi_from_seed(beta, line_point, starting_x, initial_t):
        problem = DroneOptimization(beta)

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

        def objective_nbi(y):
            return -float(y[-1])

        def twr_constraint(y):
            x_cont = y[:-1]
            return -evaluate_continuous_cached(x_cont)["G"]

        def eq_ft(y):
            x_cont = y[:-1]
            t = float(y[-1])
            evaluation = evaluate_continuous_cached(x_cont)
            scaled_ft = (evaluation["FT"] - objective_offset[0]) * objective_scale[0]
            return scaled_ft - (line_point[0] + t * normal[0])

        def eq_twr(y):
            x_cont = y[:-1]
            t = float(y[-1])
            evaluation = evaluate_continuous_cached(x_cont)
            scaled_twr = (evaluation["TWR"] - objective_offset[1]) * objective_scale[1]
            return scaled_twr - (line_point[1] + t * normal[1])

        bounds = [
            (float(xl[2]), float(xu[2])),
            (float(xl[3]), float(xu[3])),
            (float(xl[4]), float(xu[4])),
            (float(xl[5]), float(xu[5])),
            (float(xl[6]), float(xu[6])),
            (0.0, None),
        ]
        initial_guess = np.concatenate([starting_x[2:7], [max(0.0, float(initial_t))]])

        res = scipy_minimize(
            objective_nbi,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=[
                {"type": "ineq", "fun": twr_constraint},
                {"type": "eq", "fun": eq_ft},
                {"type": "eq", "fun": eq_twr},
            ],
            options={"maxiter": 3, "disp": False}
        )

        evaluation = evaluate_continuous_cached(res.x[:-1])
        best_x = evaluation_cache["x_full"].copy()
        nbi_t = float(res.x[-1])
        ineq_value = float(twr_constraint(res.x))
        eq_error = float(abs(eq_ft(res.x)) + abs(eq_twr(res.x)))
        constraint_violation = float(max(0.0, -ineq_value) + eq_error)
        merit = float(nbi_t - 100.0 * constraint_violation)

        return {
            "best_x": best_x,
            "nbi_t": nbi_t,
            "best_obj": float(evaluation["F"]),
            "flight_time_hr": float(evaluation["FT"]),
            "thrust_to_weight": float(evaluation["TWR"]),
            "constraint_violation": constraint_violation,
            "merit": merit,
        }

    beta_values = np.linspace(0, 1, num_runs)
    nbi_results = []
    completed_beta_keys = set()
    continuation_solution = None

    if NBI_csv_path.exists():
        existing_nbi_df = pd.read_csv(NBI_csv_path).sort_values("Beta").reset_index(drop=True)
        completed_beta_keys = {beta_key(value) for value in existing_nbi_df["Beta"]}

    for beta in beta_values:
        current_beta_key = beta_key(beta)
        if skip_existing_points and current_beta_key in completed_beta_keys:
            print(f"\n===== Skipping NBI beta = {beta:.2f}; already present in {NBI_csv_path.name} =====")
            continue

        print(f"\n===== Running NBI for beta = {beta:.2f} =====")
        start_time = time.time()
        line_point = scaled_twr_anchor + beta * (scaled_ft_anchor - scaled_twr_anchor)

        seed_candidates = []

        if continuation_solution is not None:
            continuation_point = np.array([
                (continuation_solution["flight_time_hr"] - objective_offset[0]) * objective_scale[0],
                (continuation_solution["thrust_to_weight"] - objective_offset[1]) * objective_scale[1]
            ], dtype=float)
            continuation_t0 = float(np.dot(continuation_point - line_point, normal))
            seed_candidates.append({
                "x": continuation_solution["x"].copy(),
                "t0": max(0.0, continuation_t0),
                "source": "continuation"
            })

        lawnmower_seed_df = load_lawnmower_seed_pool()
        if not lawnmower_seed_df.empty:
            delta_ft = lawnmower_seed_df["FT_norm"].to_numpy(dtype=float) - line_point[0]
            delta_twr = lawnmower_seed_df["TWR_norm"].to_numpy(dtype=float) - line_point[1]
            t_projection = delta_ft * normal[0] + delta_twr * normal[1]
            tangent_residual = np.abs(delta_ft * line_direction_unit[0] + delta_twr * line_direction_unit[1])
            seed_rank = tangent_residual + 10.0 * np.maximum(0.0, -t_projection)

            ranked_idx = np.argsort(seed_rank)[:lawnmower_candidate_pool_size]
            for idx in ranked_idx:
                row = lawnmower_seed_df.iloc[int(idx)]
                seed_candidates.append({
                    "x": row[seed_columns].to_numpy(dtype=float),
                    "t0": max(0.0, float(t_projection[idx])),
                    "source": "lawnmower"
                })

        seed_candidates.append({
            "x": nearest_gradient_seed(beta),
            "t0": 0.0,
            "source": "gradient_fallback"
        })

        deduped_candidates = []
        seen_keys = set()
        for candidate in seed_candidates:
            key = tuple(np.round(candidate["x"], 10))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped_candidates.append(candidate)
            if len(deduped_candidates) >= nbi_multi_start_count:
                break

        print(f"NBI beta={beta:.2f} multi-start candidates: {len(deduped_candidates)}")

        best_run = None
        for candidate in deduped_candidates:
            run_result = solve_nbi_from_seed(
                beta=beta,
                line_point=line_point,
                starting_x=candidate["x"],
                initial_t=candidate["t0"]
            )
            if best_run is None or run_result["merit"] > best_run["merit"]:
                best_run = run_result

        if best_run is None:
            print(f"Skipping NBI beta={beta:.2f}: no valid multi-start results.")
            continue

        best_x = best_run["best_x"]
        nbi_t = best_run["nbi_t"]
        best_obj = best_run["best_obj"]
        metrics = {
            "flight_time_hr": best_run["flight_time_hr"],
            "thrust_to_weight": best_run["thrust_to_weight"],
            "used_topology_metrics": False
        }

        best_obj, metrics = finalize_solution_metrics(
            beta,
            best_x,
            best_obj,
            metrics,
            use_topology_at_end
        )

        runtime = (time.time() - start_time) / 60
        nbi_results.append(list(best_x) + [
            beta,
            nbi_t,
            -best_obj,
            metrics["flight_time_hr"],
            metrics["thrust_to_weight"],
            runtime
        ])
        completed_beta_keys.add(current_beta_key)
        continuation_solution = {
            "x": best_x.copy(),
            "flight_time_hr": best_run["flight_time_hr"],
            "thrust_to_weight": best_run["thrust_to_weight"],
        }

    NBI_results_df = merge_results(NBI_csv_path, nbi_results)
    NBI_results_df.to_csv(NBI_csv_path, index=False)

    print(f"\nSaved NBI results to: {NBI_csv_path}")


# Start of program
#run_normal_boundary_intersection(10,False,True,False)
run_weighted_sum(10,0,1,False,SEED_TYPE_FLIGHT_TIME)

flush_lawnmower_search(force=True)
