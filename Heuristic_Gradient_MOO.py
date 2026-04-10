import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.lhs import LHS
from pymoo.termination import get_termination
from Project_simulation import full_simulation  
from Project_simulation import get_motor_from_kv
from Project_simulation import battery_calcs
from Project_simulation import full_simulationWT
from pathlib import Path

plt.ion()

downloads_path = Path.home() / "Downloads"

# =========================
# LOAD DOE DATA
# =========================
df = pd.read_csv(r"C:\Users\matt6\Downloads\feasible_points.csv")
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

    def _evaluate(self, X, out, *args, **kwargs):
        
        f = []
        g = []
        valid_flag = []
        for x in X:
            try:
                mass, flight_time, TWR = full_simulation(x)
                print("TWR:", TWR)
                Norm_FT = flight_time/avg_FT
                Norm_TWR = TWR/avg_TWR
                
                f.append(-((Norm_FT*self.ld)+(Norm_TWR*(1-self.ld))))
                g.append(1.5 - TWR)

            except Exception as e:
                print("FAILED x:", x)
                print("Error:", e)

                f.append(1e6)
                g.append(1e6)

        out["F"] = np.array(f)
        out["G"] = np.array(g)


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



def run_ga_for_lambda(ld):

    problem = DroneOptimization(ld)

    algorithm = GA(
        pop_size=8,   #GA PARAM 1
        n_offsprings=8,  #GA PARAM 2
        sampling=SeedSampling(),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", 5)   #GA PARAM 3

    callback = LivePlotCallback()

    res = minimize(
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
    best_obj = res.F[0]
    print("CurrentBESTOBJ",best_obj)

    total_mass_kg, aux_mass, flight_time_hr, thrust_to_weight = full_simulationWT(best_x)
    if (total_mass_kg-aux_mass)<0.25:
        new_mass = total_mass_kg
        best_obj = -(((flight_time_hr/avg_FT)*ld)+((thrust_to_weight/avg_TWR)*(1-ld)))
        print("W/TOPOBESTOBJ",best_obj)
    else:
        best_obj = best_obj

    

    return best_x, best_obj, callback.best_history


lambda_values = np.linspace(0.1, 1, 10)    # CHANGE 3rd VALUE FOR HIGHER WS DISCRETIZATION

all_results = []

for ld in lambda_values:
    print(f"\n===== Running GA for lambda = {ld:.2f} =====")

    start_time = time.time()

    best_x, best_obj, history = run_ga_for_lambda(ld)

    runtime = (time.time() - start_time) / 60

    best_score = -best_obj  # convert back

    result_row = list(best_x) + [ld, best_score, runtime]

    all_results.append(result_row)

columns = [
    "Motor", "Battery", "Diameter", "Root Chord", "Tip Chord",
    "Root Pitch", "Tip Pitch", "Airfoil",
    "Lambda", "WeightedObjSum", "Runtime [min]"
]

results_df = pd.DataFrame(all_results, columns=columns)

csv_path = downloads_path / "lambda_sweep_results.csv"
results_df.to_csv(csv_path, index=False)

print(f"\nSaved lambda sweep results to: {csv_path}")














