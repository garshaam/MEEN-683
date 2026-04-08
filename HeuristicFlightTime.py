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

# Sort by flight time
df = df.sort_values("Flight Time [hr]", ascending=False)

# Take top 3
top3 = df.head(3)

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

    def __init__(self):
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

        for x in X:
            try:
                # Cast discrete variables
                x[0] = int(round(x[0]))  # kv
                x[1] = int(round(x[1]))  # Np
                x[7] = int(round(x[7]))  # airfoil idx

                mass, flight_time, TWR = full_simulation(x)

                # Min neg flight
                f.append(-flight_time)

                # Constraint: T/W ≥ 1.5  g ≤ 0
                g.append(1.5 - TWR)

            except:
                # Penalize failed sims
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
        self.ax.set_ylabel("Best Flight Time (hr)")
        self.ax.set_title("GA Convergence")

    def notify(self, algorithm):
        gen = algorithm.n_gen
        F = algorithm.pop.get("F")

        best = np.min(F)
        best_flight_time = -best

        self.best_history.append(best_flight_time)

        # Update plot data
        self.line.set_xdata(range(1, len(self.best_history) + 1))
        self.line.set_ydata(self.best_history)

        self.ax.relim()
        self.ax.autoscale_view()

        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        print(f"Gen {gen}: Best Flight Time = {best_flight_time:.4f} hr")

# =========================
# RUN GA
# =========================
problem = DroneOptimization()

algorithm = GA(
    pop_size=8,  #GA PARAM 1
    n_offsprings=8,   #GA PARAM 2
    sampling=SeedSampling(),
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 8)  #GA PARAM 3

callback = LivePlotCallback()
start_time = time.time()
res = minimize(
    problem,
    algorithm,
    termination,
    seed=1,
    verbose=True,
    callback=callback
)
end_time = time.time()
runtime = (end_time - start_time)/60 #in min
# =========================
# SAVE CONVERGENCE PLOT
# =========================
plot_path = downloads_path / "ga_convergence.png"
callback.fig.savefig(plot_path, dpi=300, bbox_inches='tight')

print(f"Plot saved to: {plot_path}")

# =========================
# Optima
# =========================
best_x = res.X
best_flight_time = -res.F[0]

# =========================
# For Adam
# =========================
columns = [
    "Motor", "Battery", "Diameter", "Root Chord", "Tip Chord",
    "Root Pitch", "Tip Pitch", "Airfoil", "Flight Time [hr]",
    "Runtime [min]"
]


best_x_clean = best_x.copy()
best_x_clean[0] = int(round(best_x_clean[0]))  # Motor
best_x_clean[1] = int(round(best_x_clean[1]))  # Battery
airfoil_list = [
    "xfoild_cl_cd_Re5000","NACA_4412","NACA_2412","NACA_0012","E63",
    "Selig_S1223","SD7037","HN1033","MH30_7_DOE","NACA2412_DOE",
    "RG-15_8_DOE","SD7037_DOE","SD7062_DOE"
]




result_row = list(best_x_clean) + [best_flight_time, runtime]


results_df = pd.DataFrame([result_row], columns=columns)

# Save to CSV
csv_path = downloads_path / "best_design_result.csv"
results_df.to_csv(csv_path, index=False)

print(f"CSV saved to: {csv_path}")


print("\n===== OPTIMIZATION RESULT =====")
print("Best design:", best_x)
print("Max flight time (hr):", best_flight_time)
print(f"\nTotal runtime: {runtime:.2f} minutes")