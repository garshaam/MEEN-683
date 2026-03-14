from ansys.mechanical.core import launch_mechanical

# =========================================================
# LAUNCH MECHANICAL ONCE
# =========================================================

mechanical = launch_mechanical(batch=True, cleanup_on_exit=False, transport_mode="insecure")

wbpj_path = r"C:\Users\matt6\Downloads\drone_model_default_files\dp0\global\MECH\SYS.mechdb"
export_path = r"C:\Users\matt6\Downloads\optimized_drone.stl"
mass_file = r"C:\Users\matt6\Downloads\mass_output.txt"

# Load the Workbench project once
init_script = f"""
ExtAPI.Application.OpenProject(r"{wbpj_path}")
"""
mechanical.run_python_script(init_script)


# =========================================================
# TOPOLOGY STUDY FUNCTION
# =========================================================

def TopoStudy(PT, MM, BM):

    Prop_Thrust = PT
    Motor_Mass = MM
    Battery_Mass = BM

    AUX_Mass = 0.200
    Hardware_Mass = 0.010

    g = 9.81

    Central_Force = (AUX_Mass + Battery_Mass + Hardware_Mass/2) * g
    Quarter_Force = Prop_Thrust - (Motor_Mass * g)

    # -------------------------------
    # Python script to run in Mechanical
    # -------------------------------
    script = f"""
analysis = Model.Analyses[0]
solution = analysis.Solution

# --------------------------------
# Remove only previous Forces and Topology
# --------------------------------
for obj in list(analysis.Children):
    if obj.GetType().Name == "Force":
        obj.Delete()

for obj in list(solution.Children):
    if obj.GetType().Name == "TopologyOptimization":
        obj.Delete()

# --------------------------------
# Named Selections (already in mechdb)
# --------------------------------
Q1 = Model.NamedSelections["Q1"]
Q2 = Model.NamedSelections["Q2"]
Q3 = Model.NamedSelections["Q3"]
Q4 = Model.NamedSelections["Q4"]
Base = Model.NamedSelections["Base"]

# --------------------------------
# Apply Motor Forces
# --------------------------------
for ns in [Q1, Q2, Q3, Q4]:
    force = analysis.AddForce()
    force.Location = ns
    force.DefineBy = LoadDefineBy.Components
    force.ZComponent.Output.SetDiscreteValue(
        0,
        Quantity("{Quarter_Force} N")
    )

# --------------------------------
# Apply Battery + AUX Weight at Base
# --------------------------------
central_force = analysis.AddForce()
central_force.Location = Base
central_force.DefineBy = LoadDefineBy.Components
central_force.ZComponent.Output.SetDiscreteValue(
    0,
    Quantity("-{Central_Force} N")
)

# --------------------------------
# Add Topology Optimization
# --------------------------------
topo = solution.AddTopologyOptimization()
topo.ObjectiveType = TopologyOptimizationObjectiveType.MinimizeCompliance
topo.VolumeFraction = 0.35

body = Model.Geometry.Children[0]
region = topo.AddOptimizationRegion()
region.Location = body

# Keep motor mounts and base unchanged
for ns in [Q1, Q2, Q3, Q4, Base]:
    keep = topo.AddKeepRegion()
    keep.Location = ns

# --------------------------------
# Add von Mises Stress Constraint (FOS = 1.3)
# --------------------------------
yield_strength = 150e6  # Adjust to actual PA6 CF yield (Pa)
allowable_stress = yield_strength / 1.3
stress_constraint = topo.AddStressConstraint()
stress_constraint.MaximumStress = Quantity(str(allowable_stress) + " Pa")

# --------------------------------
# Solve
# --------------------------------
analysis.Solve()

# --------------------------------
# Extract Mass
# --------------------------------
mass = body.Mass.Value
with open(r"{mass_file}", "w") as f:
    f.write(str(mass))

# --------------------------------
# Export Optimized STL
# --------------------------------
topo_result = solution.Children[0]
topo_result.ExportToSTL(r"{export_path}")
"""

    # Run the script inside Mechanical
    mechanical.run_python_script(script)

    # Read mass from file
    with open(mass_file, "r") as f:
        optimized_mass = float(f.read())

    return optimized_mass, export_path