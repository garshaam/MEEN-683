import numpy as np
from pybemt.solver import Solver
from scipy.optimize import brentq
from ansys.mechanical.core import launch_mechanical
# Generate Propeller, which will be used to create .ini file 
def generate_propeller(
    diameter,
    radius_hub,
    chord_root,
    chord_tip,
    pitch_root,
    pitch_tip,
    rpm, # rpm value is preliminary and must be modified iteratively
    airfoil
):
    """
    Full propeller generator that combines: 
        
        Radius generation, 
        Linear chord and pitch distributions,
        Constant airfoil assignment,
        Fluid, 
        Solver ,
        Case configuration,
        Rotor data (nblades, diameter, radius_hub)

    Fixed internal parameters:
        clustering = 1.2
        epsilon = 1e-3
        v_inf = 0
        solver = brute
        rho = 1.225
        mu = 1.81e-5
        nblades = 2
        n_elements = scaled based on previous data set based on blade diameter (40 elements/0.254in)

    """

    # Fixed values
    # Higher clustering puts more elements near the root
    clustering = 1.2
    epsilon = 1e-3

    # Radius generation
    R = diameter / 2
    #scale the number elements based on previous data set
    n_elements = round((40/0.254) * diameter)

    xi = np.linspace(0, 1, n_elements)
    radius = radius_hub + (R - radius_hub) * (xi ** clustering)
    t = (radius - radius_hub) / (R - radius_hub)
    
    # Linear equation modeling
    chord = chord_root + t * (chord_tip - chord_root)
    pitch = pitch_root + t * (pitch_tip - pitch_root)

    # Airfoil (assumed constant along entire blade)
    sections = [airfoil] * n_elements

    # Fluid / Solver / Case Requirements
    fluid = {
        "rho": 1.225,
        "mu": 1.81e-5
    }

    solver = {
        "solver": "brute"
    }

    case = {
        "v_inf": 0.0,
        "rpm": rpm
    }

    # Function Output
    return {
        "nblades": 2,
        "diameter": diameter,
        "radius_hub": radius_hub,
        "radius": radius,
        "chord": chord,
        "pitch": pitch,
        "sections": sections,
        "fluid": fluid,
        "solver": solver,
        "case": case
    }

# .ini file automation 
def write_ini_file(prop, filename):
    """
    Writes the propeller dictionary to a .ini file
    in the exact required format.
    """

    with open(filename, "w") as f:

        # Rotor formatting/sorting
        f.write("[rotor]\n")
        f.write(f"nblades = {prop['nblades']}\n")
        f.write(f"diameter = {prop['diameter']}\n")
        f.write(f"radius_hub = {prop['radius_hub']}\n")

        f.write("radius = " + " ".join(map(str, prop["radius"])) + "\n")
        f.write("chord = " + " ".join(map(str, prop["chord"])) + "\n")
        f.write("pitch = " + " ".join(map(str, prop["pitch"])) + "\n")
        f.write("section = " + " ".join(prop["sections"]) + "\n\n")
        # Extra line needed for formatting reasons
        f.write("\n")
        # Fluid formatting/sorting
        f.write("[fluid]\n")
        f.write(f"rho = {prop['fluid']['rho']}\n")
        f.write(f"mu = {prop['fluid']['mu']}\n\n")

        # Solver formatting/sorting
        f.write("[solver]\n")
        f.write(f"solver = {prop['solver']['solver']}\n\n")

        # Case formatting/sorting
        f.write("[case]\n")
        f.write(f"v_inf = {prop['case']['v_inf']}\n")
        f.write(f"rpm = {prop['case']['rpm']}\n")

# MOTOR #

def get_motor_from_kv(kv):

    motor_db = {
        2300: {'mass_g': 9.1, 'V_rate': 24.0, 'R_winding_ohm': 0.614, 'I_max': 9.3},
        3600: {'mass_g': 9.1, 'V_rate': 16.0, 'R_winding_ohm': 0.35, 'I_max': 12.1},
        3800: {'mass_g': 9.4, 'V_rate': 12.0, 'R_winding_ohm': 0.105, 'I_max': 8.1}
    }

    if kv not in motor_db:
        raise ValueError(f"Kv {kv} not in motor database")

    data = motor_db[kv]

    return {
        'kv_rpm_per_V': kv,
        'kt_NmpA': 60/(2*np.pi*kv),
        'R_winding_ohm': data['R_winding_ohm'],
        'I_max': data['I_max'],
        'V_rate': data['V_rate'],
        'mass_g': data['mass_g']
    }

def motor_model(kv, Q, battery):

    motor = get_motor_from_kv(kv)

    kt = motor['kt_NmpA']
    kv = motor['kv_rpm_per_V']
    R_winding = motor['R_winding_ohm']
    V_rate = motor['V_rate']
    I_max = motor['I_max']

    Np = battery['Np']
    #V_batt does not need to equal V_rate but it cannot be higher
    V_batt = battery['V_batt']
    R_batt = battery['R_batt']
    C_cell_Ah = battery['C_cell_Ah']
    m_cell_kg = battery['m_cell_kg']

    # Current limit check
    if V_batt > V_rate:
        raise ValueError(f"Battery voltage too high")

    # Current from torque (used internally)
    I = Q / kt

    # For 4 motors
    I_total = I * 4

    # Current limit check
    if I > I_max:
        raise ValueError(f"Overcurrent: I={I:.2f}A exceeds motor limit I_max={I_max:.2f}A")
    
    # Battery terminal voltage
    # Which is motor rated voltage minus drop due to battery resistance
    Vterm = V_batt - I_total * R_batt

    # Voltage constraint check
    V_motor = Vterm - I * R_winding

    if V_motor <= 0:
        raise ValueError(f"Voltage collapse: V_motor={V_motor:.2f}V (I too large, torque too high)")

    # Electrical RPM
    # Back EMF Equation Used:
    # Assuming Voc = Vrate
    # RPM = Kv (V_rate - (Q/Kt)Rbattery - (Q/Kt)Rwinding)
    RPM = kv * V_motor
    # equivalent to RPM = kv * V_motor

    # RPM CHECKS!
    if RPM <= 0:
        raise ValueError(f"Invalid RPM computed: {RPM:.2f}")
    if np.isnan(RPM) or np.isinf(RPM):
        raise ValueError("RPM became NaN or Inf")
    
    # Angular velocity
    omega = RPM * 2 * np.pi / 60

    # Power
    P_mechanical = Q * omega
    P_electrical = Vterm * I

    return {
        'RPM': RPM,
        'P_mechanical': P_mechanical,
        'P_electrical': P_electrical,
        'mass_g': motor['mass_g'],
        'V_rate': V_rate
    }


# BATTERY #

def create_battery(Np, V_batt):
    # Ns derived from battery_calcs
    Ns = V_batt / 4
    return {
        'Np': Np,
        'V_batt': V_batt,
        'R_batt': (Ns / Np) * 0.05,
        'C_cell_Ah': 2.2,   # per cell capacity
        'm_cell_kg': 0.045  # per cell mass
    }


def battery_calcs(battery):

    Np = battery['Np']
    V_batt = battery['V_batt']

    # Map voltage to number of series cells (Ns)
    if V_batt == 24:
        Ns = 6
    elif V_batt == 16:
        Ns = 4
    elif V_batt == 12:
        Ns = 3
    else:
        raise ValueError(f"Unsupported battery voltage: {V_batt}")

    # Total capacity (parallel adds)
    total_capacity_Ah = Np * battery['C_cell_Ah']

    # Total mass (all cells count)
    total_mass_kg = Ns * Np * battery['m_cell_kg']

    return total_mass_kg, total_capacity_Ah

# STRUCTURES #
def TopoStudy(PT, MM, BM):

    # PT = Thrust per motor [N]
    # MM = Motor Mass [kg]
    # BM = Battery Mass [kg]
    
    g = 9.81
    AUX_Mass = 0.200
    Hardware_Mass = 0.010

    Central_Force = (AUX_Mass + BM + Hardware_Mass) * g
    Quarter_Force = PT - (MM * g)

    
    mechanical = launch_mechanical(batch=False, cleanup_on_exit=False, transport_mode="insecure")

    script = rf"""
ExtAPI.DataModel.Project.Open(r'C:\\Users\\matt6\\Downloads\\DroneModelV3_files\\dp0\\global\\MECH\\SYS.mechdb')
def get_force(name):
    forces = ExtAPI.DataModel.GetObjectsByType(DataModelObjectCategory.Force)
    for f in forces:
        if f.Name == name:
            return f
    raise Exception("Force not found: " + name)

q1 = get_force("Q1")
q2 = get_force("Q2")
q3 = get_force("Q3")
q4 = get_force("Q4")
m1 = get_force("M1")

# Set values
q1.ZComponent.Output.SetDiscreteValue(0, Quantity("{Quarter_Force} [N]"))
q2.ZComponent.Output.SetDiscreteValue(0, Quantity("{Quarter_Force} [N]"))
q3.ZComponent.Output.SetDiscreteValue(0, Quantity("{Quarter_Force} [N]"))
q4.ZComponent.Output.SetDiscreteValue(0, Quantity("{Quarter_Force} [N]"))

m1.ZComponent.Output.SetDiscreteValue(0, Quantity("-{Central_Force} [N]"))

analyses = ExtAPI.DataModel.GetObjectsByType(DataModelObjectCategory.Analysis)

if len(analyses) == 0:
    raise Exception("No analysis found in model")

analysis = analyses[0]

analysis.ClearGeneratedData()
analysis.Solve(True)


# --- Get Structural Optimization analysis ---
opt_analyses = ExtAPI.DataModel.GetObjectsByName("Structural Optimization")

if not opt_analyses:
    raise Exception("Structural Optimization analysis not found")

opt_analysis = opt_analyses[0]
opt_analysis.Solve(True)

# --- Get its solution ---
opt_solution = opt_analysis.Solution
opt_solution.EvaluateAllResults()

# --- Activate Topology Density result ---
topo_density = None

for res in opt_solution.Children:
    if "Topology Density" in res.Name:
        topo_density = res
        break

if topo_density is None:
    raise Exception("Topology Density result not found")

# Show it in GUI
topo_density.Activate()

# --- Extract Final Mass ---
try:
    final_mass = topo_density.FinalMass
except:
    try:
        final_mass = topo_density.InternalObject.FinalMass
    except:
        raise Exception("Could not extract Final Mass")

# Return to Python
final_mass


"""

    mass = mechanical.run_python_script(script)
    print("Final Mass:", mass)
    return mass

# Function to update RPM in .ini file
def write_rotor_config(filename, rpm):
    # Open the config file in read mode
    with open(filename, "r") as f:
        lines = f.readlines()
# Loop through each line to find the rpm entry
    for i, line in enumerate(lines):
        if line.strip().startswith("rpm"):
            lines[i] = f"rpm = {rpm}\n"
# Write the updated lines back to the file
    with open(filename, "w") as f:
        f.writelines(lines)

# Function to run the BEMT solver
def run_prop_analysis(config_file):
    # Create solver object using the .ini file
    solver = Solver(config_file)
    # Run the solver
    # Returns:
    # T = thrust (N)
    # Q = torque (Nm)
    # P = power (W)
    # sections = blade element breakdown (not used here)
    T, Q, P, sections = solver.run()
    return T, Q, P


# To help check thrust to weight constraint
def calc_thrust_to_weight(thrust_per_motor, mass_of_drone):
    '''thrust_per_motor: Newtons
    mass_of_drone: kilograms'''

    return thrust_per_motor * 4 / (mass_of_drone * 9.81)

############ INPUT #############

# Airfoil list to be used
# Just the options we can choose from in the x vector
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

x = [
    2300,        # kv
    1,           # Np
    0.254,       # diameter
    0.019,       # chord_root
    0.010,       # chord_tip
    76,          # pitch_root
    9.5,         # pitch_tip
    1            # airfoil_index
]
'''
# Optimization Input Vector
x = [
    kv,               # motor kV
    Np,               # number of batteries in parallel
    prop_diameter,
    chord_root,
    chord_tip,
    pitch_root,
    pitch_tip,
    airfoil_index   # integer to index into airfoil_list
]
'''
################## FINAL SIMULATION ####################
# Should I include max iterations?
def full_simulation(x, ini_file="propeller.ini"):

    x = np.array(x, dtype=float)  # ensure consistent type

    kv = int(round(x[0]))
    Np = int(round(x[1]))
    diameter = x[2]
    chord_root = x[3]
    chord_tip = x[4]
    pitch_root = x[5]
    pitch_tip = x[6]
    airfoil_idx = int(round(x[7]))
    print(airfoil_idx)
    
    if airfoil_idx < 0 or airfoil_idx >= len(airfoil_list):
        raise ValueError(f"Invalid airfoil index: {airfoil_idx}")

    airfoil = airfoil_list[airfoil_idx]
    # Fixed
    radius_hub = 0.01016
    
    # Motor/Battery Setup
    motor = get_motor_from_kv(kv)
    motor_mass_kg = motor['mass_g'] / 1000
    V_batt = motor['V_rate']

    battery = create_battery(Np, V_batt)
    battery_mass, battery_capacity = battery_calcs(battery)

    # Propeller and .ini file generation
    rpm_guess = 10000  # initial guess will be overwriten by brentq solver now
    prop = generate_propeller(
        diameter=diameter,
        radius_hub=radius_hub,
        chord_root=chord_root,
        chord_tip=chord_tip,
        pitch_root=pitch_root,
        pitch_tip=pitch_tip,
        rpm=rpm_guess, # placeholder
        airfoil=airfoil
    )
    write_ini_file(prop, ini_file)

    # Currently the thrust to weight requirement is handled by raising a warning
    # For optimization purposes it will probably be more convenient to calculate it outside
    # of the propellor submodule so that it can be checked easily and we can apply a penalty or fancier stuff.
    def rpm_residual(rpm):

        if rpm <= 0:
            return -1e6  # Try and keep solver away from invalid regions

        write_rotor_config(ini_file, rpm)

        try:
            T, Q, P = run_prop_analysis(ini_file)
            motor_out = motor_model(kv, Q, battery)
            rpm_motor = motor_out['RPM']
        except:
            #raise ValueError("Invalid physics")
            return -1e6  # If there are invalid physics: reject

        return rpm_motor - rpm

    # Solve for consistent RPM
    print(rpm_residual(1000))
    print(rpm_residual(60000))

    rpm_solution = brentq(rpm_residual, 1000, 60000, xtol=50)

    # Final evaluation at RPM solution found
    write_rotor_config(ini_file, rpm_solution)
    T, Q, P = run_prop_analysis(ini_file)

    structure_mass = 0.25  #TopoStudy(motor_mass_kg, battery_mass, T)
    total_mass_kg = structure_mass + 4*motor_mass_kg + battery_mass

    # Flight Time (minutes)
    print("Power per motor:", P)
    print("Vbatt:", V_batt)
    print("Battery Capacity:", battery_capacity)
    energy_Wh = battery_capacity * V_batt
    print("Energy Wh:", energy_Wh)
    total_power = 4 * P
    flight_time_hr = energy_Wh / total_power if total_power > 0 else 0
    #flight_time_hr = energy_Wh / (P / 1000) if P > 0 else 0
    print("Flight Time hrs:", flight_time_hr)
    print("Thrust per motor N:", T)
    print("Total mass kg:", total_mass_kg)

    thrust_to_weight = calc_thrust_to_weight(4*T, total_mass_kg)
    print("Thrust to weight ratio:", thrust_to_weight)
    #flight_time_min = flight_time_hr * 60

    return total_mass_kg, flight_time_hr, thrust_to_weight

if __name__ == "__main__":
    print(full_simulation(x, ini_file="propeller.ini"))

# Checking rpm thrust values for analysis
#for rpm in [5000, 10000, 20000, 30000, 40000]:
    #write_rotor_config("propeller.ini", rpm)
    #T, Q, P = run_prop_analysis("propeller.ini")
    #print(f"RPM: {rpm}, Thrust: {T}")

def full_simulationWT(x, ini_file="propeller.ini"):

    x = np.array(x, dtype=float)  # ensure consistent type

    kv = int(round(x[0]))
    Np = int(round(x[1]))
    diameter = x[2]
    chord_root = x[3]
    chord_tip = x[4]
    pitch_root = x[5]
    pitch_tip = x[6]
    airfoil_idx = int(round(x[7]))
    print(airfoil_idx)
    
    if airfoil_idx < 0 or airfoil_idx >= len(airfoil_list):
        raise ValueError(f"Invalid airfoil index: {airfoil_idx}")

    airfoil = airfoil_list[airfoil_idx]
    # Fixed
    radius_hub = 0.01016
    
    # Motor/Battery Setup
    motor = get_motor_from_kv(kv)
    motor_mass_kg = motor['mass_g'] / 1000
    V_batt = motor['V_rate']

    battery = create_battery(Np, V_batt)
    battery_mass, battery_capacity = battery_calcs(battery)

    # Propeller and .ini file generation
    rpm_guess = 10000  # initial guess will be overwriten by brentq solver now
    prop = generate_propeller(
        diameter=diameter,
        radius_hub=radius_hub,
        chord_root=chord_root,
        chord_tip=chord_tip,
        pitch_root=pitch_root,
        pitch_tip=pitch_tip,
        rpm=rpm_guess, # placeholder
        airfoil=airfoil
    )
    write_ini_file(prop, ini_file)

    # Currently the thrust to weight requirement is handled by raising a warning
    # For optimization purposes it will probably be more convenient to calculate it outside
    # of the propellor submodule so that it can be checked easily and we can apply a penalty or fancier stuff.
    def rpm_residual(rpm):

        if rpm <= 0:
            return -1e6  # Try and keep solver away from invalid regions

        write_rotor_config(ini_file, rpm)

        try:
            T, Q, P = run_prop_analysis(ini_file)
            motor_out = motor_model(kv, Q, battery)
            rpm_motor = motor_out['RPM']
        except:
            #raise ValueError("Invalid physics")
            return -1e6  # If there are invalid physics: reject

        return rpm_motor - rpm

    # Solve for consistent RPM
    print(rpm_residual(1000))
    print(rpm_residual(60000))

    rpm_solution = brentq(rpm_residual, 1000, 60000, xtol=50)

    # Final evaluation at RPM solution found
    write_rotor_config(ini_file, rpm_solution)
    T, Q, P = run_prop_analysis(ini_file)

    structure_mass =TopoStudy(motor_mass_kg, battery_mass, T)
    total_mass_kg = structure_mass + 4*motor_mass_kg + battery_mass
    aux_mass = 4*motor_mass_kg + battery_mass
    # Flight Time (minutes)
    print("Power per motor:", P)
    print("Vbatt:", V_batt)
    print("Battery Capacity:", battery_capacity)
    energy_Wh = battery_capacity * V_batt
    print("Energy Wh:", energy_Wh)
    total_power = 4 * P
    flight_time_hr = energy_Wh / total_power if total_power > 0 else 0
    #flight_time_hr = energy_Wh / (P / 1000) if P > 0 else 0
    print("Flight Time hrs:", flight_time_hr)
    print("Thrust per motor N:", T)
    print("Total mass kg:", total_mass_kg)

    thrust_to_weight = calc_thrust_to_weight(4*T, total_mass_kg)
    print("Thrust to weight ratio:", thrust_to_weight)
    #flight_time_min = flight_time_hr * 60

    return total_mass_kg, aux_mass, flight_time_hr, thrust_to_weight
