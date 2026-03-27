import numpy as np
from pybemt.solver import Solver
from scipy.optimize import brentq

# Generate Propeller, which will be used to create .ini file 
def generate_propeller(
    diameter,
    radius_hub,
    chord_root,
    chord_tip,
    pitch_root,
    pitch_tip,
    rpm,
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
        n_elements = sclaed based on previous data set based on blade diameter (40 elements/0.254in)

    """

    # Fixed values
    clustering = 1.2
    epsilon = 1e-3

    # Radius generation
    R = diameter / 2
    #scale the number elements based on previous data set
    n_elements = round((40/0.254) * diameter)
    xi = np.linspace(epsilon, 1, n_elements)
    radius = R * (xi ** clustering)

    # Chord and Pitch generation
    r_root = radius[0]
    r_tip = R

    t = (radius - r_root) / (r_tip - r_root)
    
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

#Example Calling

#prop = generate_propeller(
    #diameter=0.254,
    #radius_hub=0.01016,
    #nblades=2,
    #chord_root=0.019,
    #chord_tip=0.010,
    #pitch_root=76,
    #pitch_tip=9.5,
    #rpm=21000,
    #airfoil="NACA_4412"
#)


# Example Calling
#radius = prop["radius"]
#chord = prop["chord"]
#pitch = prop["pitch"]
#sections = prop["sections"]

#fluid = prop["fluid"]
#solver = prop["solver"]
#case = prop["case"]

# rotor data
#print("nblades =", prop["nblades"])
#print("diameter =", prop["diameter"])
#print("radius_hub =", prop["radius_hub"])

# Print Check Everything Else
#print("\n[rotor]")
#print("radius =", " ".join(map(str, radius)))
#print("chord  =", " ".join(map(str, chord)))
#print("pitch  =", " ".join(map(str, pitch)))
#print("section =", " ".join(sections))

#print("\n[fluid]")
#print("rho =", fluid["rho"])
#print("mu  =", fluid["mu"])

#print("\n[solver]")
#print("solver =", solver["solver"])

#print("\n[case]")
#print("v_inf =", case["v_inf"])
#print("rpm   =", case["rpm"])



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

# Example .ini file creation check
#prop = generate_propeller(
    #diameter=0.254,
    # Keep radius const.
    #radius_hub=0.01016,
    # Keep nblades const.
    ###nblades=2,
    #chord_root=0.019,
    #chord_tip=0.010,
    #pitch_root=76,
    #pitch_tip=9.5,
    # This will stay our initial rpm guess
    #rpm=21000,
    #airfoil="NACA_4412"
#)

#write_ini_file(prop, "propeller.ini")



# # # # # # # # # # # # # #
#what are we doing for the battery?
#def create_battery(Ns, Np):
    # Ns: number of cells in series
    # Np: number of cells in parallel
    #battery = {
        #'Ns' : Ns,
        #'Np' : Np,
        #'V_cell' : 3.7, # V, nominal voltage of each cell
        #'C_cell' : 2.5, # Ah, capacity of each cell
        #'m_cell' : 0.045 # kg, mass of each cell
    #}
    #return battery



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

def battery_terminal_voltage(I, V_rate):
    # Voc battery assumed to be the motor voltage rate
    Voc = V_rate
    # Assumed constant
    Rbatt = 0.05
    return Voc - I * Rbatt

def motor_model(kv, Q):

    motor = get_motor_from_kv(kv)

    kt = motor['kt_NmpA']
    kv = motor['kv_rpm_per_V']
    R_winding = motor['R_winding_ohm']
    V_rate = motor['V_rate']
    I_max = motor['I_max']

    # Current from torque (used internally)
    I = Q / kt

    # Current limit check
    if I > I_max:
        raise ValueError(f"Overcurrent: I={I:.2f}A exceeds motor limit I_max={I_max:.2f}A")
    
    # Battery terminal voltage
    Vterm = battery_terminal_voltage(I, V_rate)

    # Voltage constraint check
    V_motor = Vterm - I * R_winding

    if V_motor <= 0:
        raise ValueError(f"Voltage collapse: V_motor={V_motor:.2f}V (I too large, torque too high)")

    # Electrical RPM
    # Back EMF Equation Used:
    # Assuming Voc = Vrate
    # RPM = Kv (V_rate - (Q/Kt)Rbattery - (Q/Kt)Rwinding)
    RPM = kv * (Vterm - I * R_winding)

    # RPM CHECKS!
    if RPM <= 0:
        raise ValueError(f"Invalid RPM computed: {RPM:.2f}")
    if np.isnan(RPM) or np.isinf(RPM):
        raise ValueError("RPM became NaN or Inf")
    
    # Angular velocity
    omega = RPM * 2 * np.pi / 60

    # Mechanical power
    P = Q * omega

    return {
        'RPM': RPM,
        'P_motor': P,
        'mass_g': motor['mass_g'],
        'V_rate': V_rate
    }


# BATTERY #

def create_battery(Np, V_batt):
    return {
        'Np': Np,
        'V_batt': V_batt,
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





 
#def create_structure():
    #return 1

#def create_propellor(airfoil, chord, pitch, prop_diam):
    #prop = {
        #'airfoil' : airfoil,
        #'chord' : chord,
        #'pitch' : pitch,
        #'diameter' : prop_diam
    #}
    #return prop

#def create_design_vector(batt_series, batt_parallel, motor, prop_diam, airfoil, chord, pitch):
    #design_vector = {
        #'battery' : create_battery(batt_series, batt_parallel),
        #'motor' : motor,
        #'propellor' : create_propellor(airfoil, chord, pitch, prop_diam)
    #}
    #return design_vector

#def battery_calcs(battery):
    #battery['Voltage_V'] = battery['Ns'] * 3.7
    #battery['Capacity_mAh'] = battery['Np'] * 2200
    #battery['Mass_kg'] = battery['Np'] * battery['Ns'] * 0.045
    #battery['Length_mm'] = 70
    #battery['Width_mm'] = 43
    #battery['Height_mm'] = 6*battery['Ns']*battery['Np']
    #return battery






# STRUCTURES #

def matt_structures(motor_mass, battery_mass, propeller_thrust):
    chassis_mass = 1
    return chassis_mass



############NEED TO WRITE THIS###########################################################
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


# Main propeller function
# Testing out generated confi_file, the previous one was "apc10x5.ini"
def propellor_calcs(config_file="propeller.ini"):
    """
    Reads rpm from the .ini file, runs the BEMT solver,
    and returns thrust, torque, and power.
    """

    # Read current rpm from config file
    rpm = None
    with open(config_file, "r") as f:
        for line in f:
            # Look for the line that defines rpm
            if line.strip().startswith("rpm"):
                # Extract the numeric value after '='
                rpm = float(line.split("=")[1])
                break
# If rpm is not found, raise an error
    if rpm is None:
        raise ValueError("RPM not found in config file.")

    # Run solver (geometry and rpm guess already defined in .ini)
    # T = thrust (Newtons)
    # Q = torque (Nm)
    # P = power (Watts)
    T, Q, P = run_prop_analysis(config_file)

    return T, Q, P, rpm
#########################################################################################

def propellor_calcs2(
    config_file="propeller.ini",
    motor_mass_kg=0.0,
    battery_mass_kg=0.0,
    structure_mass_kg=0.25,
    TW_Constraint=1.5,
    rpm_min=1000,
    rpm_max=60000,
    tol=50
):
    """
    Computes RPM required to satisfy thrust-to-weight constraint,
    then runs BEMT at that RPM.

    Returns:
        rpm_solution, T, Q, P
    """

    g = 9.81

    # Total mass
    total_mass_kg = structure_mass_kg + 4 * motor_mass_kg + battery_mass_kg

    # Required total thrust
    T_required_total = TW_Constraint * total_mass_kg * g

    # Per motor (quadrotor)
    T_target = T_required_total / 4

    # Thrust-Based Residual Function
    def thrust_residual(rpm):
        # Prevent invalid RPM
        if rpm <= 0:
            return -1e6

        write_rotor_config(config_file, rpm)

        try:
            T, Q, P = run_prop_analysis(config_file)
        except Exception:
            return -1e6

        # Root = when thrust matches requirement
        return T - T_target

    # Check Bounds prior to solving
    f_low = thrust_residual(rpm_min)
    f_high = thrust_residual(rpm_max)

    if f_low > 0:
        raise ValueError(
            "Even at minimum RPM, thrust exceeds required → overspecced prop or design too powerful"
        )

    if f_high < 0:
        raise ValueError(
            "Even at maximum RPM, cannot reach required thrust → infeasible design"
        )

    # RPM Solver
    rpm_solution = brentq(
        thrust_residual,
        rpm_min,
        rpm_max,
        xtol=tol
    )

    # Final eval at soln.
    write_rotor_config(config_file, rpm_solution)
    T, Q, P = run_prop_analysis(config_file)

    # Sanity check lol
    if abs(T - T_target) > 0.05 * T_target:
        print("Warning: thrust not tightly converged")

    return rpm_solution, T, Q, P

'''
# New Function
def propellor_calcs2(
    config_file="propeller.ini",
    motor_mass_kg=0.0,
    battery_mass_kg=0.0,
    structure_mass_kg=0.25,
    TW_Constraint=1.5,
    rpm_min=1000,
    rpm_max=60000,
    tol=100):

    """
    Computes minimum RPM required for hover (with Thrust/Weight FOS),
    then runs BEMT at that RPM. Usese the brentq scipy optimizer to find this feasible RPM.

    Returns:
        rpm, T, Q, P
    """

    # Mass
    total_mass_kg = structure_mass_kg + 4 * motor_mass_kg + battery_mass_kg

    # Acceleration
    g = 9.81 #m/s^2

    # Required Thrust per motor
    T_required_total = TW_Constraint * total_mass_kg * g
    # Fourth Drone
    T_target = T_required_total / 4

    # Residual Function
    def thrust_residual(rpm):
        write_rotor_config(config_file, rpm)
        T, Q, P = run_prop_analysis(config_file)
        return T - T_target
    
    f_low = thrust_residual(rpm_min)
    f_high = thrust_residual(rpm_max)

    if f_low > 0:
        raise ValueError("Even at low RPM, thrust exceeds target → bad design")

    if f_high < 0:
        raise ValueError("Even at max RPM, cannot reach required thrust → infeasible design")

    # Solve for RPM
    rpm_solution = brentq(thrust_residual, rpm_min, rpm_max, xtol=10)

    # RPM Bisection
    #rpm_low = rpm_min
    #rpm_high = rpm_max

    #rpm_solution = None
    #T_solution = None

    #for _ in range(50):

        #rpm_mid = 0.5 * (rpm_low + rpm_high)

        # Update config file
        #write_rotor_config(config_file, rpm_mid)

        # Run BEMT
        #T, Q, P = run_prop_analysis(config_file)

        # Bisection update
        #if T < T_target:
            #rpm_low = rpm_mid
        #else:
            #rpm_high = rpm_mid

        # Convergence check
        #if abs(T - T_target) < tol:
            #rpm_solution = rpm_mid
            #T_solution = T
            #break

        #rpm_solution = rpm_mid
        #T_solution = T

    # Final eval at converged RPM
    write_rotor_config(config_file, rpm_solution)
    T, Q, P = run_prop_analysis(config_file)

    return rpm_solution, T, Q, P
'''


############ INPUT #############


# Airfoil list to be used
airfoil_list = [
    "xfoild_cl_cd_Re5000",
    "NACA_4412",
    "NACA_2412",
    "NACA_0012"
]
'''
# Optimization Input Vector
x = [
    kv,               # motor kV
    Np,               # number of batteries in parallel
    prop_diameter,
    chord_root,
    chord_tip,
    n_blades,
    pitch_root,
    pitch_tip,
    airfoil_index   # integer to index into airfoil_list
]
'''
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


################## FINAL SIMULATION ####################
# Should I include max iterations?
def full_simulation(x, ini_file="propeller.ini"):
    """
    Full aircraft parameter sizing adn propulsion convergence simulation.

    Returns:
        total_mass, flight_time
    """

    # Unpack the input vector
    kv, Np, diameter, chord_root, chord_tip, pitch_root, pitch_tip, airfoil_idx = x

    # Index select airfoil type from the list above
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
        rpm=rpm_guess,
        airfoil=airfoil
    )
    write_ini_file(prop, ini_file)

    # NEED TO ACCOUNT FOR THRUST REQUIREMENT!
    def rpm_residual(rpm):

        if rpm <= 0:
            return -1e6  # Try and keep solver away from invalid regions

        write_rotor_config(ini_file, rpm)

        try:
            T, Q, P = run_prop_analysis(ini_file)
            motor_out = motor_model(kv, Q)
            rpm_motor = motor_out['RPM']
        except:
            return -1e6  # If there are invalid physics: reject

        return rpm_motor - rpm

    
    # NEW (haven't implemented it yet)(Just going to give you the working one for now) RPM Residual Accounting for cosntraitn and RPM convergence 
    #def rpm_residual(rpm):

        #if rpm <= 0:
            #return -1e6

        #write_rotor_config(ini_file, rpm)

        #try:
            #T, Q, P = run_prop_analysis(ini_file)
            #motor_out = motor_model(kv, Q)
            #rpm_motor = motor_out['RPM']
        #except:
            #return -1e6

        # Implement thrust cosntraint check
        #if T < T_target:
            #return -1e6  # Reject RPM that cannot meet thrust

        # Motor Consistency to run
        #return rpm_motor - rpm

    # Solve for consistent RPM
    rpm_solution = brentq(rpm_residual, 1000, 60000, xtol=50)

    # Final evaluation at RPM solution found
    write_rotor_config(ini_file, rpm_solution)
    T, Q, P = run_prop_analysis(ini_file)

    # Initial RPM guess w/ propellor_calcs2 (used in the converged loop below that is now subbed by brentq)
    #rpm_old, _, _, _ = propellor_calcs2(
        #config_file=ini_file,
        #motor_mass_kg=motor_mass_kg,
        #battery_mass_kg=battery_mass,
        #structure_mass_kg=0.25
    #)

    # Loop until converged (using brentq now)
    #tol = 100
    #max_iter = 100
    #error = 1000
    #iter_count = 0

    #while error > tol and iter_count < max_iter:
        #iter_count += 1

        #if rpm_old <= 0 or np.isnan(rpm_old) or np.isinf(rpm_old):
            #raise ValueError(f"Invalid RPM passed to BEMT: {rpm_old}")

        #write_rotor_config(ini_file, rpm_old)
        #T, Q, P = run_prop_analysis(ini_file)

        #motor_out = motor_model(kv, Q)
        #rpm_new = motor_out['RPM']

        # Using some relaxation to hopefully prevent the divergence that's occuring
        #alpha = 0.3
        #rpm_new = alpha * rpm_new + (1 - alpha) * rpm_old

        #error = abs(rpm_new - rpm_old)
        #rpm_old = rpm_new


# This was the origigal conbergence/motor loop, which was replaced by the loop above using the relaxation
    #while error > tol and iter_count < max_iter:
        #iter_count += 1

        # Update Propellor RPM
        #write_rotor_config(ini_file, rpm_old)
        #T, Q, P = run_prop_analysis(ini_file)

        # Motor RPM Update
        #I = Q / motor['kt_NmpA']
        #Vterm = battery_terminal_voltage(I, V_batt)
        #rpm_new = motor['kv_rpm_per_V'] * Vterm
        #motor_mass_kg = motor['mass_g'] / 1000

        # Check convergence
        #error = abs(rpm_new - rpm_old)
        #rpm_old = rpm_new

    # Total Mass (chassis, batteries, motors) (kg)
    #print(motor_mass_kg)
    #print(battery_mass)
    #print(T)
    structure_mass = matt_structures(motor_mass_kg, battery_mass, T)
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
    #flight_time_min = flight_time_hr * 60

    return total_mass_kg, flight_time_hr

print(full_simulation(x, ini_file="propeller.ini"))

# Checking rpm thrust values for analysis
#for rpm in [5000, 10000, 20000, 30000, 40000]:
    #write_rotor_config("propeller.ini", rpm)
    #T, Q, P = run_prop_analysis("propeller.ini")
    #print(f"RPM: {rpm}, Thrust: {T}")