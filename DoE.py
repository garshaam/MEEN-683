from Project_simulation import full_simulation
import csv
import itertools
import pandas as pd
import time

# PART 1: write a DoE csv with factorial design based on specified design vars
# write_DoE_csv_input()
def write_DoE_csv_input(filename, motors, batteries, diameters, chords, pitches, airfoils):
    
    headers = ["Airfoil", "Motor", "Diameter", "Battery", "Root Chord", "Tip Chord", "Root Pitch", "Tip Pitch"]

    # Generate full factorial combinations
    combinations = itertools.product(
        airfoils, motors, diameters, batteries, chords, pitches
    )

    # Write to CSV
    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for combo in combinations:
            airfoil, motor, diameter, battery, chord, pitch = combo
            row = [airfoil, motor, diameter, battery, chord[0], chord[1], pitch[0], pitch[1]]
            writer.writerow(row)

    print(f"Input csv saved to {filename}")

#### BEGIN PART 2: RUNNING THE DOE SIMULATION AND STORING RESULTS IN A NEW CSV ####
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

def run_doe(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    total_mass_list = []
    flight_time_list = []
    sim_time_list = []
    thrust_to_weight_list = []

    for i, row in df.iterrows():
        # Build design vector x
        x = [
            row["Motor"],          # kv
            row["Battery"],        # Np
            row["Diameter"],       # prop_diameter
            row["Root Chord"],     # chord_root
            row["Tip Chord"],      # chord_tip
            row["Root Pitch"],     # pitch_root
            row["Tip Pitch"],      # pitch_tip
            map_airfoil_to_number(row["Airfoil"])  # airfoil_index
        ]

        start = time.time()
        # Run simu
        try:
            total_mass_kg, flight_time_hr, thrust_to_weight = full_simulation(x)

        except Exception as e:
            for qqq in range(20): print(f"Error occurred while running simulation for row {i}: {e}")
            total_mass_kg = -1
            flight_time_hr = -1
            thrust_to_weight = -1
        
        sim_time = time.time() - start

        # Store data
        total_mass_list.append(total_mass_kg)
        flight_time_list.append(flight_time_hr)
        sim_time_list.append(sim_time)
        thrust_to_weight_list.append(thrust_to_weight)

        if i % 10 == 0: 
            df["Mass [kg]"] = total_mass_list
            df["Flight Time [hr]"] = flight_time_list
            df["Thrust to Weight"] = thrust_to_weight_list
            df["Sim Time [s]"] = sim_time_list
            cur_name = str(i)+"_"+output_csv
            df.to_csv(cur_name, index=False) # save progress every 10 rows

    
    df["Mass [kg]"] = total_mass_list
    df["Flight Time [hr]"] = flight_time_list
    df["Thrust to Weight"] = thrust_to_weight_list
    df["Sim Time [s]"] = sim_time_list

    df.to_csv(output_csv, index=False)

    return df

# Define the design variable levels
def full_factorial_doe():
    motors = [2300, 3600, 3800]
    batteries = [1, 2]
    diameters = [3*0.0254, 4*0.0254]  # in inches
    chords = [(8/1000, 4/1000), (18/1000, 9/1000)]  # (root, tip) in mm
    pitches = [(10, 4), (6, 1)]  # (root, tip) in degrees
    airfoils = ["MH30_7_DOE", "NACA2412_DOE", "RG-15_8_DOE", "SD7037_DOE", "SD7062_DOE"]

    # execute above function to write a DoE csv with factorial design based on specified design vars
    filename = "full_factorial_DoE_input.csv"
    write_DoE_csv_input(filename, motors, batteries, diameters, chords, pitches, airfoils)

    run_doe("full_factorial_DoE_input.csv", "full_factorial_doe_results.csv")
    print('Done')

def initial_test_airfoils():
    motors_test = [2300]
    batteries_test = [1]
    diameters_test = [3 * 0.0254]
    chords_test = [(18/1000, 4/1000)]
    pitches_test = [(15, 0)]
    airfoils_test = ["MH30_7_DOE", "NACA2412_DOE", "RG-15_8_DOE", "SD7037_DOE", "SD7062_DOE"]

    filename = "airfoil_test_input.csv"

    write_DoE_csv_input(
        filename,
        motors_test,
        batteries_test,
        diameters_test,
        chords_test,
        pitches_test,
        airfoils_test
    )

    run_doe(filename, "airfoil_test_results.csv")
    print('Done!')

# initial_test_airfoils() # use this to test functionality before running entire test
full_factorial_doe() # use this to execute full factorial test as shown above
