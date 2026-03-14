import numpy as np

def create_battery(Ns, Np):
    # Ns: number of cells in series
    # Np: number of cells in parallel
    battery = {
        'Ns' : Ns,
        'Np' : Np,
        'V_cell' : 3.7, # V, nominal voltage of each cell
        'C_cell' : 2.5, # Ah, capacity of each cell
        'm_cell' : 0.045 # kg, mass of each cell
    }
    return battery


def create_motor(kv, I0, R, V):
    # kv: RPM/V
    # I0: A, idle current
    # R: Ohm, widning resistance
    # V_rate: V, rated voltage (cant exceed)
    motor = {
        'kt_NmpA' : 60/(2*np.pi*kv), # Nm/A
        'ke_Vsprad' : 60/(2*np.pi*kv),
        'I0_A' : I0,
        'R_ohm' : R,
        'V_rate' : V
    }
    return motor

def create_structure():
    return 1

def create_propellor(airfoil, chord, pitch, prop_diam):
    prop = {
        'airfoil' : airfoil,
        'chord' : chord,
        'pitch' : pitch,
        'diameter' : prop_diam
    }
    return prop


motor_2300kv = create_motor(2300, 0.53, 1.23)
motor_3600kv = create_motor(3600, 0.66, 0.7)
motor_3800kv = create_motor(3800, 0.75, 0.21)
motors = [motor_2300kv, motor_3600kv, motor_3800kv]

def create_design_vector(batt_series, batt_parallel, motor, prop_diam, airfoil, chord, pitch):
    design_vector = {
        'battery' : create_battery(batt_series, batt_parallel),
        'motor' : motor,
        'propellor' : create_propellor(airfoil, chord, pitch, prop_diam)
    }
    return design_vector

def battery_calcs(battery):
    battery['Voltage_V'] = battery['Ns'] * 3.7
    battery['Capacity_mAh'] = battery['Np'] * 2200
    battery['Mass_kg'] = battery['Np'] * battery['Ns'] * 0.045
    battery['Length_mm'] = 70
    battery['Width_mm'] = 43
    battery['Height_mm'] = 6*battery['Ns']*battery['Np']
    return battery

def matt_structures(INPUTS):
    mass = 1
    return mass

def propellor_calcs():
    return 1

