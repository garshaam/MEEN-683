# function that one shots via library to find Cd and Cl
# function that one shots via surrogate surfaces to find Cd and Cl
# function that develops a surrogate surface near a point
# function that converts a full propeller into discrete elements that can be fed into the calculator

import math

import pandas as pd
from pyxfoil import Xfoil, set_workdir, set_xfoilexe

# Load xfoil
set_workdir('.\\')
set_xfoilexe('.\\XFOIL6.99\\xfoil.exe')

# Create an instance of xfoil
xfoil = Xfoil('NACA 2412')
xfoil.points_from_dat('NACA\\NACA2412.dat')
xfoil.set_ppar(30) # I do not think the number of panels has to equal the
# number of points used within the .dat file

class GradientDetails:
    '''Useful class for telling xfoild_cl_cd how to compute gradients'''
    def __init__(self, angle_of_attack_step, log_reynolds_step):
        # Move in angle of attack
        self.angle_of_attack_step: int = angle_of_attack_step 
        # Percentage move in Reynolds number
        self.log_reynolds_step: int = log_reynolds_step

def xfoild_cl_cd_with_gradient(angle_of_attack, reynolds_number, gradient_details: GradientDetails):
    '''Same as xfoild_cl_cd but computes gradients in both directions as well. 
    Requires 5 function evaluations instead of 1.'''

    minAlpha = angle_of_attack - gradient_details.angle_of_attack_step / 2
    maxAlpha = angle_of_attack + gradient_details.angle_of_attack_step / 2

    # Perturbation is applied over log_10(Re) because exponential changes in
    # Reynolds number are more appropriate than linear changes.
    minRe = reynolds_number * (10 ** (-gradient_details.log_reynolds_step / 2))
    maxRe = reynolds_number * (10 ** (gradient_details.log_reynolds_step / 2))

    # Center is the same as the normal xfoild_cl_cd function
    polar_center = xfoil.run_polar(almin=angle_of_attack, almax=angle_of_attack, alint=1.0, Re=reynolds_number)
    c_l = float(polar_center.cl[0])
    c_d = float(polar_center.cd[0])

    polar_alpha_min = xfoil.run_polar(almin=minAlpha, almax=minAlpha, alint=1.0, Re=reynolds_number)
    polar_alpha_max = xfoil.run_polar(almin=maxAlpha, almax=maxAlpha, alint=1.0, Re=reynolds_number)
    cl_alpha_min = float(polar_alpha_min.cl[0])
    cl_alpha_max = float(polar_alpha_max.cl[0])
    cd_alpha_min = float(polar_alpha_min.cd[0])
    cd_alpha_max = float(polar_alpha_max.cd[0])

    polar_re_min = xfoil.run_polar(almin=angle_of_attack, almax=angle_of_attack, alint=1.0, Re=minRe)
    polar_re_max = xfoil.run_polar(almin=angle_of_attack, almax=angle_of_attack, alint=1.0, Re=maxRe)
    cl_re_min = float(polar_re_min.cl[0])
    cl_re_max = float(polar_re_max.cl[0])
    cd_re_min = float(polar_re_min.cd[0])
    cd_re_max = float(polar_re_max.cd[0])

    delta_alpha = maxAlpha - minAlpha
    delta_log_re = math.log10(maxRe) - math.log10(minRe)

    dcl_dalpha: float = (cl_alpha_max - cl_alpha_min) / delta_alpha
    dcd_dalpha: float = (cd_alpha_max - cd_alpha_min) / delta_alpha
    dcl_dlogre: float = (cl_re_max - cl_re_min) / delta_log_re
    dcd_dlogre: float = (cd_re_max - cd_re_min) / delta_log_re

    return c_l, c_d, dcl_dalpha, dcd_dalpha, dcl_dlogre, dcd_dlogre

def xfoild_cl_cd(angle_of_attack, reynolds_number):
    '''Uses the xfoil library to compute the lift and drag coefficients 
    for an incremental element of an airfoil.
    
    Assumes viscous flow (in constrast to broader BEMT theory which neglects 
    viscosity). This is standard practice.
    
    An error will be displayed if xfoil is used past its "effective" range
    (xfoil is not accurate after stall).'''
    # Is any error/warning actually being raised if used past effective range?

    polar = xfoil.run_polar(almin=angle_of_attack, almax=angle_of_attack, alint=1.0, Re=reynolds_number)
    c_l = float(polar.cl[0])
    c_d = float(polar.cd[0])

    return c_l, c_d

def surrogate_cl_cd(angles_of_attack, reynolds_numbers):
    '''Uses the xfoil library and inputted arrays to calculate a grid of Cl
    and Cd values. Then, creates a local quadratic at each point to make
    gradient calculation easy.

    Grid spacing is determined by the inputted arrays.'''


def use_surrogate(angle_of_attack, reynolds_number, calculate_gradients=False):
    '''Uses the surrogate model to compute the lift and drag coefficients
    for an incremental element of an airfoil.
    
    The surrogate is only valid within its domain (no extrapolation).'''


def build_xfoild_cl_cd_dataframe(reynolds_number=50000, alpha_start=-180, alpha_end=180):
    polar = xfoil.run_polar(almin=alpha_start, almax=alpha_end, alint=1.0, Re=reynolds_number)
    df = pd.DataFrame({
        'Alpha': polar.alpha,
        'Cl': polar.cl,
        'Cd': polar.cd,
    })
    return df[['Alpha', 'Cl', 'Cd']]

# Program starts here (CSV FILE)
# This is Where we change, Re & Alpha start/end below
xfoild_cl_cd_df = build_xfoild_cl_cd_dataframe(reynolds_number=50000, alpha_start=-5, alpha_end=10)
print(xfoild_cl_cd_df)
xfoild_cl_cd_df.to_csv('xfoild_cl_cd_Re50000.csv', index=False)

# DAT FILE PROGRAM
def write_aerodyn_dat(df, filename, reynolds_number=50000):
    with open(filename, "w") as f:

        # Header (same structure as E63.dat)
        f.write("Airfoil data for AeroDyn v13\n")
        f.write(f"Generated from XFOIL (Re={reynolds_number})\n")
        # It does not matter if these first header rows are legit
        # pybemt skips them anyway (refer to airfoil.py in pybemt library)
        f.write("1\n") # These values are nonsense but do not matter.
        f.write("0\n")
        f.write("13.5\n")
        f.write("0\n")
        f.write("0\n")
        f.write("0\n")
        f.write("-2.98\n")
        f.write("8.97761\n")
        f.write("1.0039\n")
        f.write("-0.3607\n")
        f.write("-6.0\n")
        f.write("0.0138\n")

        # Data table
        for _, row in df.iterrows():
            alpha = row["Alpha"]
            cl = row["Cl"]
            cd = row["Cd"]

            f.write(f"{alpha:10.2f}{cl:12.4f}{cd:12.4f}\n")

write_aerodyn_dat(xfoild_cl_cd_df, "xfoild_cl_cd_Re5000.dat", 50000)