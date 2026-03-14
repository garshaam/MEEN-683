import numpy as np
from pybemt.solver import Solver

def write_rotor_config(filename, r, chord, twist, rpm, B, airfoil):
    # this only modifies the rpm line of the .ini file

    with open(filename, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.strip().startswith("rpm"):
            lines[i] = f"rpm = {rpm}\n"

    with open(filename, "w") as f:
        f.writelines(lines)

    # with open(filename,"a") as f:
    #     # f.write("[case]\n")
    #     # f.write('v_inf = 0.0\n')
    #     # f.write(f"rpm = {rpm}\n")


    #     # f.write("[rotor]\n")
    #     # f.write(f"nblades = {B}\n")
    #     # f.write("diameter = 0.254\n")
    #     # f.write("radius_hub = 0.01016\n")

    #     # f.write("chord = " + ",".join(map(str,chord)) + "\n")
    #     # f.write("pitch = " + ",".join(map(str,twist)) + "\n")
    #     # f.write(f"airfoil = {airfoil}\n")
    #     # f.write("nblades = 2\n")


    #     # f.write("\n[fluid]\n")
    #     # f.write("rho = 1.225\n")
    #     # f.write("mu = 1.81e-5\n")
    #     f.write(f"rpm = {rpm}\n")

def run_prop_analysis(config_file):
    # this function does work

    solver = Solver(config_file)

    T, Q, P, sections = solver.run()

    return T, Q, P


# inches → meters
IN2M = 0.0254

r_in = np.array([
0.0950,0.1900,0.2850,0.3800,0.4750,0.5700,0.6650,0.7600,0.8550,0.9500,
1.0467,1.1728,1.3005,1.4283,1.5560,1.6838,1.8116,1.9393,2.0671,2.1948,
2.3226,2.4503,2.5781,2.7059,2.8336,2.9614,3.0891,3.2169,3.3447,3.4724,
3.6002,3.7279,3.8557,3.9834,4.1112,4.2390,4.3667,4.4945,4.6222,4.7447
])

chord_in = np.array([
0.7665,0.7265,0.6867,0.6544,0.6358,0.6351,0.6527,0.6863,0.7326,0.7898,
0.8498,0.9092,0.9501,0.9819,1.0065,1.0241,1.0352,1.0402,1.0394,1.0332,
1.0220,1.0063,0.9863,0.9625,0.9353,0.9051,0.8722,0.8370,0.8000,0.7615,
0.7219,0.6817,0.6411,0.6005,0.5605,0.5213,0.4833,0.4470,0.4127,0.3820
])

pitch_in = np.array([
2.4719,2.7742,3.0572,3.3210,3.5656,3.7909,3.9970,4.1838,4.3514,4.4998,
4.6311,4.7722,4.8807,4.9544,4.9934,5.0000,5.0000,5.0000,5.0000,5.0000,
5.0000,5.0000,5.0000,5.0000,5.0000,5.0000,5.0000,5.0000,5.0000,5.0000,
5.0000,5.0000,5.0000,5.0000,5.0000,5.0000,5.0000,5.0000,5.0000,5.0000
])

R = 5.0 * IN2M

r = r_in * IN2M
chord = chord_in * IN2M

pitch = pitch_in * IN2M

twist = np.arctan(pitch / (2*np.pi*r))
twist_deg = np.degrees(twist)

###

from pybemt.solver import Solver

RPM = np.arange(1000,21001,1000)

T_pred = []
Q_pred = []
P_pred = []

counter_Ben = 1
for rpm in RPM:

    write_rotor_config(
        "apc10x5.ini",
        r,
        chord,
        twist_deg,
        rpm,
        B=2,
        airfoil="NACA4412"
    )

    T,Q,P = run_prop_analysis("apc10x5.ini")
    print(f'Worked {counter_Ben} / 21')
    counter_Ben += 1

    T_pred.append(T)
    Q_pred.append(Q)
    P_pred.append(P)

T_pred = np.array(T_pred)
Q_pred = np.array(Q_pred)
P_pred = np.array(P_pred)

# exp means experimental. This is for validation

###
exp_T = np.array([0.032,0.127,0.287,0.512,0.802,1.158,1.580,2.070,2.630,3.260,
3.962,4.739,5.592,6.524,7.539,8.641,9.833,11.122,12.512,14.011,15.624])*4.44822

exp_Q = np.array([0.024,0.085,0.181,0.312,0.477,0.677,0.911,1.181,1.487,1.831,
2.213,2.635,3.099,3.608,4.163,4.768,5.427,6.147,6.953,7.916,9.149])*0.113

exp_P = np.array([0.281,2.012,6.436,14.768,28.224,48.032,75.450,111.774,158.349,
216.588,287.979,374.110,476.686,597.560,738.770,902.589,1091.613,1309.064,
1563.034,1873.044,2273.170])

def nrmse(pred,exp):
    return np.sqrt(np.mean((pred-exp)**2))/np.mean(exp)

print("Thrust error:", nrmse(T_pred,exp_T))
print("Torque error:", nrmse(Q_pred,exp_Q))
print("Power error:", nrmse(P_pred,exp_P))

import matplotlib.pyplot as plt

plt.plot(RPM,exp_T,label="exp")
plt.plot(RPM,T_pred,label="pyBEMT")
plt.xlabel("RPM")
plt.ylabel("Thrust (N)")
plt.legend()
plt.show()

plt.plot(RPM,exp_P,label="exp")
plt.plot(RPM,P_pred,label="pyBEMT")
plt.xlabel("RPM")
plt.ylabel("Power (W)")
plt.legend()
plt.show()

plt.plot(RPM,exp_Q,label="exp")
plt.plot(RPM,Q_pred,label="pyBEMT")
plt.xlabel("RPM")
plt.ylabel("Torque (N·m)")
plt.legend()
plt.show()

def percent_error(pred, exp):
    return np.abs(pred - exp) / np.abs(exp) * 100

T_err = percent_error(T_pred, exp_T)
Q_err = percent_error(Q_pred, exp_Q)
P_err = percent_error(P_pred, exp_P)

print("\nTHRUST ERRORS")
for rpm, pred, exp, err in zip(RPM, T_pred, exp_T, T_err):
    print(f"RPM {rpm:5d} | Pred {pred:8.3f} | Exp {exp:8.3f} | %Err {err:6.2f}")

print("\nTORQUE ERRORS")
for rpm, pred, exp, err in zip(RPM, Q_pred, exp_Q, Q_err):
    print(f"RPM {rpm:5d} | Pred {pred:8.4f} | Exp {exp:8.4f} | %Err {err:6.2f}")

print("\nPOWER ERRORS")
for rpm, pred, exp, err in zip(RPM, P_pred, exp_P, P_err):
    print(f"RPM {rpm:5d} | Pred {pred:8.2f} | Exp {exp:8.2f} | %Err {err:6.2f}")