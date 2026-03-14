"""Generate and plot a NACA 4-digit airfoil profile."""

# AI SLOP
# This is the only file of AI slop
# Can get the NACA .dat files from a trusted source if this does not work.

import math

import matplotlib.pyplot as plt
import numpy as np


def parse_naca_code(code: str) -> tuple[float, float, float]:
    """Return (m, p, t) parameters for a 4-digit NACA code."""
    code = code.strip()
    if len(code) != 4 or not code.isdigit():
        raise ValueError("Code must be exactly 4 numeric digits (example: 2412).")

    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0
    return m, p, t


def generate_naca4(code: str, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate closed airfoil coordinates (upper TE->LE, lower LE->TE)."""
    if n_points < 20:
        raise ValueError("Please use at least 20 points for a smooth profile.")

    m, p, t = parse_naca_code(code)

    # Cosine spacing gives better point density near the leading edge.
    beta = np.linspace(0.0, math.pi, n_points)
    x = 0.5 * (1.0 - np.cos(beta))

    yt = 5.0 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    if m > 0.0 and p > 0.0:
        left = x < p
        right = ~left

        yc[left] = m / (p**2) * (2.0 * p * x[left] - x[left] ** 2)
        dyc_dx[left] = 2.0 * m / (p**2) * (p - x[left])

        yc[right] = m / ((1.0 - p) ** 2) * (
            (1.0 - 2.0 * p) + 2.0 * p * x[right] - x[right] ** 2
        )
        dyc_dx[right] = 2.0 * m / ((1.0 - p) ** 2) * (p - x[right])

    theta = np.arctan(dyc_dx)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    x_profile = np.concatenate([xu[::-1], xl[1:]])
    y_profile = np.concatenate([yu[::-1], yl[1:]])
    return x_profile, y_profile


def write_xfoil_dat(path: str, name: str, x_coords: np.ndarray, y_coords: np.ndarray) -> None:
    """Write an XFOIL-compatible .dat airfoil file."""
    with open(path, "w", encoding="ascii") as file:
        file.write(f"{name}\n")
        for x_val, y_val in zip(x_coords, y_coords):
            file.write(f"{x_val:.6f} {y_val:.6f}\n")


def main() -> None:
    code = input("Enter 4-digit NACA airfoil code (example 2412): ").strip()
    n_points = int(input("Enter number of points along the airfoil (>= 20): ").strip())

    x_profile, y_profile = generate_naca4(code, n_points)
    airfoil_name = f"NACA{code}"
    output_file = f"{airfoil_name}.dat"
    write_xfoil_dat(output_file, airfoil_name, x_profile, y_profile)
    print(f"Saved XFOIL airfoil file: {output_file}")

    plt.figure(figsize=(10, 4))
    plt.plot(x_profile, y_profile, "b-", linewidth=1.5)
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title(f"NACA {code} Airfoil ({n_points} points)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
