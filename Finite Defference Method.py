import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# INPUT DATA
# ============================================================

L = 16.0                  # [m] pile length
D = 2.5                   # [m] diameter
Ep = 35.5e9               # [Pa] Young modulus pile
Epy = 68.07e6             # [Pa] subgrade modulus (constant)

V_head = 6.5e3            # [N] lateral load
M_head = 151.45e3         # [Nm] moment at head


# ============================================================
# FINITE DIFFERENCE SOLVER
# ============================================================

def solve_FDM(L, D, Ep, Epy, V_head, M_head, n):

    I = np.pi * D**4 / 64
    EI = Ep * I
    dx = L / n
    alpha = Epy / EI

    N = n + 4
    K = np.zeros((N, N))
    p = np.zeros(N)

    # ---------------- Interior nodes ----------------
    for m in range(2, n+2):
        K[m, m-2] = 1
        K[m, m-1] = -4
        K[m, m]   = 6 + alpha * dx**4
        K[m, m+1] = -4
        K[m, m+2] = 1

    # ---------------- Bottom: fully fixed ----------------
    # y(L) = 0
    K[0, n+1] = 1

    # dy/dx(L) = 0
    K[1, n] = -1
    K[1, n+2] = 1

    # ---------------- Head: free with loads ----------------
    # y''(0) = M/EI
    K[-2, 0] = 1
    K[-2, 1] = -2
    K[-2, 2] = 1
    p[-2] = M_head * dx**2 / EI

    # y'''(0) = V/EI
    K[-1, -4] = -1
    K[-1, -3] = 2
    K[-1, -2] = -2
    K[-1, -1] = 1
    p[-1] = V_head * dx**3 / EI

    # Solve system
    u = np.linalg.solve(K, p)

    # Extract physical nodes
    y = u[1:n+1]
    x = np.linspace(0, L, len(y))

    # ---------------- Post-processing ----------------
    theta = np.full_like(y, np.nan)
    M = np.full_like(y, np.nan)
    V = np.full_like(y, np.nan)

    theta[1:-1] = (y[2:] - y[:-2])/(2*dx)
    M[1:-1] = EI*(y[:-2] - 2*y[1:-1] + y[2:])/(dx**2)
    V[2:-2] = EI*(-y[:-4] + 2*y[1:-3] - 2*y[3:-1] + y[4:])/(2*dx**3)

    return x, y, theta, M, V


# ============================================================
# RUN MAIN SOLUTION
# ============================================================

n_main = 80
x, y, theta, M, V = solve_FDM(L, D, Ep, Epy, V_head, M_head, n_main)


# ============================================================
# PLOT RESPONSE
# ============================================================

fig, axs = plt.subplots(1, 4, figsize=(14, 6))

# Displacement
axs[0].plot(y*1000, x)
axs[0].set_title("Displacement [mm]")
axs[0].set_xlabel("mm")
axs[0].invert_yaxis()

# Rotation
axs[1].plot(theta*1e3, x)
axs[1].set_title("Rotation [-] x10³")
axs[1].invert_yaxis()

# Moment (kNm)
axs[2].plot(M/1e3, x)
axs[2].set_title("Moment [kNm]")
axs[2].invert_yaxis()

# Shear (kN)
axs[3].plot(V/1e3, x)
axs[3].set_title("Shear [kN]")
axs[3].invert_yaxis()

plt.tight_layout()
plt.show()


# ============================================================
# CONVERGENCE TEST
# ============================================================

plt.figure(figsize=(6,6))

for n in [20, 40, 80, 120, 160]:
    x_c, y_c, _, _, _ = solve_FDM(L, D, Ep, Epy, V_head, M_head, n)
    plt.plot(y_c*1000, x_c, label=f"n={n}")

plt.xlabel("Displacement [mm]")
plt.ylabel("Depth [m]")
plt.title("Convergence Test")
plt.gca().invert_yaxis()
plt.legend()
plt.show()


# ============================================================
# PRINT MAX VALUES
# ============================================================

print("\nMaximum values from elastic analysis:")
print("---------------------------------------")
print(f"Max displacement: {np.nanmax(np.abs(y))*1000:.2f} mm")
print(f"Max moment: {np.nanmax(np.abs(M))/1e3:.2f} kNm")
print(f"Max shear: {np.nanmax(np.abs(V))/1e3:.2f} kN")