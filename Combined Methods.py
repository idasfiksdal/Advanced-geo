import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# INPUT DATA
# ============================================================

L = 16.0
D = 2.5
Ep = 35.5e9
Epy = 68.07e6

V_head = 6.5e3
M_head = 151.45e3

# ============================================================
# COMMON PARAMETERS
# ============================================================

I = np.pi * D**4 / 64
EI = Ep * I
lam = (Epy/(4*EI))**0.25


# ============================================================
# FINITE DIFFERENCE METHOD
# ============================================================

def solve_FDM(n):

    dx = L / n
    alpha = Epy / EI

    N = n + 4
    K = np.zeros((N, N))
    p = np.zeros(N)

    # Interior
    for m in range(2, n+2):
        K[m, m-2] = 1
        K[m, m-1] = -4
        K[m, m]   = 6 + alpha * dx**4
        K[m, m+1] = -4
        K[m, m+2] = 1

    # Bottom fixed
    K[0, n+1] = 1
    K[1, n] = -1
    K[1, n+2] = 1

    # Head free
    K[-2, 0] = 1
    K[-2, 1] = -2
    K[-2, 2] = 1
    p[-2] = M_head * dx**2 / EI

    K[-1, -4] = -1
    K[-1, -3] = 2
    K[-1, -2] = -2
    K[-1, -1] = 1
    p[-1] = V_head * dx**3 / EI

    u = np.linalg.solve(K, p)

    y = u[1:n+1]
    x = np.linspace(0, L, len(y))

    theta = np.full_like(y, np.nan)
    M = np.full_like(y, np.nan)
    V = np.full_like(y, np.nan)

    theta[1:-1] = (y[2:] - y[:-2])/(2*dx)
    M[1:-1] = EI*(y[:-2] - 2*y[1:-1] + y[2:])/(dx**2)
    V[2:-2] = EI*(-y[:-4] + 2*y[1:-3] - 2*y[3:-1] + y[4:])/(2*dx**3)

    return x, y, theta, M, V


# ============================================================
# ANALYTICAL METHOD
# ============================================================

def solve_analytical():

    l = lam
    A = np.zeros((4,4))
    b = np.zeros(4)

    # y''(0) = M/EI
    A[0,:] = [l**2, 0, l**2, 0]
    b[0] = M_head/EI

    # y'''(0) = V/EI
    A[1,:] = [0, l**3, 0, -l**3]
    b[1] = V_head/EI

    # y(L)=0
    A[2,:] = [
        np.exp(l*L)*np.cos(l*L),
        np.exp(l*L)*np.sin(l*L),
        np.exp(-l*L)*np.cos(l*L),
        np.exp(-l*L)*np.sin(l*L)
    ]

    # y'(L)=0
    A[3,:] = [
        np.exp(l*L)*( l*np.cos(l*L) - l*np.sin(l*L)),
        np.exp(l*L)*( l*np.sin(l*L) + l*np.cos(l*L)),
        np.exp(-l*L)*(-l*np.cos(l*L) - l*np.sin(l*L)),
        np.exp(-l*L)*(-l*np.sin(l*L) + l*np.cos(l*L))
    ]

    C = np.linalg.solve(A,b)
    C1, C2, C3, C4 = C

    x = np.linspace(0, L, 400)

    y = (
        np.exp(l*x)*(C1*np.cos(l*x) + C2*np.sin(l*x))
        + np.exp(-l*x)*(C3*np.cos(l*x) + C4*np.sin(l*x))
    )

    dy = np.gradient(y, x)
    d2y = np.gradient(dy, x)
    d3y = np.gradient(d2y, x)

    theta = dy
    M = EI * d2y
    V = EI * d3y

    return x, y, theta, M, V


# ============================================================
# SOLVE BOTH METHODS
# ============================================================

x_fdm, y_fdm, th_fdm, M_fdm, V_fdm = solve_FDM(n=100)
x_an, y_an, th_an, M_an, V_an = solve_analytical()


# ============================================================
# PLOT COMPARISON
# ============================================================

fig, axs = plt.subplots(1,4, figsize=(16,6))

# Displacement
axs[0].plot(y_fdm*1000, x_fdm, label="FDM")
axs[0].plot(y_an*1000, x_an, 'o', markevery=20, label="Analytical")
axs[0].set_title("Displacement [mm]")
axs[0].invert_yaxis()

# Rotation
axs[1].plot(th_fdm*1e3, x_fdm)
axs[1].plot(th_an*1e3, x_an, 'o', markevery=20)
axs[1].set_title("Rotation [-] x10³")
axs[1].invert_yaxis()

# Moment
axs[2].plot(M_fdm/1e3, x_fdm)
axs[2].plot(M_an/1e3, x_an, 'o', markevery=20)
axs[2].set_title("Moment [kNm]")
axs[2].invert_yaxis()

# Shear
axs[3].plot(V_fdm/1e3, x_fdm)
axs[3].plot(V_an/1e3, x_an, 'o', markevery=20)
axs[3].set_title("Shear [kN]")
axs[3].invert_yaxis()

axs[0].legend()
plt.tight_layout()
plt.show()


# ============================================================
# PRINT MAX VALUES
# ============================================================

print("\nMaximum values:")
print("------------------------------")
print("FDM:")
print(f"Max displacement: {np.nanmax(np.abs(y_fdm))*1000:.2f} mm")
print(f"Max moment: {np.nanmax(np.abs(M_fdm))/1e3:.2f} kNm")
print(f"Max shear: {np.nanmax(np.abs(V_fdm))/1e3:.2f} kN")

print("\nAnalytical:")
print(f"Max displacement: {np.max(np.abs(y_an))*1000:.2f} mm")
print(f"Max moment: {np.max(np.abs(M_an))/1e3:.2f} kNm")
print(f"Max shear: {np.max(np.abs(V_an))/1e3:.2f} kN")