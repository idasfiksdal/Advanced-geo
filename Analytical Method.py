import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# INPUT DATA
# ============================================================

L = 16.0                  # [m]
D = 2.5                   # [m]
Ep = 35.5e9               # [Pa]
Epy = 68.07e6             # [Pa]

V_head = 6.5e3            # [N]
M_head = 151.45e3         # [Nm]

# ============================================================
# DERIVED VALUES
# ============================================================

I = np.pi * D**4 / 64
EI = Ep * I
lam = (Epy/(4*EI))**0.25

# ============================================================
# SET UP SYSTEM FOR C1–C4
# ============================================================

# Define shorthand
l = lam

# Expressions evaluated at x=0
# Using symbolic derivatives evaluated manually

# y(0)
# y''(0)
# y'''(0)

A = np.zeros((4,4))
b = np.zeros(4)

# --- Boundary 1: y''(0) = M/EI ---
A[0,:] = [
    l**2,      # C1
    0,         # C2
    l**2,      # C3
    0          # C4
]
b[0] = M_head/EI

# --- Boundary 2: y'''(0) = V/EI ---
A[1,:] = [
    0,
    l**3,
    0,
    -l**3
]
b[1] = V_head/EI

# --- Boundary 3: y(L)=0 ---
A[2,:] = [
    np.exp(l*L)*np.cos(l*L),
    np.exp(l*L)*np.sin(l*L),
    np.exp(-l*L)*np.cos(l*L),
    np.exp(-l*L)*np.sin(l*L)
]
b[2] = 0

# --- Boundary 4: y'(L)=0 ---
A[3,:] = [
    np.exp(l*L)*( l*np.cos(l*L) - l*np.sin(l*L)),
    np.exp(l*L)*( l*np.sin(l*L) + l*np.cos(l*L)),
    np.exp(-l*L)*(-l*np.cos(l*L) - l*np.sin(l*L)),
    np.exp(-l*L)*(-l*np.sin(l*L) + l*np.cos(l*L))
]
b[3] = 0

# Solve for constants
C = np.linalg.solve(A,b)
C1, C2, C3, C4 = C

# ============================================================
# COMPUTE RESPONSE
# ============================================================

x = np.linspace(0, L, 400)

y = (
    np.exp(l*x)*(C1*np.cos(l*x) + C2*np.sin(l*x))
    + np.exp(-l*x)*(C3*np.cos(l*x) + C4*np.sin(l*x))
)

# Derivatives
dy = np.gradient(y, x)
d2y = np.gradient(dy, x)
d3y = np.gradient(d2y, x)

theta = dy
M = EI * d2y
V = EI * d3y

# ============================================================
# PLOT
# ============================================================

fig, axs = plt.subplots(1,4, figsize=(14,6))

axs[0].plot(y*1000, x)
axs[0].set_title("Displacement [mm]")
axs[0].invert_yaxis()

axs[1].plot(theta*1e3, x)
axs[1].set_title("Rotation [-] x10³")
axs[1].invert_yaxis()

axs[2].plot(M/1e3, x)
axs[2].set_title("Moment [kNm]")
axs[2].invert_yaxis()

axs[3].plot(V/1e3, x)
axs[3].set_title("Shear [kN]")
axs[3].invert_yaxis()

plt.tight_layout()
plt.show()

# ============================================================
# PRINT MAX VALUES
# ============================================================

print("\nMaximum values from analytical solution:")
print("------------------------------------------")
print(f"Max displacement: {np.max(np.abs(y))*1000:.2f} mm")
print(f"Max moment: {np.max(np.abs(M))/1e3:.2f} kNm")
print(f"Max shear: {np.max(np.abs(V))/1e3:.2f} kN")