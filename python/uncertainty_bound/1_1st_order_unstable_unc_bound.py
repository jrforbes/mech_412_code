"""Uncertainty bound on 1st order unstable system.

James Richard Forbes, 2023/10/07
Modified 2025/01/11

Plant example from Mathworks ucover example,
https://www.mathworks.com/help/robust/ref/dynamicsystem.ucover.html
"""
# %%
# Libraries
import numpy as np
from matplotlib import pyplot as plt
import control
# import pathlib

# Custom libraries
import unc_bound

# %%
# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')
# path = pathlib.Path('figs')
# path.mkdir(exist_ok=True)

# %%
# Create uncertain transfer functions

# Laplace variable s
s = control.tf('s')

# First order TF
P = 2 / (s - 2)
P1 = P * 1 / (s * 0.06 + 1)
P2 = P * (-0.02 * s + 1) / (0.02 * s + 1)
P3 = P * (50 ** 2) / (s ** 2 + 2 * 0.1 * 50 * s + 50 ** 2)
P4 = P * (70 ** 2) / (s ** 2 + 2 * 0.2 * 70 * s + 70 ** 2)
P5 = 2.4 / (s - 2.2)
P6 = 1.6 / (s - 1.8)

# Uncertain plants as a list
P_off_nom = [P1, P2, P3, P4, P5, P6]
N = len(P_off_nom)


# %%
# Compute residuals
R = unc_bound.residuals(P, P_off_nom)

# Bode plot
N_w = 500
w_shared = np.logspace(-1, 3, N_w)

# Compute magnitude part of R(s) in both dB and in absolute units
mag_max_dB, mag_max_abs = unc_bound.residual_max_mag(R, w_shared)

# Plot Bode magnitude plot in dB and in absolute units
fig, ax = plt.subplots(2, 1)
ax[0].set_xlabel(r'$\omega$ (rad/s)')
ax[0].set_ylabel(r'Magnitude (dB)')
ax[1].set_xlabel(r'$\omega$ (rad/s)')
ax[1].set_ylabel(r'Magnitude (absolute)')
for i in range(N):
    mag_abs, _, _ = control.frequency_response(R[i], w_shared)
    mag_dB = 20 * np.log10(mag_abs)
    # Magnitude plot (dB)
    ax[0].semilogx(w_shared, mag_dB, '--', color='C0', linewidth=1)
    # Magnitude plot (absolute).
    ax[1].semilogx(w_shared, mag_abs, '--', color='C0', linewidth=1)

# Magnitude plot (dB).
ax[0].semilogx(w_shared, mag_max_dB, '-', color='C4', label='upper bound')
# Magnitude plot (absolute).
ax[1].semilogx(w_shared, mag_max_abs, '-', color='C4', label='upper bound')
ax[0].legend(loc='best')
ax[1].legend(loc='best')
# fig.savefig(path.joinpath('1st_order_unstable_R.pdf'))


# %%
# Find W2

# Order of W2
nW2 = 4

# Calculate optimal upper bound transfer function.
W2 = (unc_bound.upperbound(omega=w_shared, upper_bound=mag_max_abs, degree=nW2)).minreal()
print("The optimal weighting function W_2(s) is ", W2)


# %%
# Plot the Bode magnitude plot of the optimal W_2(s) transfer function

# Compute magnitude part of W_2(s) in absolute units
mag_W2_abs, _, _ = control.frequency_response(W2, w_shared)
# Compute magnitude part of W_2(s) in dB
mag_W2_dB = 20 * np.log10(mag_W2_abs)

fig, ax = plt.subplots(2, 1)
ax[0].set_xlabel(r'$\omega$ (rad/s)')
ax[0].set_ylabel(r'Magnitude (dB)')
ax[1].set_xlabel(r'$\omega$ (rad/s)')
ax[1].set_ylabel(r'Magnitude (absolute)')
for i in range(N):
    mag_abs, _, _ = control.frequency_response(R[i], w_shared)
    mag_dB = 20 * np.log10(mag_abs)
    # Magnitude plot (dB)
    ax[0].semilogx(w_shared, mag_dB, '--', color='C0', linewidth=1)
    # Magnitude plot (absolute).
    ax[1].semilogx(w_shared, mag_abs, '--', color='C0', linewidth=1)

# Magnitude plot (dB).
ax[0].semilogx(w_shared, mag_max_dB, '-', color='C4', label='upper bound')
# Magnitude plot (absolute).
ax[1].semilogx(w_shared, mag_max_abs, '-', color='C4', label='upper bound')
# Magnitude plot (dB).
ax[0].semilogx(w_shared, mag_W2_dB, '-', color='seagreen', label='optimal bound')
# Magnitude plot (absolute).
ax[1].semilogx(w_shared, mag_W2_abs, '-', color='seagreen', label='optimal bound')
ax[0].legend(loc='best')
ax[1].legend(loc='best')
# fig.tight_layout()
# fig.savefig(path.joinpath('1st_order_unstable_W2.pdf'))


# %%
# Plot
plt.show()

# %%
