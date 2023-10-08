"""Uncertainty bound on 1st order unstable system.

James Richard Forbes, 2023/10/07

Plant example from Mathwroks ucover example,
https://www.mathworks.com/help/robust/ref/dynamicsystem.ucover.html
"""
# %%
# Libraries
import numpy as np
from matplotlib import pyplot as plt
import control
import pathlib

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

# Compute residules
R = unc_bound.residuals(P, P_off_nom)

# %%
# Bode plot
N_w = 250
w_shared = np.logspace(-1, 3, N_w)

# Compute magnitude part of R(s) in both dB and in absolute units
mag_max_dB, mag_max_abs = unc_bound.residual_max_mag(R, w_shared)

# Initial guss for W_2(s)
# W_2(s) must be biproper. Will parameterize it as
# W_2(s) = kappa * ( (s / wb1)**2 + 2 * zb1 / wb1 * s + 1 ) / ( (s / wa1)**2 + 2 * za1 / wa1 * s + 1 ) ...  # noqa
# Numerator
wb1 = 4  # rad/s
zb1 = .99
wb2 = 150  # rad/s
zb2 = 0.99
# Denominator
wa1 = 30  # rad/s
za1 = 0.99
wa2 = 60  # rad/s
za2 = 0.8
# DC gain
kappa = 0.3
#
s = control.tf('s')
G1 = ((s / wb1)**2 + 2 * zb1 / wb1 * s + 1) / (
    (s / wa1)**2 + 2 * za1 / wa1 * s + 1)
G2 = ((s / wb2)**2 + 2 * zb2 / wb2 * s + 1) / (
    (s / wa2)**2 + 2 * za2 / wa2 * s + 1)
W20 = kappa * G1 * G2

# Compute magnitude part of W_2(s) in absolute units
mag_W20_abs, _, w = control.bode(W20, w_shared, plot=False)
# Copmute magnitude part of W_2(s) in dB
mag_W20_dB = 20 * np.log10(mag_W20_abs)

# Plot Bode magntude plot in dB and in absolute units
fig, ax = plt.subplots(2, 1)
ax[0].set_xlabel(r'$\omega$ (rad/s)')
ax[0].set_ylabel(r'Magnitude (dB)')
ax[1].set_xlabel(r'$\omega$ (rad/s)')
ax[1].set_ylabel(r'Magnitude (absolute)')
for i in range(N):
    mag_abs, _, _ = control.bode(R[i], w_shared, plot=False)
    mag_dB = 20 * np.log10(mag_abs)
    # Magnitude plot (dB)
    ax[0].semilogx(w, mag_dB, '--', color='C0', linewidth=1)
    # Magnitude plot (absolute).
    ax[1].semilogx(w, mag_abs, '--', color='C0', linewidth=1)

# Magnitude plot (dB).
ax[0].semilogx(w, mag_max_dB, '-', color='C4', label='upper bound')
# Magnitude plot (absolute).
ax[1].semilogx(w, mag_max_abs, '-', color='C4', label='upper bound')
# Magnitude plot (dB).
ax[0].semilogx(w, mag_W20_dB, '-', color='C1', label='initial bound')
# Magnitude plot (absolute).
ax[1].semilogx(w, mag_W20_abs, '-', color='C1', label='initial bound')
ax[0].legend(loc='lower right')
ax[1].legend(loc='upper left')
# fig.savefig(path.joinpath('1st_order_unstable_R_W2_IC.pdf'))

# %%
# Optimize
# Design variable initial conditions
x0 = np.array([wa1, za1, wa2, za2, wb1, zb1, wb2, zb2, kappa])

# Size of design variables
n_x = x0.size

# Lower bound on the design variables x. Can't be zero else forming TFs fails.
lb = 1e-4 * np.ones(n_x, )
# Upper bound on the design variables x.
ub = np.ones(n_x, )  # Upper bound on the design variables x.

# Upper bound on DC gain
ub[-1] = 10

# Add more specific lb and ub constrints.
# Don't let the max natural frequency be above omega_max
omega_max = w_shared[-1]
# Don't let the max zeta be above zeta_max
zeta_max = 100
# Don't let the minimum zeta (damping ratio) be less than zeta_min
zeta_min = 0.2
# Form the upper and lower bound np arrays.
for i in range(0, n_x - 1, 2):
    ub[i] = omega_max * ub[i]
    ub[i + 1] = zeta_max * ub[i + 1]
    lb[i + 1] = lb[i + 1] + zeta_min

# Specify max number of iterations.
max_iter = 5000

# Run optimization
x_opt, f_opt, objhist, xlast = unc_bound.run_optimization(
    x0, lb, ub, max_iter, w_shared, mag_max_abs)

# Compute the optimal W_2(s)
W2 = unc_bound.extract_W2(x_opt)

print("The optimal weighting function W_2(s) is ", W2)

# Compute magnitude part of W_2(s) in absolute units
mag_W2_abs, _, _ = control.bode(W2, w_shared, plot=False)
# Compute magnitude part of W_2(s) in dB
mag_W2_dB = 20 * np.log10(mag_W2_abs)

# Plot the opimization objective function as a function of iterations
fig, ax = plt.subplots()
ax.set_xlabel(r'iteration, k')
ax.set_ylabel(r'objective function, $f(x)$')
ax.semilogy(objhist, '-', color='C3', label=r'$f(x_k)$')
ax.legend(loc='upper right')
fig.tight_layout()
# fig.savefig('figs/obj_vs_iter.pdf')

# %%
# Plot the Bode magnitute plot of the optimal W_2(s) tranfer function
fig, ax = plt.subplots(2, 1)
ax[0].set_xlabel(r'$\omega$ (rad/s)')
ax[0].set_ylabel(r'Magnitude (dB)')
ax[1].set_xlabel(r'$\omega$ (rad/s)')
ax[1].set_ylabel(r'Magnitude (absolute)')
for i in range(N):
    mag_abs, _, _ = control.bode(R[i], w_shared, plot=False)
    mag_dB = 20 * np.log10(mag_abs)
    # Magnitude plot (dB)
    ax[0].semilogx(w, mag_dB, '--', color='C0', linewidth=1)
    # Magnitude plot (absolute).
    ax[1].semilogx(w, mag_abs, '--', color='C0', linewidth=1)

# Magnitude plot (dB).
ax[0].semilogx(w, mag_max_dB, '-', color='C4', label='upper bound')
# Magnitude plot (absolute).
ax[1].semilogx(w, mag_max_abs, '-', color='C4', label='upper bound')
# Magnitude plot (dB).
ax[0].semilogx(w, mag_W2_dB, '-', color='seagreen', label='optimal bound')
# Magnitude plot (absolute).
ax[1].semilogx(w, mag_W2_abs, '-', color='seagreen', label='optimal bound')
ax[0].legend(loc='lower right')
ax[1].legend(loc='upper left')
# fig.tight_layout()
# fig.savefig(path.joinpath('1st_order_unstable_W2.pdf'))

# %%
# Plot
plt.show()
