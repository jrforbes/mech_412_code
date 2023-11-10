"""Control for robust performance.

James Forbes
2023/09/26

All custom functions ``siso_rob_perf" file.

My use this example code for homework, the course
project, etc. 

No warrenty, responsibility, etc. 
"""
# %%
# Packages
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import control

# Custom packages
import siso_rob_perf as srp

# %%
# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')


# %%
# Common parameters

# Golden ratio
gr = (1 + np.sqrt(5)) / 2

# Figure height
height = 4.25

# time
t_start, t_end, dt = 0, 20, 1e-2
t = np.arange(t_start, t_end, dt)
n_t = t.shape[0]

# Laplace variable
s = control.tf('s')

# Frequencies for Bode plot
w_shared_low, w_shared_high, N_w = -2, 2, 500
w_shared = np.logspace(w_shared_low, w_shared_high, N_w)

# Frequencies for Nyquist plot
w_shared_low_2, w_shared_high_2, N_w_2 = -6, 6, 5000
w_shared_2 = np.logspace(w_shared_low_2, w_shared_high_2, N_w_2)

# %%
# Create nominal model, uncertainty weight, and off-nominal models

# Nominal model
g = 10 # unitless
m = 1  # kg
d = 2  # N s / m
P = g / (m * s**2 + d * s)

# W2, uncertainty weight
W2_num = np.array([1.01097854e+00, 3.03303632e+01, 3.43858706e+02, 1.09902748e+03, 1.05325567e+03])
W2_den = np.array([1.00000000e+00, 2.75243287e+01, 3.53177231e+02, 1.66166012e+03, 2.52102297e+03])
W2 = control.tf(W2_num, W2_den)
W2_inv = 1 / W2

# Off nominal plants sampled from W2 where \Delta is a real number less than one.
N_off_nom = 10
P_off_nom = [P * (1 + W2 * i / N_off_nom) for i in range(-N_off_nom, N_off_nom + 1, 1)]

# %%
# Performance

# Noise and reference bounds
gamma_r, w_r_h = 10**(-5 / 20), 0.35
gamma_d, w_d_h = 10**(-5 / 20), 0.05
gamma_n, w_n_l = 10**(-40 / 20), 20
gamma_u, w_u_l = 10**(-10 / 20), 20

# Set up design specifications plot
w_r = np.logspace(w_shared_low, np.log10(w_r_h), 100)
w_d = np.logspace(w_shared_low, np.log10(w_d_h), 100)
w_n = np.logspace(np.log10(w_n_l), w_shared_high, 100)
w_u = np.logspace(np.log10(w_u_l), w_shared_high, 100)

# In dB
gamma_r_dB = 20 * np.log10(gamma_r) * np.ones(w_r.shape[0],)
gamma_d_dB = 20 * np.log10(gamma_d) * np.ones(w_d.shape[0],)
gamma_n_dB = 20 * np.log10(gamma_n) * np.ones(w_n.shape[0],)
gamma_u_dB = 20 * np.log10(gamma_u) * np.ones(w_u.shape[0],)

# Weight W_1(s) according to Zhou et al.
k = 2
epsilon = 10**(-40 / 20)
Ms = 10**(40 / 20)
w1 = 0.5
W1 = ((s / Ms**(1 / k) + w1) / (s + w1 * (epsilon)**(1 / k)))**k
W1_inv = 1 / W1


# %%
# Plot both weights, W1 and W2 (and their inverses).

fig, ax = srp.bode_mag_W1_W2(W1, W2, w_d_h, w_n_l, w_shared)
fig.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower left')
# fig.savefig('figs/bode_W1_W2.pdf')

fig, ax = srp.bode_mag_W1_inv_W2_inv(W1, W2, gamma_r, w_r_h, w_d_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared)
fig.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='upper left')
# fig.savefig('figs/bode_W1_W2_design_specs.pdf')


# %%
# Nyquist of open-loop plant withouth control
wmin, wmax, N_w_robust_nyq = -1, 2, 500
count, fig, ax = srp.robust_nyq(P, P_off_nom, W2, wmin, wmax, N_w_robust_nyq)
fig.tight_layout()
# fig.savefig('figs/nyquist_P_W2.pdf')


# %%
# 0. Crossover frequency (bandwidth), w_c

# The midway point between high and low is the geometric mean.
# https://stackoverflow.com/questions/30908600/visually-midway-between-two-points-on-a-x-axis-log-scale
# 10^((log(a) + log(b)) / 2) = 10^((log(a * b) / 2) = ( 10^((log(a * b) )^(1/2) = sqrt(a * b) 

w_c = np.sqrt(w_n_l * w_d_h)  # Geometric mean
print(r'w_c = ', w_c, '\n')
w_c = 1.25  # Push w_c a bit higher to maximize performance
print(r'w_c = ', w_c, '\n')

# %%
# 1. Gain
k_g = 1 / np.linalg.norm((P.horner(w_c)).ravel(), 2)
print(r'k_g = ', k_g, '\n')
C = k_g * control.tf([1], [1])

'''
fig_L, ax = srp.bode_mag_L(P, C, gamma_r, w_r_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared)
fig_L.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower left')
'''

fig_S_T, ax = srp.bode_mag_S_T(P, C, gamma_r, w_r_h, w_d_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

fig_L_W1_W2_inv, ax = srp.bode_mag_L_W1_W2_inv(P, C, W1, W2, gamma_r, w_r_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared)
fig_L_W1_W2_inv.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower left')

fig_L_W1_W2_inv, ax = srp.bode_mag_L_W1_W2_inv(P, C, W1, W2, gamma_r, w_r_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared)
fig_L_W1_W2_inv.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower left')

fig_RP, ax = srp.bode_mag_rob_perf(P, C, W1, W2, w_shared)
fig_RP.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower left')

fig_RP_RD, ax = srp.bode_mag_rob_perf_RD(P, C, W1, W2, w_shared)
fig_RP_RD.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='upper right')

fig_S_T_W1_inv_W2_inv, ax = srp.bode_mag_S_T_W1_inv_W2_inv(P, C, W1, W2, gamma_r, w_r_h, gamma_d, w_d_h, gamma_n, w_n_l, gamma_u, w_u_l, w_shared_low, w_shared_high, w_shared)
fig_S_T_W1_inv_W2_inv.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower center')
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

fig_L_P, ax = srp.bode_mag_L_P(P, C, gamma_d, w_d_h, gamma_u, w_u_l, w_shared)
fig_L_P.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower left')

fig_L_P_C, ax = srp.bode_mag_L_P_C(P, C, gamma_r, w_r_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared)
fig_L_P_C.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower left')

fig_margins, ax, gm, pm, vm, wpc, wgc, wvm = srp.bode_margins(P, C, w_shared)
fig_margins.set_size_inches(height * gr, height, forward=True)
print(f'\nGain margin is', 20 * np.log10(gm),
      '(dB) at phase crossover frequency', wpc, '(rad/s)')
print(f'Phase margin is', pm, '(deg) at gain crossover frequency',
      wgc, '(rad/s)')
print(f'Vector margin is', vm, 'at frequency', wvm, '(rad/s)\n')

fig_Gof4, ax = srp.bode_mag_Gof4(P, C, gamma_r, w_r_h, gamma_d, w_d_h, gamma_n, w_n_l, gamma_u, w_u_l, w_shared_low, w_shared_high, w_shared)
fig_Gof4.set_size_inches(height * gr, height, forward=True)

# Nyquist
fig_Nyquist, ax_Nyquist = plt.subplots()
count, contour = control.nyquist_plot(control.minreal(P * C),
                                      omega=w_shared_2,
                                      plot=True,
                                      return_contour=True)
# ax_Nyquist.axis('equal')
fig_Nyquist.tight_layout()

'''
# fig_L.savefig('figs/L_C1.pdf')
# fig_S_T.savefig('figs/S_T_C1.pdf')
fig_L_W1_W2_inv.savefig('figs/L_W1_W2_inv_C1.pdf')
fig_L_W1_W2_inv.savefig('figs/L_W1_W2_inv_C1.pdf')
fig_RP.savefig('figs/RP_C1.pdf')
fig_RP_RD.savefig('figs/RP_RD_C1.pdf')
fig_S_T_W1_inv_W2_inv.savefig('figs/S_T_weights_C1.pdf')
fig_L_P.savefig('figs/L_P_C1.pdf')
fig_L_P_C.savefig('figs/L_P_C_C1.pdf')
fig_margins.savefig('figs/margins_C1.pdf')
fig_Gof4.savefig('figs/Gof4_C1.pdf')
fig_Nyquist.savefig('figs/Nyquist_C1.pdf')
'''

# %%
# 2. Integral boost
w_beta = w_c
beta = 4  # np.sqrt(10)
C_boost = (beta * s + w_beta) / (s * np.sqrt(beta**2 + 1))
C = k_g * C_boost

'''
fig_L, ax = srp.bode_mag_L(P, C, gamma_r, w_r_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared)
fig_L.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower left')
'''

fig_S_T, ax = srp.bode_mag_S_T(P, C, gamma_r, w_r_h, w_d_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

fig_L_W1_W2_inv, ax = srp.bode_mag_L_W1_W2_inv(P, C, W1, W2, gamma_r, w_r_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared)
fig_L_W1_W2_inv.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower left')

fig_RP, ax = srp.bode_mag_rob_perf(P, C, W1, W2, w_shared)
fig_RP.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower left')

fig_RP_RD, ax = srp.bode_mag_rob_perf_RD(P, C, W1, W2, w_shared)
fig_RP_RD.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='upper right')

fig_S_T_W1_inv_W2_inv, ax = srp.bode_mag_S_T_W1_inv_W2_inv(P, C, W1, W2, gamma_r, w_r_h, gamma_d, w_d_h, gamma_n, w_n_l, gamma_u, w_u_l, w_shared_low, w_shared_high, w_shared)
fig_S_T_W1_inv_W2_inv.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower center')
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

fig_L_P, ax = srp.bode_mag_L_P(P, C, gamma_d, w_d_h, gamma_u, w_u_l, w_shared)
fig_L_P.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower left')

fig_L_P_C, ax = srp.bode_mag_L_P_C(P, C, gamma_r, w_r_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared)
fig_L_P_C.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower left')

fig_margins, ax, gm, pm, vm, wpc, wgc, wvm = srp.bode_margins(P, C, w_shared)
fig_margins.set_size_inches(height * gr, height, forward=True)
print(f'\nGain margin is', 20 * np.log10(gm),
      '(dB) at phase crossover frequency', wpc, '(rad/s)')
print(f'Phase margin is', pm, '(deg) at gain crossover frequency',
      wgc, '(rad/s)')
print(f'Vector margin is', vm, 'at frequency', wvm, '(rad/s)\n')

fig_Gof4, ax = srp.bode_mag_Gof4(P, C, gamma_r, w_r_h, gamma_d, w_d_h, gamma_n, w_n_l, gamma_u, w_u_l, w_shared_low, w_shared_high, w_shared)
fig_Gof4.set_size_inches(height * gr, height, forward=True)

# Nyquist
fig_Nyquist, ax_Nyquist = plt.subplots()
count, contour = control.nyquist_plot(control.minreal(P * C),
                                      omega=w_shared_2,
                                      plot=True,
                                      return_contour=True)
# ax_Nyquist.axis('equal')
fig_Nyquist.tight_layout()

'''
# fig_L.savefig('figs/L_C2.pdf')
# fig_S_T.savefig('figs/S_T_C2.pdf')
fig_L_W1_W2_inv.savefig('figs/L_W1_W2_inv_C2.pdf')
fig_RP.savefig('figs/RP_C2.pdf')
fig_RP_RD.savefig('figs/RP_RD_C2.pdf')
fig_S_T_W1_inv_W2_inv.savefig('figs/S_T_weights_C2.pdf')
fig_L_P.savefig('figs/L_P_C2.pdf')
fig_L_P_C.savefig('figs/L_P_C_C2.pdf')
fig_margins.savefig('figs/margins_C2.pdf')
fig_Gof4.savefig('figs/Gof4_C2.pdf')
fig_Nyquist.savefig('figs/Nyquist_C2.pdf')
'''

# %%
# 3. lead
w_lam = w_c
lam = 1.1  # 1 <= lam
C_lead = (lam * s + w_lam) / (s + lam * w_lam)
C = k_g * C_boost * C_lead

'''
fig_L, ax = srp.bode_mag_L(P, C, gamma_r, w_r_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared)
fig_L.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower left')
'''

fig_S_T, ax = srp.bode_mag_S_T(P, C, gamma_r, w_r_h, w_d_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

fig_L_W1_W2_inv, ax = srp.bode_mag_L_W1_W2_inv(P, C, W1, W2, gamma_r, w_r_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared)
fig_L_W1_W2_inv.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower left')

fig_RP, ax = srp.bode_mag_rob_perf(P, C, W1, W2, w_shared)
fig_RP.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower left')

fig_RP_RD, ax = srp.bode_mag_rob_perf_RD(P, C, W1, W2, w_shared)
fig_RP_RD.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='upper right')

fig_S_T_W1_inv_W2_inv, ax = srp.bode_mag_S_T_W1_inv_W2_inv(P, C, W1, W2, gamma_r, w_r_h, gamma_d, w_d_h, gamma_n, w_n_l, gamma_u, w_u_l, w_shared_low, w_shared_high, w_shared)
fig_S_T_W1_inv_W2_inv.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower center')
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

fig_L_P, ax = srp.bode_mag_L_P(P, C, gamma_d, w_d_h, gamma_u, w_u_l, w_shared)
fig_L_P.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower left')

fig_L_P_C, ax = srp.bode_mag_L_P_C(P, C, gamma_r, w_r_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared)
fig_L_P_C.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower left')

fig_margins, ax, gm, pm, vm, wpc, wgc, wvm = srp.bode_margins(P, C, w_shared)
fig_margins.set_size_inches(height * gr, height, forward=True)
print(f'\nGain margin is', 20 * np.log10(gm),
      '(dB) at phase crossover frequency', wpc, '(rad/s)')
print(f'Phase margin is', pm, '(deg) at gain crossover frequency',
      wgc, '(rad/s)')
print(f'Vector margin is', vm, 'at frequency', wvm, '(rad/s)\n')

fig_Gof4, ax = srp.bode_mag_Gof4(P, C, gamma_r, w_r_h, gamma_d, w_d_h, gamma_n, w_n_l, gamma_u, w_u_l, w_shared_low, w_shared_high, w_shared)
fig_Gof4.set_size_inches(height * gr, height, forward=True)

# Nyquist
fig_Nyquist, ax_Nyquist = plt.subplots()
count, contour = control.nyquist_plot(control.minreal(P * C),
                                      omega=w_shared_2,
                                      plot=True,
                                      return_contour=True)
# ax_Nyquist.axis('equal')
fig_Nyquist.tight_layout()

'''
# fig_L.savefig('figs/L_C3.pdf')
# fig_S_T.savefig('figs/S_T_C3.pdf')
fig_L_W1_W2_inv.savefig('figs/L_W1_W2_inv_C3.pdf')
fig_RP.savefig('figs/RP_C3.pdf')
fig_RP_RD.savefig('figs/RP_RD_C3.pdf')
fig_S_T_W1_inv_W2_inv.savefig('figs/S_T_weights_C3.pdf')
fig_L_P.savefig('figs/L_P_C3.pdf')
fig_L_P_C.savefig('figs/L_P_C_C3.pdf')
fig_margins.savefig('figs/margins_C3.pdf')
fig_Gof4.savefig('figs/Gof4_C3.pdf')
fig_Nyquist.savefig('figs/Nyquist_C3.pdf')
'''

# %%
# 4. Roll-off
w_rho = w_c
rho = 15  # 1 <= rho
C_roll = (w_rho * np.sqrt(rho**2 + 1)) / (s + rho * w_rho)
C = k_g * C_boost * C_roll
C = k_g * C_boost * C_lead * C_roll

'''
fig_L, ax = srp.bode_mag_L(P, C, gamma_r, w_r_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared)
fig_L.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower left')
'''

fig_S_T, ax = srp.bode_mag_S_T(P, C, gamma_r, w_r_h, w_d_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

fig_L_W1_W2_inv, ax = srp.bode_mag_L_W1_W2_inv(P, C, W1, W2, gamma_r, w_r_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared)
fig_L_W1_W2_inv.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower left')

fig_RP, ax = srp.bode_mag_rob_perf(P, C, W1, W2, w_shared)
fig_RP.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower left')

fig_RP_RD, ax = srp.bode_mag_rob_perf_RD(P, C, W1, W2, w_shared)
fig_RP_RD.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='upper right')

fig_S_T_W1_inv_W2_inv, ax = srp.bode_mag_S_T_W1_inv_W2_inv(P, C, W1, W2, gamma_r, w_r_h, gamma_d, w_d_h, gamma_n, w_n_l, gamma_u, w_u_l, w_shared_low, w_shared_high, w_shared)
fig_S_T_W1_inv_W2_inv.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower center')
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

fig_L_P, ax = srp.bode_mag_L_P(P, C, gamma_d, w_d_h, gamma_u, w_u_l, w_shared)
fig_L_P.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower left')

fig_L_P_C, ax = srp.bode_mag_L_P_C(P, C, gamma_r, w_r_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared)
fig_L_P_C.set_size_inches(height * gr, height, forward=True)
ax.legend(loc='lower left')

fig_margins, ax, gm, pm, vm, wpc, wgc, wvm = srp.bode_margins(P, C, w_shared)
fig_margins.set_size_inches(height * gr, height, forward=True)
print(f'\nGain margin is', 20 * np.log10(gm),
      '(dB) at phase crossover frequency', wpc, '(rad/s)')
print(f'Phase margin is', pm, '(deg) at gain crossover frequency',
      wgc, '(rad/s)')
print(f'Vector margin is', vm, 'at frequency', wvm, '(rad/s)\n')

fig_Gof4, ax = srp.bode_mag_Gof4(P, C, gamma_r, w_r_h, gamma_d, w_d_h, gamma_n, w_n_l, gamma_u, w_u_l, w_shared_low, w_shared_high, w_shared)
fig_Gof4.set_size_inches(height * gr, height, forward=True)

# Nyquist
fig_Nyquist, ax_Nyquist = plt.subplots()
count, contour = control.nyquist_plot(control.minreal(P * C),
                                      omega=w_shared_2,
                                      plot=True,
                                      return_contour=True)
# ax_Nyquist.axis('equal')
fig_Nyquist.tight_layout()

'''
# fig_L.savefig('figs/L_C4.pdf')
# fig_S_T.savefig('figs/S_T_C4.pdf')
fig_L_W1_W2_inv.savefig('figs/L_W1_W2_inv_C4.pdf')
fig_RP.savefig('figs/RP_C4.pdf')
fig_RP_RD.savefig('figs/RP_RD_C4.pdf')
fig_S_T_W1_inv_W2_inv.savefig('figs/S_T_weights_C4.pdf')
fig_L_P.savefig('figs/L_P_C4.pdf')
fig_L_P_C.savefig('figs/L_P_C_C4.pdf')
fig_margins.savefig('figs/margins_C4.pdf')
fig_Gof4.savefig('figs/Gof4_C4.pdf')
fig_Nyquist.savefig('figs/Nyquist_C4.pdf')
'''

# %%
# Robust Nyquist plot to assess robustness
L_off_nom = [C * P * (1 + W2 * i / N_off_nom) for i in range(-N_off_nom, N_off_nom + 1, 1)]
wmin, wmax, N_w_robust_nyq = 0.05, 2, 500
count, fig, ax = srp.robust_nyq(control.minreal(P * C), L_off_nom, W2, wmin, wmax, N_w_robust_nyq)
ax.axis('equal')
fig.tight_layout()
# fig.savefig('figs/nyquist_L_W2.pdf')

# %%
# References

# Create command to follow
r_max = 2
N = np.divmod(n_t, 4)[0]
N = n_t - N
r = r_max * np.ones(n_t)
r[N:] = r[N:] - r_max * np.ones(n_t - N)
a = 3
_, r = control.forced_response(1 / (1 / a * s + 1), t, r, 0)

# Noise
np.random.seed(123321)
mu, sigma = 0, 1 * gamma_n
noise_raw = np.random.normal(mu, sigma, n_t)
# Butterworth filter, high pass
b_bf, a_bf = signal.butter(6, w_n_l, 'high', analog=True)
G_bf = control.tf(b_bf, a_bf)
_, noise = control.forced_response(G_bf, t, noise_raw)

# %%
# Time-domine response

# Create T, S, and CS
T = control.feedback(P * C, 1, -1)
S = control.feedback(1, P * C, -1)
CS = control.minreal(C * S)

# Forced Response
_, z = control.forced_response(T, t, r - noise)
_, u = control.forced_response(CS, t, r - noise)

# Plot
fig, ax = plt.subplots(2, 1, figsize=(height * gr, height))
ax[0].set_ylabel(r'$z(t)$ (units)')
ax[1].set_ylabel(r'$u(t)$ (units)')
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')

ax[0].plot(t, r, '--', label=r'$r(t)$', color='C3')
ax[0].plot(t, z, '-', label=r'$z(t)$', color='C0')
ax[1].plot(t, u, '-', label=r'$u(t)$', color='C1')
ax[0].legend(loc='upper right')
fig.tight_layout()
# fig.savefig('figs/time_dom_response.pdf')

# Plot
fig, ax = plt.subplots(figsize=(height * gr, height))
ax.set_ylabel(r'$z(t)$ (units)')
ax.set_xlabel(r'$t$ (s)')
ax.plot(t, r, '--', label=r'$r(t)$', color='C3')
ax.plot(t, z, '-', label=r'$z(t)$', color='C0')
ax.legend(loc='upper right')
fig.tight_layout()
# fig.savefig('figs/time_dom_response.pdf')


# %%
# Plot
plt.show()
