"""Python control basics.

J R Forbes, 2022/03/27

Based on
https://python-control.readthedocs.io/en/0.9.0/
https://python-control.readthedocs.io/en/0.9.0/control.html#function-ref
https://jckantor.github.io/CBE30338/
https://jckantor.github.io/CBE30338/05.03-Creating-Bode-Plots.html
"""

# %%
# Packages
import numpy as np
import control
from scipy import signal
from matplotlib import pyplot as plt

# %%
# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# time
t_start, t_end, dt = 0, 5, 1e-2
t = np.arange(t_start, t_end, dt)

# %%
# Create systems
# First-order transfer function, P(s) = 1 / (tau * s + 1)
tau = 1 / 10
P_a = control.tf([1], [tau, 1])

# Second-order transfer function with zero
# P(s) = omega**2 * (s / b + 1) / (s**2 + 2 * zeta * omega * s + omega**2)
b = 10
omega = 5
zeta = 0.5
s = control.tf('s')
# Short way to create transfer function
P_b = omega**2 * (1 / b * s + 1) / (s**2 + 2 * zeta * omega * s + omega**2)
# Long way to create transfer function
# P_b = control.tf([omega**2 / b, omega**2], [1, 2 * zeta * omega, omega**2])
P_b_num = np.array(P_b.num).ravel()
P_b_den = np.array(P_b.den).ravel()
P_b_zeros = np.roots(P_b_num)
P_b_poles = np.roots(P_b_den)
print(f'plant num = {P_b_num}\n')
print(f'plant den = {P_b_den}\n')
print(f'plant zeros = {P_b_zeros}\n')
print(f'plant poles = {P_b_poles}\n')

# Mass-spring_damper system
m = 1  # kg, mass
d = 0.05  # N s / m, damper
k = 1  # N / m, spring
# Form state-space matrices.
A = np.array([[0, 1],
              [-k / m, -d / m]])
B = np.array([[0],
              [1 / m]])
C = np.array([[1, 0]])
D = np.array([[0]])
x0 = np.array([0.25, -0.5])  # Initial conditions
P_c = control.ss(A, B, C, D)
# breakpoint()

# Single integrator plant with delay
num_delay, den_delay = control.pade(0.01, 2)  # Pade approximation, 2rd order
P_d = 1 / s * control.tf(num_delay, den_delay)

# PI Control
C_PI = 10 * (0.5 + 1 / s)

# Open-loop transfer function
L = P_d * C_PI
print(f'\nL(s) =', L)

# Roots of the closed-loop characteristic polynomial
char_poly_tf = (control.tf(P_d.den, 1) * control.tf(C_PI.den, 1)
                + control.tf(P_d.num, 1) * control.tf(C_PI.num, 1))
char_poly = np.array(char_poly_tf.num).ravel()
char_poly_roots = np.roots(char_poly)
print(f'The characteristic polynomial (CP) roots are {char_poly_roots}\n')
print(f'The real part CP roots are {np.real(char_poly_roots)}\n')

# A plant * controller with an interesting Bode plot and Nyquist plot
L_e = 3 / (s**2 + s) / (s + 2)
print(f'plant zeros = {control.zeros(L_e)}\n')
print(f'plant poles = {control.poles(L_e)}\n')


# %%
# System interconnections
# Feedback interconnection of plant and control
# Complementary sensitivity transfer function
T = control.feedback(L, 1, -1)
S = 1 - T  # Sensitivity transfer function
print(f'\nT(s) =', T)
print(f'\nS(s) =', S)

# %%
# Step response
t_a, y_a = control.step_response(P_a, t)
t_b, y_b = control.step_response(P_b, t)
t_c, y_c = control.step_response(P_c, t, x0)
t_T, y_T = control.step_response(T, t)

# Plot step response
fig, ax = plt.subplots()
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'$y(t)$ (units)')
# Plot data
ax.plot(t_a, y_a, label='$P_a(s)$', color='C1')
ax.plot(t_b, y_b, label='$P_b(s)$', color='C2')
ax.plot(t_c, y_c, label='$P_c(s)$', color='C3')
ax.plot(t_T, y_T, label='$T(s)$', color='C4')
ax.legend(loc='upper right')
fig.tight_layout()
# fig.savefig('figs/control_step_response.pdf')

# %%
# Impulse response
t_a, y_a = control.impulse_response(P_a, t)
t_b, y_b = control.impulse_response(P_b, t)
t_c, y_c = control.impulse_response(P_c, t)
t_T, y_T = control.impulse_response(T, t)

# Plot impulse response
fig, ax = plt.subplots()
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'$y(t)$ (units)')
# Plot data
ax.plot(t_a, y_a, label='$P_a(s)$', color='C1')
ax.plot(t_b, y_b, label='$P_b(s)$', color='C2')
ax.plot(t_c, y_c, label='$P_c(s)$', color='C3')
ax.plot(t_T, y_T, label='$T(s)$', color='C4')
ax.legend(loc='upper right')
fig.tight_layout()
# fig.savefig('figs/control_impulse_response.pdf')

# %%
# Initial condition (IC) response
t_c, y_c = control.initial_response(P_c, t, x0)

# Plot step response
fig, ax = plt.subplots()
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'$y(t)$ (units)')
# Plot data
ax.plot(t_c, y_c, label='$P_c(s)$', color='C3')
ax.legend(loc='upper right')
fig.tight_layout()
# fig.savefig('figs/control_IC_response.pdf')

# %%
# Time-domaine forced response
# Square wave input
u = signal.square(2 * np.pi / 2 * t)

# Forced response of each system
t_a, y_a = control.forced_response(P_a, t, u)
t_b, y_b = control.forced_response(P_b, t, u)
t_c, y_c = control.forced_response(P_c, t, u, x0)
t_T, y_T = control.forced_response(T, t, u)

# Plot forced response
fig, ax = plt.subplots(2, 1)   
ax[0].set_ylabel(r'$u(t)$ (units)')
ax[1].set_ylabel(r'$y(t)$ (units)')
# Plot data
ax[0].plot(t, u, label='input')
ax[1].plot(t_a, y_a, label='$P_a(s)$', color='C1')
ax[1].plot(t_b, y_b, label='$P_b(s)$', color='C2')
ax[1].plot(t_c, y_c, label='$P_c(s)$', color='C3')
ax[1].plot(t_T, y_T, label='$T(s)$', color='C4')
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')
    a.legend(loc='upper right')
fig.tight_layout()
# fig.savefig('figs/control_square_wave_response.pdf')

# %%
# Bode plots
# Calculate freq, magnitude, and phase
w_shared = np.logspace(-3, 3, 1000)
# Note, can use plot=False
mag_a, phase_a, w_a = control.frequency_response(P_a, w_shared)
control.bode_plot(P_a, w_shared, dB=True, deg=True, label=r"$P_a(s)$")
mag_b, phase_b, w_b = control.frequency_response(P_b, w_shared)
control.bode_plot(P_b, w_shared, dB=True, deg=True, label=r"$P_b(s)$")
mag_T, phase_T, w_T = control.frequency_response(T, w_shared)
control.bode_plot(T, w_shared, dB=True, deg=True, label=r"$P_c(s)$")
# fig.savefig('figs/control_Bode_plot_T.pdf')

# %%
# Gain and phase margin
# Compare to control.margin
gm, pm, wpc, wgc = control.margin(L)
gm_dB = 20 * np.log10(gm)  # Convert to dB
print(f'Gain margin is', gm_dB, '(dB) at phase crossover frequency',
      wpc, '(rad/s)')
print(f'Phase margin is', pm, '(deg) at gain crossover frequency',
      wgc, '(deg)\n')

# Compare to control.stability_margins
gm, pm, vm, wpc, wgc, wvm = control.stability_margins(L)
gm_dB = 20 * np.log10(gm)  # Convert to dB
print(f'Gain margin is', gm_dB, '(dB) at phase crossover frequency',
      wpc, '(rad/s)')
print(f'Phase margin is', pm, '(deg) at gain crossover frequency',
      wgc, '(rad/s)')
print(f'Vector margin is', vm, 'at frequency', wvm, '(deg)\n')

# Margins of L_e(s) = 3 / (s**2 + s) / (s + 2)
gm, pm, vm, wpc, wgc, wvm = control.stability_margins(L_e)
gm_dB = 20 * np.log10(gm)  # Convert to dB
print(f'Gain margin is', gm_dB, '(dB) at phase crossover frequency',
      wpc, '(rad/s)')
print(f'Phase margin is', pm, '(deg) at gain crossover frequency',
      wgc, '(rad/s)')
print(f'Vector margin is', vm, 'at frequency', wvm, '(rad/s)')

# %%
# Nyquist plot
response = control.nyquist_response(L, omega=w_shared)
count_L = response.count
print(f'Number of encirclements of the -1 is {count_L}.')

# Plot Nyquist plot of L
fig, ax = plt.subplots()
response.plot()
fig.tight_layout()
plt.show()
# fig.savefig('figs/nyquist.pdf')

# %%
# Plot gang of 4
control.gangof4_plot(P_d, C_PI, omega=w_shared)  # [[S, PS], [CS, T]]
# fig.savefig('figs/control_Gof4_Bode_plot.pdf')


# %%
# Root locus
fig, ax = plt.subplots()
rlist, klist = control.root_locus(L, plot=True)  # deprecated; use root_locus_map()


# %%
# Analysis tools
DC_gain_T = control.dcgain(T)
print(f'The DC gain of T(s) is {DC_gain_T}.')

# %%
plt.show()

# %%
