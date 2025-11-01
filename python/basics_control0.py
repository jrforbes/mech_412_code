"""Python control basics.

J R Forbes, 2021/12/18
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
# Functions


def nyq(G, wmin, wmax):
    """Plot nyquist plot, output if stable or not."""
    # Frequency
    w_shared = np.logspace(wmin, wmax, 1000)

    # Call control.nyquist_response to get the count of the -1 point
    response = control.nyquist_response(G, omega=w_shared)
    count_G = response.count
    
    # Use control.frequency_response to extract mag and phase information
    mag_G, phase_G, _ = control.frequency_response(G, w_shared)
    Re_G = mag_G * np.cos(phase_G)
    Im_G = mag_G * np.sin(phase_G)

    # Plot Nyquist plot
    fig, ax = plt.subplots()
    ax.set_xlabel(r'Real axis')
    ax.set_ylabel(r'Imaginary axis')
    ax.plot(-1, 0, '+', color='C3')  # -1 point
    ax.plot(Re_G[0], Im_G[0], 'o', color='C0')  # Starting point
    ax.plot(Re_G, Im_G, color='C0')
    ax.plot(Re_G, -Im_G, '--', color='C0')

    return count_G


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
# Convert to ss
P_b_ss = control.tf2ss(P_b_num, P_b_den)
A_b, B_b, C_b, D_b = P_b_ss.A, P_b_ss.B, P_b_ss.C, P_b_ss.D

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
# Convert to tf
P_c_tf = control.ss2tf(A, B, C, D)
P_c_tf_num = np.array(P_c_tf.num).ravel()
P_c_tf_den = np.array(P_c_tf.den).ravel()

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
plt.show()
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
plt.show()
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
plt.show()
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
plt.show()
# fig.savefig('figs/control_square_wave_response.pdf')

# %%
# Bode plots
# Calculate freq, magnitude, and phase
w_shared = np.logspace(-3, 3, 1000)
mag_a, phase_a, w_a = control.frequency_response(P_a, w_shared)
mag_b, phase_b, w_b = control.frequency_response(P_b, w_shared)
mag_T, phase_T, w_T = control.frequency_response(T, w_shared)

# Convert to dB and deg.
mag_a_dB = 20 * np.log10(mag_a)
phase_a_deg = phase_a / np.pi * 180
mag_b_dB = 20 * np.log10(mag_b)
phase_b_deg = phase_b / np.pi * 180
mag_T_dB = 20 * np.log10(mag_T)
phase_T_deg = phase_T / np.pi * 180

# Plot first-order system Bode plot
fig, ax = plt.subplots(2, 1)
# Magnitude plot
ax[0].semilogx(w_a, mag_a_dB)
ax[0].set_xlabel(r'$\omega$ (rad/s)')
ax[0].set_ylabel(r'Magnitude (dB)')
# Phase plot
ax[1].semilogx(w_a, phase_a_deg)
ax[1].set_xlabel(r'$\omega$ (rad/s)')
ax[1].set_ylabel(r'Phase (deg)')
# fig.savefig('figs/control_P_a_Bode_plot.pdf')

# Plot Bode plot of both P_a, P_b, and T
fig, ax = plt.subplots(2, 1)
# Magnitude plot
ax[0].semilogx(w_a, mag_a_dB, label=r'$P_a(s)$')
ax[0].semilogx(w_b, mag_b_dB, label=r'$P_b(s)$')
ax[0].semilogx(w_T, mag_T_dB, label=r'$T(s)$')
ax[0].set_xlabel(r'$\omega$ (rad/s)')
ax[0].set_ylabel(r'Magnitude (dB)')
# Phase plot
ax[1].semilogx(w_a, phase_a_deg)
ax[1].semilogx(w_b, phase_b_deg)
ax[1].semilogx(w_T, phase_T_deg)
ax[1].set_xlabel(r'$\omega$ (rad/s)')
ax[1].set_ylabel(r'Phase (deg)')
ax[0].legend(loc='upper right')
# fig.savefig('figs/control_Bode_plot_P_a_P_b_T.pdf')

# %%
# Gain and phase margin
# From https://jckantor.github.io/CBE30338/05.03-Creating-Bode-Plots.html
mag_L, phase_L, w_L = control.frequency_response(L, w_shared)
mag_L_dB = 20 * np.log10(mag_L)  # Convert to dB
phase_L_deg = phase_L / np.pi * 180

# Find the phase crossover frequency and gain margin
omega_pc = np.interp(-180.0, np.flipud(phase_L_deg), np.flipud(w_L))
gain_margin = -np.interp(omega_pc, w_L, mag_L_dB)
print(f'Gain margin is', gain_margin, '(dB) at phase crossover frequency',
      omega_pc, '(rad/s)')

# Find the gain crossover frequency and phase margin
omega_gc = np.interp(0, np.flipud(mag_L_dB), np.flipud(w_L))
phase_at_omega_gc = np.interp(omega_gc, w_L, phase_L_deg)
phase_margin = 180 + phase_at_omega_gc
print(f'Phase margin is', phase_margin, '(deg) at gain crossover frequency',
      omega_gc, '(rad/s)\n')

# Compare to control.margin
gm, pm, wpc, wgc = control.margin(L)
gm_dB = 20 * np.log10(gm)  # Convert to dB
print(f'Gain margin is', gm_dB, '(dB) at phase crossover frequency',
      wpc, '(rad/s)')
print(f'Phase margin is', pm, '(deg) at gain crossover frequency',
      wgc, '(rad/s))\n')

# Compare to control.stability_margins
gm, pm, vm, wpc, wgc, wvm = control.stability_margins(L)
gm_dB = 20 * np.log10(gm)  # Convert to dB
print(f'Gain margin is', gm_dB, '(dB) at phase crossover frequency',
      wpc, '(rad/s)')
print(f'Phase margin is', pm, '(deg) at gain crossover frequency',
      wgc, '(rad/s)')
print(f'Vector margin is', vm, 'at frequency', wvm, '(rad/s)\n')

# Plot open-loop system Bode plot
fig, ax = plt.subplots(2, 1)
# Magnitude plot
ax[0].semilogx(w_L, mag_L_dB)
ax[0].semilogx(omega_pc, -gain_margin, 'o', color='C3')
ax[0].set_xlabel(r'$\omega$ (rad/s)')
ax[0].set_ylabel(r'$|L(j\omega)|$ (dB)')
# Phase plot
ax[1].semilogx(w_L, phase_L_deg)
ax[1].semilogx(omega_gc, phase_at_omega_gc, 'o', color='C3')
ax[1].set_xlabel(r'$\omega$ (rad/s)')
ax[1].set_ylabel(r'$\angle L(j\omega)$ (deg)')
# fig.savefig('figs/control_L_Bode_plot.pdf')

# Margins of L_e(s) = 3 / (s**2 + s) / (s + 2)
gm, pm, vm, wpc, wgc, wvm = control.stability_margins(L_e)
gm_dB = 20 * np.log10(gm)  # Convert to dB
print(f'Gain margin is', gm_dB, '(dB) at phase crossover frequency',
      wpc, '(deg)')
print(f'Phase margin is', pm, '(deg) at gain crossover frequency',
      wgc, '(deg)')
print(f'Vector margin is', vm, 'at frequency', wvm, '(deg)')

# %%
# Nyquist plot
count_b_nyq = nyq(P_b, -3, 3)
# fig.savefig('figs/control_Nyquist_plot_P_b.pdf')

response = control.nyquist_response(L, omega=w_shared)
count_L = response.count
print(f'Number of encirclements of the -1 is {count_L}.')

# Plot Nyquist plot of L
fig, ax = plt.subplots()
response.plot(ax=ax)
ax.set_title('Nyquist Plot')
fig.tight_layout()
plt.show()
# fig.savefig('figs/nyquist.pdf')


# %%
# Plot gang of 4
control.gangof4_plot(P_d, C_PI, omega=w_shared)  # [[S, PS], [CS, T]]

# Compare to custom gang of 4 code
mag_T, phase_T, w_T = control.frequency_response(T, w_shared)
mag_S, phase_S, w_S = control.frequency_response(S, w_shared)
mag_PS, phase_PS, w_PS = control.frequency_response(P_d * S, w_shared)
mag_CS, phase_CS, w_CS = control.frequency_response(C_PI * S, w_shared)

mag_T_dB = 20 * np.log10(mag_T)
phase_T_deg = phase_T / np.pi * 180
mag_S_dB = 20 * np.log10(mag_S)
phase_S_deg = phase_S / np.pi * 180
mag_PS_dB = 20 * np.log10(mag_PS)
phase_PS_deg = phase_PS / np.pi * 180
mag_CS_dB = 20 * np.log10(mag_CS)
phase_CS_deg = phase_CS / np.pi * 180

fig, ax = plt.subplots(2, 2)
# Magnitude plot of S
ax[0, 0].semilogx(w_S, mag_S_dB)
ax[0, 0].set_xlabel(r'$\omega$ (rad/s)')
ax[0, 0].set_ylabel(r'$|S(j\omega)|$ (dB)')
# Magnitude plot of P * S
ax[0, 1].semilogx(w_PS, mag_PS_dB)
ax[0, 1].set_xlabel(r'$\omega$ (rad/s)')
ax[0, 1].set_ylabel(r'$|P(j\omega)S(j\omega)|$ (dB)')
# Magnitude plot of C * S
ax[1, 0].semilogx(w_CS, mag_CS_dB)
ax[1, 0].set_xlabel(r'$\omega$ (rad/s)')
ax[1, 0].set_ylabel(r'$|C(j\omega)S(j\omega)|$ (dB)')
# Magnitude plot of T
ax[1, 1].semilogx(w_T, mag_T_dB)
ax[1, 1].set_xlabel(r'$\omega$ (rad/s)')
ax[1, 1].set_ylabel(r'$|T(j\omega)|$ (dB)')
fig.tight_layout()
# fig.savefig('figs/control_Gof4_Bode_plot.pdf')

# %%
# Analysis tools
DC_gain_T = control.dcgain(T)
print(f'The DC gain of T(s) is {DC_gain_T}.')

# %%
# Root locus
fig, ax = plt.subplots()
rlist, klist = control.root_locus(L, plot=True)  # deprecated; use root_locus_map()


# %%
# Plot show
plt.show()

# %%
