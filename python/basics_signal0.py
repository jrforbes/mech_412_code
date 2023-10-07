"""Python signal processing basics.

J R Forbes, 2021/12/18
Based on
https://docs.scipy.org/doc/scipy/reference/signal.html
"""

# %%
# Packages
import numpy as np
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
# Firsr-order transfer function, P(s) = 1 / (tau * s + 1)
tau = 1 / 1
P_a = signal.lti([1], [tau, 1])

# Second-order transfer function with zero
# P(s) = (s / a + 1) / (s**2 + 2 * zeta * omega * s + omega**2)
b = 10
omega = 5
zeta = 0.5
P_b = signal.lti([omega**2 / b, omega**2], [1, 2 * zeta * omega, omega**2])

# Mass-spring_damper system
m = 1  # kg, mass
d = 0.05  # N s / m, damper
k = 1  # N / m, spring
# Form state-space matrics.
A = np.array([[0, 1],
              [-k / m, -d / m]])
B = np.array([[0],
              [1 / m]])
C = np.array([[1, 0]])
D = np.array([[0]])
x0 = np.array([0.25, -0.5])  # Initial conditions
P_c = signal.lti(A, B, C, D)


# %%
# Step response
t_a, y_a = signal.step(P_a, T=t)
t_b, y_b = signal.step(P_b, T=t)
t_c, y_c = signal.step(P_c, X0=x0, T=t)

# Plot step response
fig, ax = plt.subplots()
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'$y(t)$ (units)')
# Plot data
ax.plot(t_a, y_a, label='$P_a(s)$', color='C1')
ax.plot(t_b, y_b, label='$P_b(s)$', color='C2')
ax.plot(t_c, y_c, label='$P_c(s)$', color='C3')
ax.legend(loc='upper right')
fig.tight_layout()
# fig.savefig('figs/step_response.pdf')

# %%
# Impulse response
t_a, y_a = signal.impulse(P_a, T=t)
t_b, y_b = signal.impulse(P_b, T=t)

# Plot impulse response
fig, ax = plt.subplots()
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'$y(t)$ (units)')
# Plot data
ax.plot(t_a, y_a, label='$P_a(s)$', color='C1')
ax.plot(t_b, y_b, label='$P_b(s)$', color='C2')
ax.legend(loc='upper right')
fig.tight_layout()
# fig.savefig('figs/impulse_response.pdf')

# %%
# Time-domaine forced response
# Square wave input
u = signal.square(2 * np.pi / 2 * t)

# Forced response of each system
t_a, y_a, x_a = signal.lsim(P_a, u, t)
t_b, y_b, x_b = signal.lsim(P_b, u, t)
t_c, y_c, x_c = signal.lsim(P_c, u, t, x0)

# Plot forced response
fig, ax = plt.subplots(2, 1)   
ax[0].set_ylabel(r'$u(t)$ (units)')
ax[1].set_ylabel(r'$y(t)$ (units)')
# Plot data
ax[0].plot(t, u, label='input')
ax[1].plot(t_a, y_a, label='$P_a(s)$', color='C1')
ax[1].plot(t_b, y_b, label='$P_b(s)$', color='C2')
ax[1].plot(t_c, y_c, label='$P_c(s)$', color='C3')
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')
    a.legend(loc='upper right')
fig.tight_layout()
# fig.savefig('figs/square_wave_response.pdf')

# %%
# Bode plots
# Calculate freq, magnitude, and phase
w_shared = np.logspace(-3, 3, 1000)
w_a, mag_a, phase_a = signal.bode(P_a, w_shared)
w_b, mag_b, phase_b = signal.bode(P_b, w_shared)

# Plot first-order system Bode plot
fig, ax = plt.subplots(2, 1)
# Magnitude plot
ax[0].semilogx(w_a, mag_a)
ax[0].set_xlabel(r'$\omega$ (rad/s)')
ax[0].set_ylabel(r'Magnitude (dB)')  # $|G(\omega)|$
# Phase plot
ax[1].semilogx(w_a, phase_a)
ax[1].set_xlabel(r'$\omega$ (rad/s)')
ax[1].set_ylabel(r'Phase (deg)')  # $\angle G(\omega)$
# fig.savefig('figs/P_a_Bode_plot.pdf')

# Plot Bode plot of both P_a and P_b
fig, ax = plt.subplots(2, 1)
# Magnitude plot
ax[0].semilogx(w_a, mag_a, label=r'$P_a(s)$')
ax[0].semilogx(w_b, mag_b, label=r'$P_b(s)$')
ax[0].set_xlabel(r'$\omega$ (rad/s)')
ax[0].set_ylabel(r'Magnitutde (dB)')
# Phase plot
ax[1].semilogx(w_a, phase_a)
ax[1].semilogx(w_b, phase_b)
ax[1].set_xlabel(r'$\omega$ (rad/s)')
ax[1].set_ylabel(r'Phase (deg)')
ax[0].legend(loc='upper right')
# fig.savefig('figs/Bode_plot_P_a_P_b.pdf')

# %%
# Nyquist plot using freqresp
w_b, fr_b = signal.freqresp(P_b, w_shared)
# Plot Nyquist plot
fig, ax = plt.subplots()
ax.set_xlabel(r'$Re$')
ax.set_ylabel(r'$Im$')
ax.plot(-1, 0, '+', color='C3')
ax.plot(fr_b.real, fr_b.imag, color='C0')
ax.plot(fr_b.real, -fr_b.imag, '--', color='C0')
# fig.savefig('figs/Nyquist_plot_P_b.pdf')

# %%
plt.show()
