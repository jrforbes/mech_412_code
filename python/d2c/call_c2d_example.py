"""
Continuous-time to discrete time, discrete-time to continuous-time, examples.

This file calles the d2c module that has Forbes' custom c2d function in it.

J R Forbes, 2022/01/18
"""

# %%
# Libraries
import numpy as np
import control
from matplotlib import pyplot as plt
import pathlib
import d2c


# %% 
# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')
path = pathlib.Path('figs')
path.mkdir(exist_ok=True)

# %%
# Laplace s
s = control.tf('s')

# %%
# Continuous and discrete first-order system
tau = 1 / 10
T = 0.1  # s
P = 1 / (tau * s + 1)

# P.sample is the built in control package c2d function. In general, use it.
Pd = P.sample(T, method='zoh')  # method='bilinear'
Pd_num = np.array(Pd.num).ravel()
Pd_den = np.array(Pd.den).ravel()

print('P_d numerator coefficents are', Pd_num)
print('P_d denominator coefficents are', Pd_den, '\n')

# Go back from discrete time to continuous time
Pc = d2c.d2c(Pd)
print(Pc)

# %%
# Step response
# time
dt = 1e-2
t_start = 0
t_end = 0.6
t = np.arange(t_start, t_end, dt)
t, y = control.step_response(P, t)

td = np.arange(t_start, t_end, T)
td, yd = control.step_response(Pd, td)

# Plot step response
fig, ax = plt.subplots()
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'$y(t)$ (units)')
# Plot data
ax.plot(t, y, label='continuous time', color='C0')
plt.step(td, yd, where='post', label='discrete time', color='C1')
ax.legend(loc='lower right')
fig.tight_layout()
# fig.savefig(path.joinpath('first_order_sys_step_response.pdf'))

# %%
# Continuous and discrete second-order system
omega = 10  # rad/s
zeta = 0.25
tau = 1 / 5
T = 0.05
P = omega ** 2 * (tau * s + 1) / (s ** 2 + 2 * zeta * omega * s + omega ** 2)

# P.sample is the built in control package c2d function. In general, use it.
Pd = P.sample(T, method='zoh')  # method='bilinear'
Pd_num = np.array(Pd.num).ravel()
Pd_den = np.array(Pd.den).ravel()

print('P_d numerator coefficents are', Pd_num)
print('P_d denominator coefficents are', Pd_den)

# Go back from discrete time to continuous time
Pc = d2c.d2c(Pd)
print(Pc)

# %%
# Step response of each system
# time
dt = 1e-2
t_start = 0
t_end = 2
t = np.arange(t_start, t_end, dt)
t, y = control.step_response(P, t)

td = np.arange(t_start, t_end, T)
td, yd = control.step_response(Pd, td)

# Plot step response
fig, ax = plt.subplots()
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'$y(t)$ (units)')
# Plot data
ax.plot(t, y, label='continuous time', color='C0')
plt.step(td, yd, where='post', label='discrete time', color='C1')
ax.legend(loc='lower right')
fig.tight_layout()
# fig.savefig(path.joinpath('second_order_sys_step_response.pdf'))

# %%
# Plot show
plt.show()