"""SISO robust performance tools.

James Forbes
2023/09/26
"""
# %%
# Libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import control


# %%
# Functions

rps2Hz = lambda w: w / 2 / np.pi
Hz2rps = lambda w: w * 2 * np.pi

def circle(x_c, y_c, r):
    """Plot a circle."""
    # Theta, x, an y
    th = np.linspace(0, 2 * np.pi, 100)
    x = x_c + np.cos(th) * r
    y = y_c + np.sin(th) * r
    return x, y


def robust_nyq(P, P_off_nom, W2, wmin, wmax, N_w):
    """Plot nyquist plot, output if stable or not."""
    # Frequencies
    w_shared = np.logspace(wmin, wmax, N_w)

    # Call control.nyquist to get the count of the -1 point
    response = control.nyquist_response(P, omega=w_shared)
    count_P = response.count

    # Set Nyquist plot up
    fig, ax = plt.subplots()
    ax.set_xlabel(r'Real axis')
    ax.set_ylabel(r'Imaginary axis')
    ax.plot(-1, 0, '+', color='C3')

    # Plot uncertain systems
    for k in range(len(P_off_nom)):
        # Use control.frequency_response to extract mag and phase information
        mag_P_off_nom, phase_P_off_nom, _ = control.frequency_response(P_off_nom[k], w_shared)
        Re_P_off_nom = mag_P_off_nom * np.cos(phase_P_off_nom)
        Im_P_off_nom = mag_P_off_nom * np.sin(phase_P_off_nom)

        # Plot Nyquist plot
        ax.plot(Re_P_off_nom, Im_P_off_nom, color='C0', linewidth=0.75)

    # Plot nominal system
    mag_P, phase_P, _ = control.frequency_response(P, w_shared)
    Re_P = mag_P * np.cos(phase_P)
    Im_P = mag_P * np.sin(phase_P)

    # Plot Nyquist plot
    ax.plot(Re_P, Im_P, '-', color='C3')

    # Plot circles
    w_circle = np.geomspace(10**wmin, 10**wmax, 50)
    mag_P_W2, _, _ = control.frequency_response(P * W2, w_circle)
    mag_P, phase_P, _ = control.frequency_response(P, w_circle)
    Re_P = mag_P * np.cos(phase_P)
    Im_P = mag_P * np.sin(phase_P)
    for k in range(w_circle.size):
        x, y = circle(Re_P[k], Im_P[k], mag_P_W2[k])
        ax.plot(x, y, color='C1', linewidth=0.75, alpha=0.75)

    return count_P, fig, ax


def bode_mag_W1_W2(W1, W2, w_d_h, w_n_l, w_shared, **kwargs):
    """Plot W1 and W2."""
    # Frequency response
    mag_W1, _, _ = control.frequency_response(W1, w_shared)
    mag_W1_dB = 20 * np.log10(mag_W1)

    mag_W2, _, _ = control.frequency_response(W2, w_shared)
    mag_W2_dB = 20 * np.log10(mag_W2)

    # Plot
    fig, ax = plt.subplots()
    ax.set_ylabel(r'Magnitude (dB)')
    if kwargs.get("Hz", False):
        # Plot in Hz
        ax.set_xlabel(r'$\omega$ (Hz)')
        # Magnitude plot
        ax.semilogx(rps2Hz(w_d_h), 0, 'd', color='C4', label=r'$\omega_d$')
        ax.semilogx(rps2Hz(w_n_l), 0, 'd', color='C6', label=r'$\omega_n$')
        ax.semilogx(rps2Hz(w_shared), mag_W1_dB, color='gold', label=r'$|W_1(j\omega)|$')
        ax.semilogx(rps2Hz(w_shared), mag_W2_dB, '-', color='seagreen', label=r'$|W_2(j \omega)|$')
    else:
        # Plot in rad/s
        ax.set_xlabel(r'$\omega$ (rad/s)')
        # Magnitude plot
        ax.semilogx(w_d_h, 0, 'd', color='C4', label=r'$\omega_d$')
        ax.semilogx(w_n_l, 0, 'd', color='C6', label=r'$\omega_n$')
        ax.semilogx(w_shared, mag_W1_dB, color='gold', label=r'$|W_1(j\omega)|$')
        ax.semilogx(w_shared, mag_W2_dB, '-', color='seagreen', label=r'$|W_2(j \omega)|$')

    return fig, ax


def bode_mag_W1_inv_W2_inv(W1, W2, gamma_r, w_r_h, w_d_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared, **kwargs):
    """Plot W1 and W2."""
    # Noise bounds
    w_r = np.logspace(w_shared_low, np.log10(w_r_h), 100)
    w_n = np.logspace(np.log10(w_n_l), w_shared_high, 100)

    gamma_r_dB = 20 * np.log10(gamma_r) * np.ones(w_r.shape[0],)
    gamma_n_dB = 20 * np.log10(gamma_n) * np.ones(w_n.shape[0],)

    # Frequency response
    W1_inv = 1 / W1
    mag_W1_inv, _, _ = control.frequency_response(W1_inv, w_shared)
    mag_W1_inv_dB = 20 * np.log10(mag_W1_inv)

    W2_inv = 1 / W2
    mag_W2_inv, _, _ = control.frequency_response(W2_inv, w_shared)
    mag_W2_inv_dB = 20 * np.log10(mag_W2_inv)

    # Plot
    fig, ax = plt.subplots()
    ax.set_ylabel(r'Magnitude (dB)')
    if kwargs.get("Hz", False):
        # Plot in Hz
        ax.set_xlabel(r'$\omega$ (Hz)')
        # Magnitude plot
        ax.semilogx(rps2Hz(w_shared), 6 * np.ones(w_shared.shape[0],), '--', color='silver')
        ax.semilogx(rps2Hz(w_d_h), 0, 'd', color='C4', label=r'$\omega_d$')
        ax.semilogx(rps2Hz(w_r), gamma_r_dB, color='C3', label=r'$\gamma_r$')
        ax.semilogx(rps2Hz(w_n), gamma_n_dB, color='C6', label=r'$\gamma_n$')
        ax.semilogx(rps2Hz(w_shared), mag_W1_inv_dB, color='gold', label=r'$|W_1(j\omega)^{-1}|$')
        ax.semilogx(rps2Hz(w_shared), mag_W2_inv_dB, '-', color='seagreen', label=r'$|W_2(j \omega)^{-1}|$')
    else:
        # Plot in rad/s
        ax.set_xlabel(r'$\omega$ (rad/s)')
        # Magnitude plot
        ax.semilogx(w_shared, 6 * np.ones(w_shared.shape[0],), '--', color='silver')
        ax.semilogx(w_d_h, 0, 'd', color='C4', label=r'$\omega_d$')
        ax.semilogx(w_r, gamma_r_dB, color='C3', label=r'$\gamma_r$')
        ax.semilogx(w_n, gamma_n_dB, color='C6', label=r'$\gamma_n$')
        ax.semilogx(w_shared, mag_W1_inv_dB, color='gold', label=r'$|W_1(j\omega)^{-1}|$')
        ax.semilogx(w_shared, mag_W2_inv_dB, '-', color='seagreen', label=r'$|W_2(j \omega)^{-1}|$')

    return fig, ax


def bode_mag_L(P, C, gamma_r, w_r_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared, **kwargs):
    """Plot L."""
    # Noise bounds
    w_r = np.logspace(w_shared_low, np.log10(w_r_h), 100)
    w_n = np.logspace(np.log10(w_n_l), w_shared_high, 100)

    # Gain bounds
    gamma_n_dB = 20 * np.log10(gamma_n) * np.ones(w_n.shape[0],)
    gamma_r_inverse_dB = 20 * np.log10(1 / gamma_r) * np.ones(w_r.shape[0],)

    # L
    L = control.minreal(P * C)

    # Frequency response
    mag_L, phase_L, w_L = control.frequency_response(L, w_shared)
    mag_L_dB = 20 * np.log10(mag_L)
    phase_L_deg = phase_L / np.pi * 180

    # Plot open-loop TF L
    fig_L, ax_L = plt.subplots()
    ax_L.set_ylabel(r'Magnitude (dB)')
    if kwargs.get("Hz", False):
        # Plot in Hz
        ax_L.set_xlabel(r'$\omega$ (Hz)')
        # Magnitude plot
        ax_L.semilogx(rps2Hz(w_L), mag_L_dB, label=r'$|L(j\omega)|$')
        ax_L.semilogx(rps2Hz(w_r), gamma_r_inverse_dB, '-', color='C3', label=r'$1/\gamma_r$')
        ax_L.semilogx(rps2Hz(w_n), gamma_n_dB, color='C6', label=r'$\gamma_n$')
    else:
        # Plot in rad/s
        ax_L.set_xlabel(r'$\omega$ (rad/s)')
        # Magnitude plot
        ax_L.semilogx(w_L, mag_L_dB, label=r'$|L(j\omega)|$')
        ax_L.semilogx(w_r, gamma_r_inverse_dB, '-', color='C3', label=r'$1/\gamma_r$')
        ax_L.semilogx(w_n, gamma_n_dB, color='C6', label=r'$\gamma_n$')

    return fig_L, ax_L


def bode_mag_L_W1_W2_inv(P, C, W1, W2, gamma_r, w_r_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared, **kwargs):
    """Plot L."""
    # Noise bounds
    w_r = np.logspace(w_shared_low, np.log10(w_r_h), 100)
    w_n = np.logspace(np.log10(w_n_l), w_shared_high, 100)

    # Gain bounds
    gamma_n_dB = 20 * np.log10(gamma_n) * np.ones(w_n.shape[0],)
    gamma_r_inverse_dB = 20 * np.log10(1 / gamma_r) * np.ones(w_r.shape[0],)

    # W1 and W2 inverse
    w_mid = np.floor((w_shared_low + w_shared_high) / 2)
    mag_W1_low_freq, _, w_W1_low_freq = control.frequency_response(W1, np.logspace(w_shared_low, w_mid, 100))
    mag_W1_low_freq_dB = 20 * np.log10(mag_W1_low_freq)  # Convert to dB

    mag_W2_inv_high_freq, _, w_W2_inv_high_freq = control.frequency_response(1 / W2, np.logspace(w_mid, w_shared_high, 100))
    mag_W2_inv_high_freq_dB = 20 * np.log10(mag_W2_inv_high_freq) # Convert to dB

    # L
    L = control.minreal(P * C)

    # Frequency response
    mag_L, phase_L, w_L = control.frequency_response(L, w_shared)
    mag_L_dB = 20 * np.log10(mag_L)
    phase_L_deg = phase_L / np.pi * 180

    # Plot open-loop TF L
    fig_L, ax_L = plt.subplots()
    ax_L.set_ylabel(r'Magnitude (dB)')
    if kwargs.get("Hz", False):
        # Plot in Hz
        ax_L.set_xlabel(r'$\omega$ (Hz)')
        # Magnitude plot
        ax_L.semilogx(rps2Hz(w_L), mag_L_dB, label=r'$|L(j\omega)|$')
        ax_L.semilogx(rps2Hz(w_W1_low_freq), mag_W1_low_freq_dB, color='gold', label=r'$|W_1(j\omega)|$')
        ax_L.semilogx(rps2Hz(w_W2_inv_high_freq), mag_W2_inv_high_freq_dB, color='seagreen', label=r'$|W_2(j\omega)^{-1}|$')
        ax_L.semilogx(rps2Hz(w_r), gamma_r_inverse_dB, '-', color='C3', label=r'$1/\gamma_r$')
        ax_L.semilogx(rps2Hz(w_n), gamma_n_dB, color='C6', label=r'$\gamma_n$')
    else:
        # Plot in rad/s
        ax_L.set_xlabel(r'$\omega$ (rad/s)')
        # Magnitude plot
        ax_L.semilogx(w_L, mag_L_dB, label=r'$|L(j\omega)|$')
        ax_L.semilogx(w_W1_low_freq, mag_W1_low_freq_dB, color='gold', label=r'$|W_1(j\omega)|$')
        ax_L.semilogx(w_W2_inv_high_freq, mag_W2_inv_high_freq_dB, color='seagreen', label=r'$|W_2(j\omega)^{-1}|$')
        ax_L.semilogx(w_r, gamma_r_inverse_dB, '-', color='C3', label=r'$1/\gamma_r$')
        ax_L.semilogx(w_n, gamma_n_dB, color='C6', label=r'$\gamma_n$')
    
    return fig_L, ax_L


def bode_mag_rob_perf(P, C, W1, W2, w_shared, **kwargs):
    """Plot robust performance condition."""
    # S and T
    T = control.feedback(P * C, 1, -1)
    S = control.feedback(1, P * C, -1)

    # Frequency response
    mag_S_W1, phase_S_W1, w_S_W1 = control.frequency_response(S * W1, w_shared)
    mag_T_W2, phase_T_W2, w_T_W2 = control.frequency_response(T * W2, w_shared)

    # Combined
    mag_S_W1_plus_T_W2 = mag_S_W1 + mag_T_W2
    mag_S_W1_plus_T_W2_dB = 20 * np.log10(mag_S_W1_plus_T_W2)

    # Max value on robust performance curve
    max_mag_S_W1_plus_T_W2_dB = np.max(mag_S_W1_plus_T_W2_dB)
    N_max_mag_S_W1_plus_T_W2_dB = np.argmax(mag_S_W1_plus_T_W2_dB)

    # Plot robust performance
    fig_RP, ax_RP = plt.subplots()
    ax_RP.set_ylabel(r'Magnitude (dB)')
    if kwargs.get("Hz", False):
        # Plot in Hz
        ax_RP.set_xlabel(r'$\omega$ (Hz)')
        ax_RP.semilogx(rps2Hz(w_S_W1), mag_S_W1_plus_T_W2_dB, '-', color='C0', label=r'$|S(j \omega) W_1(j \omega)| + |T(j \omega) W_2(j \omega)|$')
        ax_RP.semilogx(rps2Hz(w_shared), np.zeros(w_shared.shape[0],), '--', color='C3')
        ax_RP.semilogx(rps2Hz(w_shared[N_max_mag_S_W1_plus_T_W2_dB]), max_mag_S_W1_plus_T_W2_dB, 'd', color='black', label=r'max value')
    else:
        # Plot in rad/s
        ax_RP.set_xlabel(r'$\omega$ (rad/s)')
        ax_RP.semilogx(w_S_W1, mag_S_W1_plus_T_W2_dB, '-', color='C0', label=r'$|S(j \omega) W_1(j \omega)| + |T(j \omega) W_2(j \omega)|$')
        ax_RP.semilogx(w_shared, np.zeros(w_shared.shape[0],), '--', color='C3')
        ax_RP.semilogx(w_shared[N_max_mag_S_W1_plus_T_W2_dB], max_mag_S_W1_plus_T_W2_dB, 'd', color='black', label=r'max value')
    
    # ax_RP.legend(loc='best')

    return fig_RP, ax_RP


def bode_mag_rob_perf_RD(P, C, W1, W2, w_shared, **kwargs):
    """Plot robust performance with return difference, |1 + L(j omega)|."""
    # L
    L = control.minreal(P * C)

    # Frequency response
    mag_L, phase_L, w_L = control.frequency_response(L, w_shared)
    mag_W1, phase_W1, w_W1 = control.frequency_response(W1, w_shared)
    mag_L_W2, phase_L_W2, w_L_W2 = control.frequency_response(L * W2, w_shared)
    mag_W1_plus_L_W2 = mag_W1 + mag_L_W2
    mag_RD, phase_RD, w_RD = control.frequency_response(1 + L, w_shared)

    mag_W1_plus_L_W2_dB = 20 * np.log10(mag_W1_plus_L_W2)
    mag_RD_dB = 20 * np.log10(mag_RD)

    # Min value on return difference curve
    min_mag_RD_dB = np.min(mag_RD_dB)
    N_min_mag_RD_dB = np.argmin(mag_RD_dB)

    # Plot robust performance using return difference
    fig_RP_RD, ax_RP_RD = plt.subplots()
    ax_RP_RD.set_ylabel(r'Magnitude (dB)')
    if kwargs.get("Hz", False):
        # Plot in Hz
        ax_RP_RD.set_xlabel(r'$\omega$ (Hz)')
        ax_RP_RD.semilogx(rps2Hz(w_RD), mag_RD_dB, '-', color='C0', label=r'$|1 + L(j \omega)|$')
        ax_RP_RD.semilogx(rps2Hz(w_L_W2), mag_W1_plus_L_W2_dB, '--', color='C3', label=r'$|W_1(j \omega)| + |L(j \omega) W_2(j \omega)|$')
        ax_RP_RD.semilogx(rps2Hz(w_shared[N_min_mag_RD_dB]), min_mag_RD_dB, 'd', color='black', label=r'min value')
    else:
        # Plot in rad/s
        ax_RP_RD.set_xlabel(r'$\omega$ (rad/s)')
        ax_RP_RD.semilogx(w_RD, mag_RD_dB, '-', color='C0', label=r'$|1 + L(j \omega)|$')
        ax_RP_RD.semilogx(w_L_W2, mag_W1_plus_L_W2_dB, '--', color='C3', label=r'$|W_1(j \omega)| + |L(j \omega) W_2(j \omega)|$')
        ax_RP_RD.semilogx(w_shared[N_min_mag_RD_dB], min_mag_RD_dB, 'd', color='black', label=r'min value')
    
    return fig_RP_RD, ax_RP_RD


def bode_mag_S_T(P, C, gamma_r, w_r_h, w_d_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared, **kwargs):
    """Plot S and T."""
    # Noise bounds
    w_r = np.logspace(w_shared_low, np.log10(w_r_h), 100)
    w_n = np.logspace(np.log10(w_n_l), w_shared_high, 100)

    gamma_r_dB = 20 * np.log10(gamma_r) * np.ones(w_r.shape[0],)
    gamma_n_dB = 20 * np.log10(gamma_n) * np.ones(w_n.shape[0],)

    # S and T
    T = control.feedback(P * C, 1, -1)
    S = control.feedback(1, P * C, -1)

    # Frequency response
    mag_T, phase_T, w_T = control.frequency_response(T, w_shared)
    mag_S, phase_S, w_S = control.frequency_response(S, w_shared)

    mag_T_dB = 20 * np.log10(mag_T)
    phase_T_deg = phase_T / np.pi * 180
    mag_S_dB = 20 * np.log10(mag_S)
    phase_S_deg = phase_S / np.pi * 180

    # Plot S and T
    fig_S_T, ax_S_T = plt.subplots()
    ax_S_T.set_ylabel(r'Magnitude (dB)')
    if kwargs.get("Hz", False):
        # Plot in Hz
        ax_S_T.set_xlabel(r'$\omega$ (Hz)')
        # Magnitude plot
        ax_S_T.semilogx(rps2Hz(w_shared), 6 * np.ones(w_shared.shape[0],), '--', color='silver')
        ax_S_T.semilogx(rps2Hz(w_d_h), 0, 'd', color='C4', label=r'$\omega_d$')
        ax_S_T.semilogx(rps2Hz(w_S), mag_S_dB, color='C1', label=r'$|S(j\omega)|$')
        ax_S_T.semilogx(rps2Hz(w_T), mag_T_dB, color='C9', label=r'$|T(j\omega)|$')
        ax_S_T.semilogx(rps2Hz(w_r), gamma_r_dB, color='C3', label=r'$\gamma_r$')
        ax_S_T.semilogx(rps2Hz(w_n), gamma_n_dB, color='C6', label=r'$\gamma_n$')
    else:
        # Plot in rad/s
        ax_S_T.set_xlabel(r'$\omega$ (rad/s)')
        # Magnitude plot
        ax_S_T.semilogx(w_shared, 6 * np.ones(w_shared.shape[0],), '--', color='silver')
        ax_S_T.semilogx(w_d_h, 0, 'd', color='C4', label=r'$\omega_d$')
        ax_S_T.semilogx(w_S, mag_S_dB, color='C1', label=r'$|S(j\omega)|$')
        ax_S_T.semilogx(w_T, mag_T_dB, color='C9', label=r'$|T(j\omega)|$')
        ax_S_T.semilogx(w_r, gamma_r_dB, color='C3', label=r'$\gamma_r$')
        ax_S_T.semilogx(w_n, gamma_n_dB, color='C6', label=r'$\gamma_n$')

    return fig_S_T, ax_S_T


def bode_mag_S_T_W1_inv_W2_inv(P, C, W1, W2, gamma_r, w_r_h, gamma_d, w_d_h, gamma_n, w_n_l, gamma_u, w_u_l, w_shared_low, w_shared_high, w_shared, **kwargs):
    """Plot S and T with W1^{-1} and W2^{-1}."""
    # Noise bounds
    w_r = np.logspace(w_shared_low, np.log10(w_r_h), 100)
    w_n = np.logspace(np.log10(w_n_l), w_shared_high, 100)

    gamma_r_dB = 20 * np.log10(gamma_r) * np.ones(w_r.shape[0],)
    gamma_n_dB = 20 * np.log10(gamma_n) * np.ones(w_n.shape[0],)

    # S and T
    T = control.feedback(P * C, 1, -1)
    S = control.feedback(1, P * C, -1)

    # Frequency response
    mag_T, phase_T, w_T = control.frequency_response(T, w_shared)
    mag_S, phase_S, w_S = control.frequency_response(S, w_shared)

    mag_T_dB = 20 * np.log10(mag_T)
    phase_T_deg = phase_T / np.pi * 180
    mag_S_dB = 20 * np.log10(mag_S)
    phase_S_deg = phase_S / np.pi * 180

    # Inverse weights and their frequency response
    W1_inv = 1 / W1
    W2_inv = 1 / W2
    mag_W1_inv, phase_W1_inv, w_W1_inv = control.frequency_response(W1_inv, w_shared)
    mag_W2_inv, phase_W2_inv, w_W2_inv = control.frequency_response(W2_inv, w_shared)

    mag_W1_inv_dB = 20 * np.log10(mag_W1_inv)
    phase_W1_inv_deg = phase_W1_inv / np.pi * 180
    mag_W2_inv_dB = 20 * np.log10(mag_W2_inv)
    phase_W2_inv_deg = phase_W2_inv / np.pi * 180

    # Plot S and T
    fig_S_T, ax_S_T = plt.subplots()
    ax_S_T.set_ylabel(r'Magnitude (dB)')
    if kwargs.get("Hz", False):
        # Plot in Hz
        ax_S_T.set_xlabel(r'$\omega$ (Hz)')
        # Magnitude plot
        ax_S_T.semilogx(rps2Hz(w_shared), 6 * np.ones(w_shared.shape[0],), '--', color='silver')
        ax_S_T.semilogx(rps2Hz(w_d_h), 0, 'd', color='C4', label=r'$\omega_d$')
        ax_S_T.semilogx(rps2Hz(w_W1_inv), mag_W1_inv_dB, color='gold', label=r'$|W_1(j\omega)^{-1}|$')
        ax_S_T.semilogx(rps2Hz(w_W2_inv), mag_W2_inv_dB, color='seagreen', label=r'$|W_2(j\omega)^{-1}|$')
        ax_S_T.semilogx(rps2Hz(w_S), mag_S_dB, color='C1', label=r'$|S(j\omega)|$')
        ax_S_T.semilogx(rps2Hz(w_T), mag_T_dB, color='C9', label=r'$|T(j\omega)|$')
        ax_S_T.semilogx(rps2Hz(w_r), gamma_r_dB, color='C3', label=r'$\gamma_r$')
        ax_S_T.semilogx(rps2Hz(w_n), gamma_n_dB, color='C6', label=r'$\gamma_n$')
    else:
        # Plot in Hz
        ax_S_T.set_xlabel(r'$\omega$ (rad/s)')
        # Magnitude plot
        ax_S_T.semilogx(w_shared, 6 * np.ones(w_shared.shape[0],), '--', color='silver')
        ax_S_T.semilogx(w_d_h, 0, 'd', color='C4', label=r'$\omega_d$')
        ax_S_T.semilogx(w_W1_inv, mag_W1_inv_dB, color='gold', label=r'$|W_1(j\omega)^{-1}|$')
        ax_S_T.semilogx(w_W2_inv, mag_W2_inv_dB, color='seagreen', label=r'$|W_2(j\omega)^{-1}|$')
        ax_S_T.semilogx(w_S, mag_S_dB, color='C1', label=r'$|S(j\omega)|$')
        ax_S_T.semilogx(w_T, mag_T_dB, color='C9', label=r'$|T(j\omega)|$')
        ax_S_T.semilogx(w_r, gamma_r_dB, color='C3', label=r'$\gamma_r$')
        ax_S_T.semilogx(w_n, gamma_n_dB, color='C6', label=r'$\gamma_n$')

    return fig_S_T, ax_S_T


def bode_mag_L_P(P, C, gamma_d, w_d_h, gamma_u, w_u_l, w_shared, **kwargs):
    """Plot L and P to assess disturbance rejection and noise in output."""
    # L
    L = control.minreal(P * C)

    # Frequency response
    mag_P, phase_P, w_P = control.frequency_response(P, w_shared)
    mag_P_dB = 20 * np.log10(mag_P)
    phase_P_deg = phase_P / np.pi * 180

    mag_L, phase_L, w_L = control.frequency_response(L, w_shared)
    mag_L_dB = 20 * np.log10(mag_L)
    phase_L_deg = phase_L / np.pi * 180

    # [mag_L_dB[0], mag_L_dB[-1]]
    vertical = [np.max([mag_L_dB[0], mag_P_dB[0]]), np.min([mag_L_dB[-1], mag_P_dB[-1]])]

    # Plot L and P
    fig_L_P, ax_L_P = plt.subplots()
    ax_L_P.set_ylabel(r'Magnitude (dB)')
    if kwargs.get("Hz", False):
        # Plot in Hz
        ax_L_P.set_xlabel(r'$\omega$ (Hz)')
        # Magnitude plot
        ax_L_P.semilogx(rps2Hz(np.array([w_d_h, w_d_h])), vertical, '--', color='C4', alpha=0.75, label=r'$20 \log(\gamma_d^{-1}) = %s$ (dB)' % np.round(20 * np.log10(1 / gamma_d), 2))
        ax_L_P.semilogx(rps2Hz(np.array([w_u_l, w_u_l])), vertical, '-.', color='C5', alpha=0.75, label=r'$20 \log(\gamma_u^{-1}) = %s$ (dB)' % np.round(20 * np.log10(1 / gamma_u), 2))
        ax_L_P.semilogx(rps2Hz(w_L), mag_L_dB, label=r'$|L(j\omega)|$')
        ax_L_P.semilogx(rps2Hz(w_P), mag_P_dB, color='C2', label=r'$|P(j\omega)|$')
    else:
        # Plot in rad/s
        ax_L_P.set_xlabel(r'$\omega$ (rad/s)')
        # Magnitude plot
        ax_L_P.semilogx(np.array([w_d_h, w_d_h]), vertical, '--', color='C4', alpha=0.75, label=r'$20 \log(\gamma_d^{-1}) = %s$ (dB)' % np.round(20 * np.log10(1 / gamma_d), 2))
        ax_L_P.semilogx(np.array([w_u_l, w_u_l]), vertical, '-.', color='C5', alpha=0.75, label=r'$20 \log(\gamma_u^{-1}) = %s$ (dB)' % np.round(20 * np.log10(1 / gamma_u), 2))
        ax_L_P.semilogx(w_L, mag_L_dB, label=r'$|L(j\omega)|$')
        ax_L_P.semilogx(w_P, mag_P_dB, color='C2', label=r'$|P(j\omega)|$')


    return fig_L_P, ax_L_P


def bode_mag_L_P_C(P, C, gamma_r, w_r_h, gamma_n, w_n_l, w_shared_low, w_shared_high, w_shared, **kwargs):
    """Plot L, P, and C."""
    # Noise bounds
    w_r = np.logspace(w_shared_low, np.log10(w_r_h), 100)
    w_n = np.logspace(np.log10(w_n_l), w_shared_high, 100)

    # Gain bounds
    gamma_n_dB = 20 * np.log10(gamma_n) * np.ones(w_n.shape[0],)
    gamma_r_inverse_dB = 20 * np.log10(1 / gamma_r) * np.ones(w_r.shape[0],)

    # L
    L = control.minreal(P * C)

    # Frequency response
    mag_L, phase_L, w_L = control.frequency_response(L, w_shared)
    mag_L_dB = 20 * np.log10(mag_L)
    phase_L_deg = phase_L / np.pi * 180

    mag_P, phase_P, w_P = control.frequency_response(P, w_shared)
    mag_P_dB = 20 * np.log10(mag_P)
    phase_P_deg = phase_P / np.pi * 180

    mag_C, phase_C, w_C = control.frequency_response(C, w_shared)
    mag_C_dB = 20 * np.log10(mag_C)
    phase_C_deg = phase_C / np.pi * 180

    # Plot L, P, and C
    fig_L_P_C, ax_L_P_C = plt.subplots()
    ax_L_P_C.set_ylabel(r'Magnitude (dB)')
    if kwargs.get("Hz", False):
        # Plot in Hz
        ax_L_P_C.set_xlabel(r'$\omega$ (Hz)')
        # Magnitude plot
        ax_L_P_C.semilogx(rps2Hz(w_L), mag_L_dB, label=r'$|L(j\omega)|$')
        ax_L_P_C.semilogx(rps2Hz(w_P), mag_P_dB, color='C2', label=r'$|P(j\omega)|$')
        ax_L_P_C.semilogx(rps2Hz(w_C), mag_C_dB, color='C8', label=r'$|C(j\omega)|$')
        ax_L_P_C.semilogx(rps2Hz(w_r), gamma_r_inverse_dB, '-', color='C3', label=r'$1/\gamma_r$')
        ax_L_P_C.semilogx(rps2Hz(w_n), gamma_n_dB, color='C6', label=r'$\gamma_n$')
    else:
        # Plot in rad/s
        ax_L_P_C.set_xlabel(r'$\omega$ (rad/s)')
        # Magnitude plot
        ax_L_P_C.semilogx(w_L, mag_L_dB, label=r'$|L(j\omega)|$')
        ax_L_P_C.semilogx(w_P, mag_P_dB, color='C2', label=r'$|P(j\omega)|$')
        ax_L_P_C.semilogx(w_C, mag_C_dB, color='C8', label=r'$|C(j\omega)|$')
        ax_L_P_C.semilogx(w_r, gamma_r_inverse_dB, '-', color='C3', label=r'$1/\gamma_r$')
        ax_L_P_C.semilogx(w_n, gamma_n_dB, color='C6', label=r'$\gamma_n$')
    
    return fig_L_P_C, ax_L_P_C


def bode_margins(P, C, w_shared, **kwargs):
    """Plot Bode plot with margins."""
    # L
    L = control.minreal(P * C)

    # Frequency response
    mag_L, phase_L, w_L = control.frequency_response(L, w_shared)
    mag_L_dB = 20 * np.log10(mag_L)
    phase_L_deg = phase_L / np.pi * 180

    # Margins
    gm, pm, vm, wpc, wgc, wvm = control.stability_margins(L)
    gm_dB = 20 * np.log10(gm)  # Convert to dB

    # Plot Bode plot with margins
    fig_margins, ax_margins = plt.subplots(2, 1)
    ax_margins[0].set_ylabel(r'Magnitude (dB)')
    ax_margins[1].set_ylabel(r'Phase (deg)')
    if kwargs.get("Hz", False):
        # Plot in Hz
        # Magnitude plot
        ax_margins[0].semilogx(rps2Hz(w_L), mag_L_dB)
        ax_margins[0].semilogx(rps2Hz(wpc), -gm_dB, 'o', color='C3')
        ax_margins[0].set_xlabel(r'$\omega$ (Hz)')
        # Phase plot
        ax_margins[1].semilogx(rps2Hz(w_L), phase_L_deg)
        ax_margins[1].semilogx(rps2Hz(wgc), pm - 180, 'd', color='C4')
        # ax_margins[1].set_yticks(np.linspace(-180, -90, 4))
        ax_margins[1].set_xlabel(r'$\omega$ (Hz)')
    else:
        # Plot in rad/s
        # Magnitude plot
        ax_margins[0].semilogx(w_L, mag_L_dB)
        ax_margins[0].semilogx(wpc, -gm_dB, 'o', color='C3')
        ax_margins[0].set_xlabel(r'$\omega$ (rad/s)')
        # Phase plot
        ax_margins[1].semilogx(w_L, phase_L_deg)
        ax_margins[1].semilogx(wgc, pm - 180, 'd', color='C4')
        # ax_margins[1].set_yticks(np.linspace(-180, -90, 4))
        ax_margins[1].set_xlabel(r'$\omega$ (rad/s)')

    return fig_margins, ax_margins, gm, pm, vm, wpc, wgc, wvm


def bode_mag_Gof4(P, C, gamma_r, w_r_h, gamma_d, w_d_h, gamma_n, w_n_l, gamma_u, w_u_l, w_shared_low, w_shared_high, w_shared, **kwargs):
    """Plot the gang of four."""
    # Noise bounds
    w_r = np.logspace(w_shared_low, np.log10(w_r_h), 100)
    w_d = np.logspace(w_shared_low, np.log10(w_d_h), 100)
    w_n = np.logspace(np.log10(w_n_l), w_shared_high, 100)
    w_u = np.logspace(np.log10(w_u_l), w_shared_high, 100)

    gamma_r_dB = 20 * np.log10(gamma_r) * np.ones(w_r.shape[0],)
    gamma_d_dB = 20 * np.log10(gamma_d) * np.ones(w_d.shape[0],)
    gamma_n_dB = 20 * np.log10(gamma_n) * np.ones(w_n.shape[0],)
    gamma_u_dB = 20 * np.log10(gamma_u) * np.ones(w_u.shape[0],)

    # TFs
    T = control.feedback(P * C, 1, -1)
    S = control.feedback(1, P * C, -1)
    CS = control.minreal(C * S)
    PS = control.minreal(P * S)

    # Freq response
    mag_T, phase_T, w_T = control.frequency_response(T, w_shared)
    mag_S, phase_S, w_S = control.frequency_response(S, w_shared)
    mag_PS, phase_PS, w_PS = control.frequency_response(PS, w_shared)
    mag_CS, phase_CS, w_CS = control.frequency_response(CS, w_shared)

    # Convert to dB and deg
    mag_T_dB = 20 * np.log10(mag_T)
    phase_T_deg = phase_T / np.pi * 180
    mag_S_dB = 20 * np.log10(mag_S)
    phase_S_deg = phase_S / np.pi * 180
    mag_PS_dB = 20 * np.log10(mag_PS)
    phase_PS_deg = phase_PS / np.pi * 180
    mag_CS_dB = 20 * np.log10(mag_CS)
    phase_CS_deg = phase_CS / np.pi * 180

    # Gang of four
    fig_Gof4, ax_Gof4 = plt.subplots(2, 2)    
    if kwargs.get("Hz", False):
        # Plot in Hz
        # Magnitude plot of S
        ax_Gof4[0, 0].semilogx(rps2Hz(w_shared), 6 * np.ones(w_shared.shape[0],), '--', color='silver')
        ax_Gof4[0, 0].semilogx(rps2Hz(w_S), mag_S_dB, color='C1')
        ax_Gof4[0, 0].semilogx(rps2Hz(w_r), gamma_r_dB, color='C3', label=r'$\gamma_r$')
        ax_Gof4[0, 0].set_xlabel(r'$\omega$ (Hz)')

        # Magnitude plot of P * S
        ax_Gof4[0, 1].semilogx(rps2Hz(w_PS), mag_PS_dB)
        ax_Gof4[0, 1].semilogx(rps2Hz(w_d), gamma_d_dB, color='C4', label=r'$\gamma_d$')
        ax_Gof4[0, 1].set_xlabel(r'$\omega$ (Hz)')

        # Magnitude plot of C * S
        ax_Gof4[1, 0].semilogx(rps2Hz(w_CS), mag_CS_dB)
        ax_Gof4[1, 0].semilogx(rps2Hz(w_u), gamma_u_dB, color='C5', label=r'$\gamma_u$')
        ax_Gof4[1, 0].set_xlabel(r'$\omega$ (Hz)')

        # Magnitude plot of T
        ax_Gof4[1, 1].semilogx(rps2Hz(w_shared), 2 * np.ones(w_shared.shape[0],), '-.', color='silver')
        ax_Gof4[1, 1].semilogx(rps2Hz(w_T), mag_T_dB, color='C9')
        ax_Gof4[1, 1].semilogx(rps2Hz(w_n), gamma_n_dB, color='C6', label=r'$\gamma_n$')
        ax_Gof4[1, 1].set_xlabel(r'$\omega$ (Hz)')

    else:
        # Plot in rad/s
        # Magnitude plot of S
        ax_Gof4[0, 0].semilogx(w_shared, 6 * np.ones(w_shared.shape[0],), '--', color='silver')
        ax_Gof4[0, 0].semilogx(w_S, mag_S_dB, color='C1')
        ax_Gof4[0, 0].semilogx(w_r, gamma_r_dB, color='C3', label=r'$\gamma_r$')
        ax_Gof4[0, 0].set_xlabel(r'$\omega$ (rad/s)')

        # Magnitude plot of P * S
        ax_Gof4[0, 1].semilogx(w_PS, mag_PS_dB)
        ax_Gof4[0, 1].semilogx(w_d, gamma_d_dB, color='C4', label=r'$\gamma_d$')
        ax_Gof4[0, 1].set_xlabel(r'$\omega$ (rad/s)')

        # Magnitude plot of C * S
        ax_Gof4[1, 0].semilogx(w_CS, mag_CS_dB)
        ax_Gof4[1, 0].semilogx(w_u, gamma_u_dB, color='C5', label=r'$\gamma_u$')
        ax_Gof4[1, 0].set_xlabel(r'$\omega$ (rad/s)')

        # Magnitude plot of T
        ax_Gof4[1, 1].semilogx(w_shared, 2 * np.ones(w_shared.shape[0],), '-.', color='silver')
        ax_Gof4[1, 1].semilogx(w_T, mag_T_dB, color='C9')
        ax_Gof4[1, 1].semilogx(w_n, gamma_n_dB, color='C6', label=r'$\gamma_n$')
        ax_Gof4[1, 1].set_xlabel(r'$\omega$ (rad/s)')

    
    # S
    ax_Gof4[0, 0].set_ylabel(r'$|S(j\omega)|$ (dB)')
    ax_Gof4[0, 0].legend(loc='lower right')
    # P * S
    ax_Gof4[0, 1].set_ylabel(r'$|P(j\omega)S(j\omega)|$ (dB)')
    ax_Gof4[0, 1].legend(loc='lower left')
    # C * s
    ax_Gof4[1, 0].set_ylabel(r'$|C(j\omega)S(j\omega)|$ (dB)')
    ax_Gof4[1, 0].legend(loc='upper left')
    # T
    ax_Gof4[1, 1].set_ylabel(r'$|T(j\omega)|$ (dB)')
    ax_Gof4[1, 1].legend(loc='lower left')
    
    fig_Gof4.tight_layout()

    return fig_Gof4, ax_Gof4

