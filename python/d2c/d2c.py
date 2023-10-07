"""
Discrete-time to continuous-time, module.

J R Forbes, 2022/01/18
"""

# %%
# Libraries
import numpy as np
import control
from scipy.linalg import expm, logm

# %%
# Classes

def d2c(Pd):
    '''
    Parameters
    ----------
    Pd : Discrete-time (DT) transfer function (TF) from control
        package computed uzing a zoh.
        A DT TF to be converted to a continuous-time (CT) TF.

    Returns
    -------
    Pc : A CT TF computed from the DT TF given.

    References
    -------
    K. J. Astrom and B. Wittenmark, Computer Controlled Systems:
        Theory and Design, 3rd., Prentice-Hall, Inc., 1997, pp. 32-37.
    '''
    # Preliminary calculations
    dt = Pd.dt  # time step
    Pd_ss = control.ss(Pd)  # Convert Pd(z) TF to state-space (SS) realization
    Ad, Bd, Cd, Dd = Pd_ss.A, Pd_ss.B, Pd_ss.C, Pd_ss.D  # Extract SS matrices
    n_x, n_u = Ad.shape[0], Bd.shape[1]  # Extract shape of SS matrices

    # Form the matrix Phi, which is composed of Ad and Bd
    Phi1 = np.hstack([Ad, Bd])
    Phi2 = np.hstack([np.zeros([n_u, n_x]), np.eye(n_u)])
    Phi = np.vstack([Phi1, Phi2])

    # Compute Upsilon the matrix log of Phi
    Upsilon = logm(Phi) / dt

    # Extract continuous-time Ac and Bc
    # (Recall, a SS realization is *not* unique. The matrices extracted
    # from Upsilon may not equal Ac and Bc in some canonical form.)
    Ac = Upsilon[:n_x, :n_x]
    Bc = Upsilon[:n_x, (n_x - n_u + 1):]

    # The continuous-time Cc and Cc equal the discrete-time Cd and Dd
    Cc, Dc = Cd, Dd

    # Compute the transfer function Pc(s)
    Pc = control.ss2tf(Ac, Bc, Cc, Dc)
    return Pc