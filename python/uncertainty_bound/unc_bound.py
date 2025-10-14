"""Uncertainty bound functions.

James Forbes, Steven Dahdah, Jonathan Eid
2023/10/07 - Initial
2025/01/11 - Added Jonathan Eid's upper bound function. 
2025/10/06 - Cleaned up some of Jonathan Eid's functions. 
2025/10/14 - Jonathan Eid modified code to ensure correct DC gain.

To use this module, first call:

    R = unc_bound.residuals(P_nominal, P_off_nominal)

to get the list of residual transfer functions. Then call:

    mag_max_dB, mag_max_abs = unc_bound.residual_max_mag(R, w_shared)

to compute the worst-case residual magnitude over all frequencies. Using that
information, call:

    W2 = upperbound(omega=w_shared,
                    upper_bound=mag_max_abs,
                    degree=nW2)

where nW2 is the degree of the fit. Note, recommend to use control.minreal(W2)
to ensure W2 is minimal. 

"""

import control
import numpy as np
from scipy import optimize as opt


def residuals(P_nom, P_off_nom):
    """Compute the residuals between P_nom and P_off_nom.

    Parameters
    ----------
    P_nom : control.TransferFunction
        Nominal transfer function.
    P_off_nom : List[control.TransferFunction]
        Off-nominal transfer functions.

    Returns
    -------
    List[control.TransferFunction]
        Residual between the nominal transfer function and each off-nominal
        transfer function.
    """
    # Number of off-nominal plants.
    R = [plant / P_nom - 1 for plant in P_off_nom]
    return R


def residual_max_mag(R, w_shared):
    """Compute the max of all the residuals.

    Parameters
    ----------
    R : List[control.TransferFunction]
        Residual transfer functions
    w_shared : np.ndarray
        Frequencies to evaluate TF at (rad/s)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Maximum of all residuals in dB and absolute value respectively.
    """
    # Compute all magnitudes
    mag = np.vstack([
        control.frequency_response(residual, w_shared)[0] for residual in R
    ])

    # Compute maximum magnitude at each frequency
    mag_max_abs = np.max(mag, axis=0)
    # Compute dB
    mag_max_dB = 20 * np.log10(mag_max_abs)
    return mag_max_dB, mag_max_abs


def upperbound(omega: np.array,
        upper_bound: np.array,
        degree: int) -> control.TransferFunction:
    """Calculate the optimal upper bound transfer function of residuals.

    Form of W2 is prescribed to be
        W2(s) = q(s) / p(s),
    where p and q are a monomial and polynomial of given degree, respectively.

    WARNING: Performance is poor due to hard-coded initial guess.

    Parameters
    ----------
    omega : np.array
        Frequency domain over which W2 should bound the residuals.
    upper_bound : np.array
        Maximum magnitude of residuals over frequency domain. Absolute units,
        not decibels. Length of array must equal to that of array omega.
    degree : int
        Degree of numerator and denominator polynomials of biproper rational
        function that is W2.
    
    Returns
    -------
    W2 : control.TransferFunction
        Optimal upper bound transfer function of residuals.
    """

    # Error function.
    def _e(c: np.array) -> np.array:
        """Calculate the error over the frequency domain.
        
        The error at a particular frequency is defined as the difference between
        the maximum magnitude of residuals at that frequency and the magnitude
        of _W2 with parameters c evaluated at that frequency:
            error(w) = upper_bound(w) - |W2(c)(w)|.
        This function returns the error over each point in the frequency domain.
        
        Parameters
        ----------
        c : np.array
            Parameters of W2 for which the error is calculated.
            Form of c is 
                [q_n, ..., q_0, p_n-1, ..., p_0].
        
        Returns
        -------
        e : np.array
            Error over the frequency domain.
        """
        num_W2 = np.polyval(c[:degree + 1], 1e0j * omega)
        # den_W2 = np.polyval(c[degree + 1:], 1e0j * omega)
        den_W2 = np.polyval(np.insert(c[degree + 1:], 0, 1.0), 1e0j * omega)
        W2 = num_W2 / den_W2
        mag_W2 = np.abs(W2)
        e = mag_W2 - upper_bound
        return e


    # Optimization objective function.
    def _J(c : np.array) -> np.double:
        """Calculate the optimization objective.
        
        The optimization objective is defined as the sum over each frequency
        point of the squared error at that frequency point.

        Parameters
        ----------
        c : np.array
            Parameters of W2 for which the optimization objective is calculated.
            Form of c is 
                [q_n, ..., q_0, p_n-1, ..., p_0].
        
        Returns
        -------
        J : np.double
            Optimization objective.
        """
        err = _e(c)
        J = np.sum(err**2, dtype=np.double)
        return J
    
    # Initial guess at W2 is a constant transfer function with value equal to
    # peak of upper bound.
    c0 = np.zeros(2 * degree + 1)
    c0[degree] = upper_bound.max() + 1e-6
    c0[-1] = 1e0

    # Optimization problem and solution
    constraint = {'type': 'ineq', 'fun': _e}
    result = opt.minimize(
        fun=_J,
        x0=list(c0),
        method='SLSQP',
        constraints=constraint,
        options={'maxiter': 100000},
    )
    
    c_opt = result.x

    # Replace real parts of poles and zeros with their absolute values to ensure
    # that W2 is asymptotically stable and minimum phase.
    num_c_opt = c_opt[: degree + 1]
    num_roots = np.roots(num_c_opt)
    new_num_roots = -np.abs(np.real(num_roots)) + 1e0j * np.imag(num_roots)

    den_c_opt = np.insert(c_opt[degree + 1 :], 0, 1.0)
    den_roots = np.roots(den_c_opt)
    new_den_roots = -np.abs(np.real(den_roots)) + 1e0j * np.imag(den_roots)

    gain_num_roots = np.prod(-new_num_roots).real
    gain_den_roots = np.prod(-new_den_roots).real
    dc_gain = np.abs(num_c_opt[-1] / den_c_opt[-1])
    zpk_gain = dc_gain * gain_den_roots / gain_num_roots

    # Form asymptotically stable, minimum phase optimal upper bound.
    W2_opt = control.zpk(new_num_roots, new_den_roots, zpk_gain)
    W2_opt_min_real = control.minreal(W2_opt, verbose=False)

    return W2_opt_min_real