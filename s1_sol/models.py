"""Physics model functions for photon detector calibration"""

import numpy as np


def mean_energy_model(E0, lam, delta):
    """
    Model for mean measured energy as a function of true energy

    μ_E = λ * E0 + Δ

    Parameters
    ----------
    E0 : float or np.ndarray
        True energy value(s)
    lam : float
        Lambda parameter (linear response)
    delta : float
        Delta parameter (energy offset)

    Returns
    -------
    mu_E : float or np.ndarray
        Expected mean measured energy
    """
    return lam * E0 + delta


def width_energy_model(E0, a, b, c):
    """
    Model for width (standard deviation) of measured energy as a function of true energy

    (σ_E / E0)^2 = (a / sqrt(E0))^2 + (b / E0)^2 + c^2

    Rearranging: σ_E = E0 * sqrt((a/sqrt(E0))^2 + (b/E0)^2 + c^2)

    Parameters
    ----------
    E0 : float or np.ndarray
        True energy value(s)
    a : float
        Stochastic term coefficient
    b : float
        Noise term coefficient
    c : float
        Constant term coefficient

    Returns
    -------
    sigma_E : float or np.ndarray
        Expected standard deviation of measured energy
    """
    # Avoid division by zero
    E0 = np.atleast_1d(E0)
    sigma_over_E0_sq = (a / np.sqrt(E0))**2 + (b / E0)**2 + c**2
    sigma_E = E0 * np.sqrt(sigma_over_E0_sq)

    # Return scalar if input was scalar
    return float(sigma_E) if len(sigma_E) == 1 else sigma_E


def relative_width_squared(E0, a, b, c):
    """
    Relative width squared: (σ_E / E0)^2

    This is useful for plotting and fitting.

    Parameters
    ----------
    E0 : float or np.ndarray
        True energy value(s)
    a, b, c : float
        Model parameters

    Returns
    -------
    rel_width_sq : float or np.ndarray
        (σ_E / E0)^2
    """
    E0 = np.atleast_1d(E0)
    result = (a / np.sqrt(E0))**2 + (b / E0)**2 + c**2
    return float(result) if len(result) == 1 else result
