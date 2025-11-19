"""Sample statistics estimation functions"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from s1_sol.models import mean_energy_model, width_energy_model


def compute_sample_stats(data_dict):
    """
    Compute sample mean and standard deviation at each energy point

    Also computes the standard errors on these estimates:
    - Standard error of mean: σ / sqrt(n)
    - Standard error of std dev: σ / sqrt(2n) (approximate for large n)

    Parameters
    ----------
    data_dict : dict
        Dictionary mapping E0 (true energy) -> array of E (measured energies)

    Returns
    -------
    stats_df : pd.DataFrame
        DataFrame with columns:
        - E0: True energy value
        - mean: Sample mean
        - mean_err: Standard error on the mean
        - std: Sample standard deviation
        - std_err: Standard error on the standard deviation
        - n: Number of measurements

    Examples
    --------
    >>> stats_df = compute_sample_stats(data_dict)
    >>> print(stats_df)
    """
    results = []

    for e0 in sorted(data_dict.keys()):
        e_measurements = data_dict[e0]
        n = len(e_measurements)

        # Sample mean and std (use ddof=1 for unbiased estimator)
        mean = np.mean(e_measurements)
        std = np.std(e_measurements, ddof=1)

        # Standard errors
        mean_err = std / np.sqrt(n)
        std_err = std / np.sqrt(2 * n)  # Approximate formula for large n

        results.append({
            'E0': e0,
            'mean': mean,
            'mean_err': mean_err,
            'std': std,
            'std_err': std_err,
            'n': n
        })

    return pd.DataFrame(results)


def fit_mean_model_least_squares(e0_values, means, mean_errors):
    """
    Fit the mean energy model: μ_E = λ * E0 + Δ using weighted least squares

    Parameters
    ----------
    e0_values : np.ndarray
        Array of true energy values
    means : np.ndarray
        Array of measured mean energies at each E0
    mean_errors : np.ndarray
        Array of errors on the means

    Returns
    -------
    lambda_fit : float
        Fitted value of lambda
    delta_fit : float
        Fitted value of Delta
    lambda_err : float
        Error on lambda
    delta_err : float
        Error on Delta

    Notes
    -----
    Uses weighted least squares with weights = 1/sigma^2
    You can use scipy.optimize.curve_fit or numpy.polyfit or implement manually
    """
    # Initial guesses: lambda ~ 1 (perfect calibration), delta ~ 0 (no offset)
    p0 = [1.0, 0.0]

    # Perform weighted least squares fit
    # sigma parameter provides weights = 1/sigma^2
    # absolute_sigma=True means errors are in same units as data
    popt, pcov = curve_fit(
        mean_energy_model,
        e0_values,
        means,
        p0=p0,
        sigma=mean_errors,
        absolute_sigma=True
    )

    # Extract parameters and errors
    lambda_fit, delta_fit = popt
    lambda_err, delta_err = np.sqrt(np.diag(pcov))

    return lambda_fit, delta_fit, lambda_err, delta_err


def fit_width_model_least_squares(e0_values, stds, std_errors):
    """
    Fit the width model: (σ_E/E0)^2 = (a/sqrt(E0))^2 + (b/E0)^2 + c^2

    This can be rearranged to: σ_E^2 = a^2 * E0 + b^2 + c^2 * E0^2

    Parameters
    ----------
    e0_values : np.ndarray
        Array of true energy values
    stds : np.ndarray
        Array of measured standard deviations at each E0
    std_errors : np.ndarray
        Array of errors on the standard deviations

    Returns
    -------
    a_fit : float
        Fitted value of a
    b_fit : float
        Fitted value of b
    c_fit : float
        Fitted value of c
    a_err : float
        Error on a
    b_err : float
        Error on b
    c_err : float
        Error on c

    Notes
    -----
    Be careful with error propagation: if you fit σ^2, the errors are ~2*σ*σ_err
    You can use scipy.optimize.curve_fit
    """
    # Initial guesses for a, b, c
    # Typical values: a ~ 0.1 (stochastic term), b ~ 0.01 (noise), c ~ 0.01 (constant)
    p0 = [0.1, 0.01, 0.01]

    # Bounds to ensure physical parameters (all positive)
    bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

    # Perform weighted least squares fit
    # Fit σ_E directly (not σ_E^2) to avoid error propagation complications
    popt, pcov = curve_fit(
        width_energy_model,
        e0_values,
        stds,
        p0=p0,
        sigma=std_errors,
        absolute_sigma=True,
        bounds=bounds
    )

    # Extract parameters and errors
    a_fit, b_fit, c_fit = popt
    a_err, b_err, c_err = np.sqrt(np.diag(pcov))

    return a_fit, b_fit, c_fit, a_err, b_err, c_err
