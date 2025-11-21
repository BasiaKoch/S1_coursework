"""Maximum likelihood fitting functions using iminuit"""

import numpy as np
from iminuit import Minuit
from scipy.stats import norm


def fit_normal_mle(data, initial_mu=None, initial_sigma=None):
    """
    Fit a normal distribution to data using maximum likelihood estimation

    Parameters
    ----------
    data : np.ndarray
        Array of measurements
    initial_mu : float, optional
        Initial guess for mu (default: sample mean)
    initial_sigma : float, optional
        Initial guess for sigma (default: sample std)

    Returns
    -------
    mu : float
        Fitted mean
    sigma : float
        Fitted standard deviation
    mu_err : float
        Error on mu
    sigma_err : float
        Error on sigma
    minuit_obj : Minuit
        The Minuit object (for advanced usage)

    Examples
    --------
    >>> mu, sigma, mu_err, sigma_err, m = fit_normal_mle(data)
    >>> print(f"μ = {mu:.3f} ± {mu_err:.3f}")
    >>> print(f"σ = {sigma:.3f} ± {sigma_err:.3f}")
    """
    # Set initial guesses
    if initial_mu is None:
        initial_mu = np.mean(data)
    if initial_sigma is None:
        initial_sigma = np.std(data, ddof=1)

    # Define negative log-likelihood
    def nll(mu, sigma):
        if sigma <= 0:
            return 1e10  # Penalty for invalid sigma
        return -np.sum(norm.logpdf(data, mu, sigma))

    # Create Minuit object
    m = Minuit(nll, mu=initial_mu, sigma=initial_sigma)
    m.limits['sigma'] = (0, None)  # sigma must be positive
    m.errordef = Minuit.LIKELIHOOD  # For likelihood fits

    # Run minimization
    m.migrad()

    # Check if fit was successful
    if not m.valid:
        print("Warning: Fit may not have converged properly")

    return m.values['mu'], m.values['sigma'], m.errors['mu'], m.errors['sigma'], m


def fit_individual_energies(data_dict):
    """
    Fit normal distribution to data at each energy point separately

    Parameters
    ----------
    data_dict : dict
        Dictionary mapping E0 -> array of E measurements

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with columns:
        - E0: True energy
        - mu: Fitted mean
        - mu_err: Error on mean
        - sigma: Fitted width
        - sigma_err: Error on width

    Examples
    --------
    >>> results = fit_individual_energies(data_dict)
    >>> print(results)
    """
    import pandas as pd

    results = []
    for e0 in sorted(data_dict.keys()):
        mu, sigma, mu_err, sigma_err, _ = fit_normal_mle(data_dict[e0])
        results.append({
            'E0': e0,
            'mu': mu,
            'mu_err': mu_err,
            'sigma': sigma,
            'sigma_err': sigma_err
        })

    return pd.DataFrame(results)

import numpy as np
from scipy.stats import norm
from iminuit import Minuit
def fit_simultaneous_all_energies(data_dict, initial_params=None):
    """
    Unbinned maximum likelihood fit of all parameters {lb, dE, a, b, c}
    to all energies simultaneously.

    Model:
        μ(E0) = lb * E0 + dE
        σ(E0) = E0 * sqrt( (a/sqrt(E0))^2 + (b/E0)^2 + c^2 )
    """

    # ------------------------------------------------------------
    # Default starting values
    # ------------------------------------------------------------
    if initial_params is None:
        initial_params = dict(
            lb=1.0,
            dE=0.0,
            a=0.5,
            b=1.0,
            c=0.05,
        )

    # ------------------------------------------------------------
    # Negative log-likelihood
    # ------------------------------------------------------------
    def nll(lb, dE, a, b, c):

        if a <= 0 or b <= 0 or c <= 0:
            return np.inf

        total = 0.0

        for e0, measurements in data_dict.items():

            mu = lb * e0 + dE
            sigma_sq = (a/np.sqrt(e0))**2 + (b/e0)**2 + c**2
            sigma = e0 * np.sqrt(sigma_sq)

            if sigma <= 0:
                return np.inf

            total -= np.sum(norm.logpdf(measurements, mu, sigma))

        return total

    # ------------------------------------------------------------
    # Minuit object
    # ------------------------------------------------------------
    m = Minuit(nll, **initial_params)
    m.errordef = Minuit.LIKELIHOOD

    m.limits["a"] = (0, None)
    m.limits["b"] = (0, None)
    m.limits["c"] = (0, None)

    # ------------------------------------------------------------
    # Minimise
    # ------------------------------------------------------------
    m.migrad()
    m.hesse()

    # ------------------------------------------------------------
    # Collect results
    # ------------------------------------------------------------
    params = dict(
        lb=m.values["lb"],
        dE=m.values["dE"],
        a=m.values["a"],
        b=m.values["b"],
        c=m.values["c"],
    )

    errors = dict(
        lb=m.errors["lb"],
        dE=m.errors["dE"],
        a=m.errors["a"],
        b=m.errors["b"],
        c=m.errors["c"],
    )

    return params, errors, m
