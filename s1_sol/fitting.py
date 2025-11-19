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


def fit_simultaneous_all_energies(data_dict, initial_params=None):
    """
    Simultaneous fit of all parameters {λ, Δ, a, b, c} to all data at once

    This fits the full model:
    - μ_E = λ * E0 + Δ
    - σ_E = E0 * sqrt((a/sqrt(E0))^2 + (b/E0)^2 + c^2)

    Parameters
    ----------
    data_dict : dict
        Dictionary mapping E0 -> array of E measurements
    initial_params : dict, optional
        Initial guesses for parameters {'lam', 'delta', 'a', 'b', 'c'}

    Returns
    -------
    params : dict
        Fitted parameter values {'lam', 'delta', 'a', 'b', 'c'}
        (Note: 'lam' is lambda, 'delta' is Delta)
    errors : dict
        Errors on fitted parameters
    minuit_obj : Minuit
        The Minuit object (for advanced usage)

    Examples
    --------
    >>> params, errors, m = fit_simultaneous_all_energies(data_dict)
    >>> print(f"λ = {params['lam']:.4f} ± {errors['lam']:.4f}")
    >>> print(f"Δ = {params['delta']:.4f} ± {errors['delta']:.4f}")

    Notes
    -----
    The likelihood is the product of normal distributions at each E0:
    L = ∏_{i,j} Normal(E_ij | μ(E0_i), σ(E0_i))

    where i runs over energy points and j runs over measurements at each energy.
    """
    # Set default initial parameters if not provided
    if initial_params is None:
        initial_params = {
            'lam': 1.0,      # Expect lambda near 1
            'delta': 0.0,    # Expect small offset
            'a': 1.0,
            'b': 1.0,
            'c': 0.1
        }

    # Define the simultaneous negative log-likelihood
    def nll(lam, delta, a, b, c):
        # Check for invalid parameter values
        if a <= 0 or b <= 0 or c <= 0:
            return 1e10

        total_nll = 0.0

        for e0, e_measurements in data_dict.items():
            # Compute expected mean and width at this E0
            mu_E = lam * e0 + delta
            sigma_over_E0_sq = (a / np.sqrt(e0))**2 + (b / e0)**2 + c**2
            sigma_E = e0 * np.sqrt(sigma_over_E0_sq)

            if sigma_E <= 0:
                return 1e10

            # Add log-likelihood for this energy point
            total_nll += -np.sum(norm.logpdf(e_measurements, mu_E, sigma_E))

        return total_nll

    # Create Minuit object
    m = Minuit(nll, **initial_params)

    # Set parameter limits (all parameters should be positive except lambda and delta)
    m.limits['a'] = (0, None)
    m.limits['b'] = (0, None)
    m.limits['c'] = (0, None)
    m.limits['lam'] = (0.5, 1.5)  # Lambda should be close to 1
    # delta can be positive or negative (no limits)

    m.errordef = Minuit.LIKELIHOOD

    # Run minimization
    m.migrad()

    # Check if fit was successful
    if not m.valid:
        print("Warning: Simultaneous fit may not have converged properly")
        print(f"Valid: {m.valid}, Accurate: {m.accurate}")

    # Extract results
    params = {
        'lam': m.values['lam'],
        'delta': m.values['delta'],
        'a': m.values['a'],
        'b': m.values['b'],
        'c': m.values['c']
    }

    errors = {
        'lam': m.errors['lam'],
        'delta': m.errors['delta'],
        'a': m.errors['a'],
        'b': m.errors['b'],
        'c': m.errors['c']
    }

    return params, errors, m
