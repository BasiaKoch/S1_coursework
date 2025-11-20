"""Sample statistics + least-squares fit functions"""

import numpy as np
import pandas as pd
from iminuit import Minuit
from iminuit.cost import LeastSquares

from s1_sol.models import mean_energy_model, width_energy_model


# ============================================================
# 1. Compute sample statistics
# ============================================================

def compute_sample_stats(data_dict):
    """
    Compute sample mean, sample std, and their uncertainties.
    """
    results = []

    for e0 in sorted(data_dict.keys()):
        x = np.asarray(data_dict[e0])
        N = len(x)

        mean = np.mean(x)
        std = np.std(x, ddof=1)

        mean_err = std / np.sqrt(N)
        std_err = std / np.sqrt(2 * (N - 1))

        results.append({
            "E0": e0,
            "mean": mean,
            "mean_err": mean_err,
            "std": std,
            "std_err": std_err,
            "n": N
        })

    return pd.DataFrame(results)


# ============================================================
# 2. Fit mean-energy model: μ = λ E0 + Δ
# ============================================================

def fit_mean_model_least_squares(e0_values, means, mean_errors):
    """
    Weighted least-squares fit for μ(E0) = λE0 + Δ.
    Uses iminuit + LeastSquares (as required).
    """

    # Define chi-square cost function
    ls = LeastSquares(e0_values, means, mean_errors, mean_energy_model)

    # Initial guesses
    m = Minuit(ls, lam=1.0, delta=0.0)

    # Run minimization
    m.migrad()

    # Extract fit parameters and uncertainties
    lam_fit   = m.values["lam"]
    delta_fit = m.values["delta"]
    lam_err   = m.errors["lam"]
    delta_err = m.errors["delta"]

    return lam_fit, delta_fit, lam_err, delta_err


# ============================================================
# 3. Fit width model:
#     (σ/E0)^2 = (a/√E0)^2 + (b/E0)^2 + c^2
# ============================================================

def fit_width_model_least_squares(e0_values, stds, std_errors):
    """
    Fit the calorimeter-style width model using iminuit least squares.
    """

    # Define chi-square cost function
    ls = LeastSquares(e0_values, stds, std_errors, width_energy_model)

    # Initial guesses for parameters
    m = Minuit(ls, a=0.5, b=0.1, c=0.01)

    # Physical limits: a >= 0, c >= 0
    m.limits["a"] = (0, None)
    m.limits["c"] = (0, None)

    m.migrad()

    # Extract best-fit parameters
    a_fit = m.values["a"]
    b_fit = m.values["b"]
    c_fit = m.values["c"]

    # Extract uncertainties
    a_err = m.errors["a"]
    b_err = m.errors["b"]
    c_err = m.errors["c"]

    return a_fit, b_fit, c_fit, a_err, b_err, c_err
