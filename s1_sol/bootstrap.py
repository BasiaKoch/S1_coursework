"""Bootstrap analysis utilities"""

import numpy as np
from tqdm import tqdm
from s1_sol.fitting import fit_simultaneous_all_energies


def bootstrap_sample(data_dict, random_state=None):
    """
    Create a bootstrap sample by sampling with replacement

    Preserves the structure: for each E0, sample N measurements with replacement
    where N is the original number of measurements at that E0.

    Parameters
    ----------
    data_dict : dict
        Dictionary mapping E0 -> array of E measurements
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    bootstrap_dict : dict
        Bootstrap sample with same structure as input
    """
    if random_state is not None:
        np.random.seed(random_state)

    bootstrap_dict = {}
    for e0, e_measurements in data_dict.items():
        n = len(e_measurements)
        # Sample with replacement
        bootstrap_indices = np.random.randint(0, n, size=n)
        bootstrap_dict[e0] = e_measurements[bootstrap_indices]

    return bootstrap_dict


def run_bootstrap_analysis(data_dict, analysis_function, n_bootstrap=2500,
                           show_progress=True, random_state=None):
    """
    Run bootstrap analysis by repeatedly resampling and applying analysis function

    Parameters
    ----------
    data_dict : dict
        Original data dictionary mapping E0 -> measurements
    analysis_function : callable
        Function that takes data_dict and returns a dict of results
        Example: lambda data: {'mean': compute_mean(data)}
    n_bootstrap : int
        Number of bootstrap samples (default: 2500)
    show_progress : bool
        Whether to show progress bar
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    bootstrap_results : list of dict
        List of results from each bootstrap iteration

    Examples
    --------
    >>> def my_analysis(data):
    ...     # Do some analysis
    ...     return {'param1': value1, 'param2': value2}
    >>> results = run_bootstrap_analysis(data_dict, my_analysis, n_bootstrap=1000)
    >>> param1_values = [r['param1'] for r in results]
    >>> param1_std = np.std(param1_values)
    """
    if random_state is not None:
        np.random.seed(random_state)

    bootstrap_results = []

    iterator = range(n_bootstrap)
    if show_progress:
        iterator = tqdm(iterator, desc="Bootstrap")

    for i in iterator:
        # Create bootstrap sample
        bootstrap_data = bootstrap_sample(data_dict)

        # Apply analysis function
        result = analysis_function(bootstrap_data)
        bootstrap_results.append(result)

    return bootstrap_results


def compute_bootstrap_confidence_interval(bootstrap_values, confidence_level=0.68):
    """
    Compute confidence interval from bootstrap distribution

    Parameters
    ----------
    bootstrap_values : array-like
        Array of bootstrap estimates
    confidence_level : float
        Confidence level (default: 0.68 for ±1σ)

    Returns
    -------
    median : float
        Median of bootstrap distribution
    lower : float
        Lower confidence bound
    upper : float
        Upper confidence bound
    """
    bootstrap_values = np.array(bootstrap_values)

    alpha = 1 - confidence_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)

    median = np.median(bootstrap_values)
    lower = np.percentile(bootstrap_values, lower_percentile)
    upper = np.percentile(bootstrap_values, upper_percentile)

    return median, lower, upper


def bootstrap_simultaneous_fit(data_dict, n_boot=500, random_seed=456):
    """
    Bootstrap the simultaneous fit to get parameter distributions

    Parameters
    ----------
    data_dict : dict
        Dictionary mapping E0 -> array of E measurements
    n_boot : int
        Number of bootstrap samples (default: 500)
    random_seed : int
        Random seed for reproducibility (default: 456)

    Returns
    -------
    lb_samples : np.ndarray
        Bootstrap samples for lambda parameter
    dE_samples : np.ndarray
        Bootstrap samples for Delta parameter
    a_samples : np.ndarray
        Bootstrap samples for a parameter
    b_samples : np.ndarray
        Bootstrap samples for b parameter
    c_samples : np.ndarray
        Bootstrap samples for c parameter
    """
    lb_samples = []
    dE_samples = []
    a_samples = []
    b_samples = []
    c_samples = []

    rng = np.random.default_rng(random_seed)

    for i in range(n_boot):
        # Bootstrap sample with replacement
        boot_dict = {}
        for e0, measurements in data_dict.items():
            boot_dict[e0] = rng.choice(measurements, size=len(measurements), replace=True)

        # Fit on bootstrap sample
        params, errors, m = fit_simultaneous_all_energies(boot_dict)

        lb_samples.append(params['lb'])
        dE_samples.append(params['dE'])
        a_samples.append(params['a'])
        b_samples.append(params['b'])
        c_samples.append(params['c'])

    return (np.array(lb_samples), np.array(dE_samples),
            np.array(a_samples), np.array(b_samples), np.array(c_samples))


###4



from s1_sol.statistics import (
    compute_sample_stats,
    fit_mean_model_least_squares,
    fit_width_model_least_squares,
)
from s1_sol.fitting import (
    fit_individual_energies,
    fit_simultaneous_all_energies,
)


def bootstrap_all_methods(data_dict,
                          n_bootstrap=2500,
                          random_seed=123,
                          show_progress=True):
    """
    Non-parametric bootstrap of the *entire* analysis.

    For each bootstrap replica:
      1. Resample the raw measurements at each E0 with replacement.
      2. Recompute:
         - Sample least-squares (LS) fit {lb, dE, a, b, c}
         - Individual-energy MLE -> LS on {mu_i, sigma_i} -> {lb, dE, a, b, c}
         - Simultaneous unbinned MLE {lb, dE, a, b, c}

    Parameters
    ----------
    data_dict : dict
        Original data, mapping E0 -> array of E measurements
    n_bootstrap : int
        Number of bootstrap replicas (default: 2500)
    random_seed : int
        Seed for reproducibility
    show_progress : bool
        Whether to show a tqdm progress bar

    Returns
    -------
    boot_ls : dict
        {"lb": np.ndarray, "dE": ..., "a": ..., "b": ..., "c": ...}
        Each array has length n_bootstrap with LS estimates.
    boot_ind : dict
        Same structure, from individual-energy MLE → LS.
    boot_sim : dict
        Same structure, from simultaneous MLE.
    """

    rng = np.random.default_rng(random_seed)

    # Containers for each method
    keys = ["lb", "dE", "a", "b", "c"]
    boot_ls  = {k: np.zeros(n_bootstrap) for k in keys}
    boot_ind = {k: np.zeros(n_bootstrap) for k in keys}
    boot_sim = {k: np.zeros(n_bootstrap) for k in keys}

    iterator = range(n_bootstrap)
    if show_progress:
        iterator = tqdm(iterator, desc="Bootstrap (all methods)")

    for ib in iterator:
        # --------------------------------------------------
        # 1. Resample raw data at each E0 (non-parametric)
        # --------------------------------------------------
        boot_dict = {}
        for e0, measurements in data_dict.items():
            boot_dict[e0] = rng.choice(measurements,
                                       size=len(measurements),
                                       replace=True)

        # --------------------------------------------------
        # 2A. Sample-based least-squares (LS)
        # --------------------------------------------------
        stats_bs = compute_sample_stats(boot_dict)

        lam_ls, dE_ls, _, _ = fit_mean_model_least_squares(
            stats_bs["E0"].values,
            stats_bs["mean"].values,
            stats_bs["mean_err"].values,
        )

        a_ls, b_ls, c_ls, _, _, _ = fit_width_model_least_squares(
            stats_bs["E0"].values,
            stats_bs["std"].values,
            stats_bs["std_err"].values,
        )

        boot_ls["lb"][ib] = lam_ls
        boot_ls["dE"][ib] = dE_ls
        boot_ls["a"][ib]  = a_ls
        boot_ls["b"][ib]  = b_ls
        boot_ls["c"][ib]  = c_ls

        # --------------------------------------------------
        # 2B. Individual-energy MLE → LS on (mu_i, sigma_i)
        # --------------------------------------------------
        indiv_df = fit_individual_energies(boot_dict)

        lam_ind, dE_ind, _, _ = fit_mean_model_least_squares(
            indiv_df["E0"].values,
            indiv_df["mu"].values,
            indiv_df["mu_err"].values,
        )

        a_ind, b_ind, c_ind, _, _, _ = fit_width_model_least_squares(
            indiv_df["E0"].values,
            indiv_df["sigma"].values,
            indiv_df["sigma_err"].values,
        )

        boot_ind["lb"][ib] = lam_ind
        boot_ind["dE"][ib] = dE_ind
        boot_ind["a"][ib]  = a_ind
        boot_ind["b"][ib]  = b_ind
        boot_ind["c"][ib]  = c_ind

        # --------------------------------------------------
        # 2C. Simultaneous unbinned MLE
        # --------------------------------------------------
        params_sim, _, _ = fit_simultaneous_all_energies(boot_dict)

        boot_sim["lb"][ib] = params_sim["lb"]
        boot_sim["dE"][ib] = params_sim["dE"]
        boot_sim["a"][ib]  = params_sim["a"]
        boot_sim["b"][ib]  = params_sim["b"]
        boot_sim["c"][ib]  = params_sim["c"]

    return boot_ls, boot_ind, boot_sim
