"""Bootstrap analysis utilities"""

import numpy as np
from tqdm import tqdm


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
