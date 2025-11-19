"""Plotting utilities for the analysis"""

import numpy as np
import matplotlib.pyplot as plt


def setup_plot_style():
    """
    Set up consistent plot styling for all figures

    Call this at the beginning of your notebook
    """
    plt.rcParams['figure.figsize'] = (6.4, 4.8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 9


def plot_residuals_total(df):
    """
    Plot E_rec - E_true for all data (Figure 1.1)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns E_true and E_rec

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    residuals = df['E_rec'] - df['E_true']

    ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('$E_{rec} - E_{true}$ [GeV]')
    ax.set_ylabel('Counts')
    ax.set_title('Energy Residuals (All Data)')
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_residuals_by_energy(data_dict):
    """
    Plot E - E0 with histograms overlaid for each E0 (Figure 1.2)

    Parameters
    ----------
    data_dict : dict
        Dictionary mapping E0 -> array of E measurements

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    colors = plt.cm.viridis(np.linspace(0, 1, len(data_dict)))

    for i, (e0, e_measurements) in enumerate(sorted(data_dict.items())):
        residuals = e_measurements - e0
        ax.hist(residuals, bins=30, alpha=0.5, label=f'$E_0$ = {e0} GeV',
                color=colors[i], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('$E - E_0$ [GeV]')
    ax.set_ylabel('Counts')
    ax.set_title('Energy Residuals by True Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_sample_statistics(stats_df, e0_values, lam=None, delta=None, a=None, b=None, c=None):
    """
    Plot sample means and standard deviations vs E0 with optional fitted curves (Figure 1.3)

    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame with columns: E0, mean, mean_err, std, std_err
    e0_values : np.ndarray
        Array of E0 values for plotting fitted curves
    lam, delta : float, optional
        Fitted parameters for mean model (if provided, will plot fitted curve)
    a, b, c : float, optional
        Fitted parameters for width model (if provided, will plot fitted curve)

    Returns
    -------
    fig, axes : matplotlib Figure and Axes (2 subplots)
    """
    from s1_sol.models import mean_energy_model, width_energy_model

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Left plot: Sample means vs E0
    axes[0].errorbar(stats_df['E0'], stats_df['mean'], yerr=stats_df['mean_err'],
                     fmt='ko', label='Sample means', capsize=3)
    if lam is not None and delta is not None:
        e0_smooth = np.linspace(e0_values.min(), e0_values.max(), 100)
        mu_fit = mean_energy_model(e0_smooth, lam, delta)
        axes[0].plot(e0_smooth, mu_fit, 'r-', label=f'Fit: λ={lam:.4f}, Δ={delta:.4f}')
    axes[0].set_xlabel('$E_0$ [GeV]')
    axes[0].set_ylabel('$\\mu_E$ [GeV]')
    axes[0].set_title('Sample Means vs True Energy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right plot: Sample stds vs E0
    axes[1].errorbar(stats_df['E0'], stats_df['std'], yerr=stats_df['std_err'],
                     fmt='ko', label='Sample std devs', capsize=3)
    if a is not None and b is not None and c is not None:
        e0_smooth = np.linspace(e0_values.min(), e0_values.max(), 100)
        sigma_fit = width_energy_model(e0_smooth, a, b, c)
        axes[1].plot(e0_smooth, sigma_fit, 'b-',
                    label=f'Fit: a={a:.4f}, b={b:.4f}, c={c:.4f}')
    axes[1].set_xlabel('$E_0$ [GeV]')
    axes[1].set_ylabel('$\\sigma_E$ [GeV]')
    axes[1].set_title('Sample Standard Deviations vs True Energy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def plot_fitted_trends_with_bootstrap(e0_values, lam, delta, a, b, c,
                                      bootstrap_results=None, alpha=0.68):
    """
    Plot fitted parameter trends with bootstrap error bands (Figure 1.4)

    Parameters
    ----------
    e0_values : np.ndarray
        Array of E0 values for plotting
    lam, delta : float
        Fitted parameters for mean model
    a, b, c : float
        Fitted parameters for width model
    bootstrap_results : list of dict, optional
        List of bootstrap results, each dict containing fitted parameters
        If provided, will show bootstrap confidence bands
    alpha : float
        Confidence level for bootstrap bands (default: 0.68 for 1-sigma)

    Returns
    -------
    fig, axes : matplotlib Figure and Axes (2 subplots)
    """
    from s1_sol.models import mean_energy_model, width_energy_model

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    e0_smooth = np.linspace(e0_values.min(), e0_values.max(), 100)

    # Left plot: Mean energy model
    mu_central = mean_energy_model(e0_smooth, lam, delta)
    axes[0].plot(e0_smooth, mu_central, 'r-', linewidth=2, label='Best fit')

    if bootstrap_results is not None:
        # Compute bootstrap confidence band
        mu_bootstrap = np.array([
            mean_energy_model(e0_smooth, res['lam'], res['delta'])
            for res in bootstrap_results
        ])
        mu_lower = np.percentile(mu_bootstrap, 100 * (1 - alpha) / 2, axis=0)
        mu_upper = np.percentile(mu_bootstrap, 100 * (1 + alpha) / 2, axis=0)
        axes[0].fill_between(e0_smooth, mu_lower, mu_upper, alpha=0.3, color='red',
                            label=f'{int(alpha*100)}% Bootstrap CI')

    axes[0].set_xlabel('$E_0$ [GeV]')
    axes[0].set_ylabel('$\\mu_E$ [GeV]')
    axes[0].set_title('Mean Energy Model')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right plot: Width model
    sigma_central = width_energy_model(e0_smooth, a, b, c)
    axes[1].plot(e0_smooth, sigma_central, 'b-', linewidth=2, label='Best fit')

    if bootstrap_results is not None:
        # Compute bootstrap confidence band
        sigma_bootstrap = np.array([
            width_energy_model(e0_smooth, res['a'], res['b'], res['c'])
            for res in bootstrap_results
        ])
        sigma_lower = np.percentile(sigma_bootstrap, 100 * (1 - alpha) / 2, axis=0)
        sigma_upper = np.percentile(sigma_bootstrap, 100 * (1 + alpha) / 2, axis=0)
        axes[1].fill_between(e0_smooth, sigma_lower, sigma_upper, alpha=0.3, color='blue',
                            label=f'{int(alpha*100)}% Bootstrap CI')

    axes[1].set_xlabel('$E_0$ [GeV]')
    axes[1].set_ylabel('$\\sigma_E$ [GeV]')
    axes[1].set_title('Energy Resolution Model')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


def plot_bootstrap_distributions(bootstrap_results, param_names=None):
    """
    Plot histograms of bootstrap parameter distributions

    Parameters
    ----------
    bootstrap_results : list of dict
        List of bootstrap results, each dict containing fitted parameters
    param_names : list of str, optional
        Names of parameters to plot. If None, plots all parameters.

    Returns
    -------
    fig, axes : matplotlib Figure and Axes
    """
    if param_names is None:
        param_names = list(bootstrap_results[0].keys())

    n_params = len(param_names)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i, param in enumerate(param_names):
        values = [res[param] for res in bootstrap_results]
        axes[i].hist(values, bins=50, alpha=0.7, edgecolor='black')
        axes[i].set_xlabel(f'{param}')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'Bootstrap Distribution: {param}')
        axes[i].axvline(np.mean(values), color='r', linestyle='--', label='Mean')
        axes[i].axvline(np.median(values), color='g', linestyle='--', label='Median')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig, axes
