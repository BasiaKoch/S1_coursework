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


# TODO: Add more plotting functions as needed for other figures
# Some suggestions:
# - plot_sample_statistics(stats_df) for Figure 1.3
# - plot_fitted_trends_with_bootstrap(e0_values, params, bootstrap_params) for Figure 1.4
# - etc.
