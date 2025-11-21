"""Plotting utilities for the analysis"""

import os
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

    fig, ax = plt.subplots(figsize=(6.4, 4.8)) #create new matplotlib figure and axes 

    residuals = df['E_rec'] - df['E_true'] #residuals for each event 

    ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black') #plot histogram of residuals
    ax.set_xlabel('$E_{rec} - E_{true}$ [GeV]') 
    ax.set_ylabel('Counts')
    ax.set_title('Energy Residuals (All Data)')
    ax.grid(True, alpha=0.3)

    return fig, ax #return the figure and axes objects


def plot_residuals_by_energy(data_dict):

    fig, ax = plt.subplots(figsize=(6.4, 4.8)) 

    colors = plt.cm.viridis(np.linspace(0, 1, len(data_dict))) #generate one color per group

    for i, (e0, e_measurements) in enumerate(sorted(data_dict.items())): #loop over each true energy 
        residuals = e_measurements - e0 #residuals per goup
        ax.hist(residuals, bins=30, alpha=0.5, label=f'$E_0$ = {e0} GeV',
                color=colors[i], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('$E - E_0$ [GeV]')
    ax.set_ylabel('Counts')
    ax.set_title('Energy Residuals by True Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax

def plot_sample_statistics(stats_df, savepath="figs/Figure1.3.pdf"):
    """
    Plot sample means and standard deviations vs E0 (Figure 1.3)
    using the sample statistics computed earlier.

    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame with columns: E0, mean, mean_err, std, std_err

    Returns
    -------
    fig, axes : matplotlib Figure and array of Axes
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # ---------------------------------------------------------
    # Left subplot: sample mean vs E0
    # ---------------------------------------------------------
    axes[0].errorbar(
        stats_df['E0'], stats_df['mean'],
        yerr=stats_df['mean_err'],
        fmt='o', capsize=4
    )
    axes[0].set_xlabel(r"$E_0$ [GeV]")
    axes[0].set_ylabel(r"$\hat{\mu}_{\rm samp}$ [GeV]")
    axes[0].set_title("Sample Means vs True Energy")
    axes[0].grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # Right subplot: sample std dev vs E0
    # ---------------------------------------------------------
    axes[1].errorbar(
        stats_df['E0'], stats_df['std'],
        yerr=stats_df['std_err'],
        fmt='o', capsize=4
    )
    axes[1].set_xlabel(r"$E_0$ [GeV]")
    axes[1].set_ylabel(r"$\hat{\sigma}_{\rm samp}$ [GeV]")
    axes[1].set_title("Sample Std Dev vs True Energy")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    # Ensure directory exists and save
    save_dir = os.path.dirname(savepath)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(savepath)

    return fig, axes


# ----------------------------------------------------------
# 1. Bootstrap function for fitted curves
# ----------------------------------------------------------
def bootstrap_fit_results(e0_values, means, mean_errs, stds, std_errs,
                          fit_mean_fn, fit_width_fn, n_boot=500):
    
    #empty lists to store the bootstrap results
    lam_samples = []
    delta_samples = []
    a_samples = []
    b_samples = []
    c_samples = []

    rng = np.random.default_rng(123)

    for _ in range(n_boot):
        #create a new bootstrap dataset 
        # resample means + widths using Gaussian errors
        means_bs = rng.normal(means, mean_errs)
        stds_bs  = rng.normal(stds, std_errs)

        # fit mean model
        lam, delta, _, _ = fit_mean_fn(e0_values, means_bs, mean_errs)
        lam_samples.append(lam)
        delta_samples.append(delta)

        # fit width model on the bootstap dataset 
        a, b, c, _, _, _ = fit_width_fn(e0_values, stds_bs, std_errs)
        #save the parameters 
        a_samples.append(a)
        b_samples.append(b)
        c_samples.append(c)
    #return all bootstrap distributions  -> 500loops
    return (
        np.array(lam_samples), np.array(delta_samples),
        np.array(a_samples), np.array(b_samples), np.array(c_samples)
    )


# ----------------------------------------------------------
# 2. Produce Figure 1.4
# ----------------------------------------------------------
def plot_rescaled_with_fits_and_bands(stats_df,
                                      lambda_samp, delta_samp,
                                      a_samp, b_samp, c_samp,
                                      lam_samples, delta_samples,
                                      a_samples, b_samples, c_samples,
                                      savepath="figs/Figure1.4.pdf"):

    E0 = stats_df["E0"].values
    
    # rescaled data
    mean_bias        = stats_df["mean"].values - E0
    mean_bias_err    = stats_df["mean_err"].values

    sigma_rel        = stats_df["std"].values / E0
    sigma_rel_err    = stats_df["std_err"].values / E0

    # smooth x for curves
    E0_smooth = np.linspace(E0.min(), E0.max(), 300)

    # fitted curves
    mean_bias_curve = (lambda_samp - 1) * E0_smooth + delta_samp
    sigma_rel_curve = np.sqrt(
        (a_samp / np.sqrt(E0_smooth))**2 +
        (b_samp / E0_smooth)**2 +
        c_samp**2
    )

    # bootstrap bands
    mean_bias_band_low = []
    mean_bias_band_high = []
    sigma_rel_band_low = []
    sigma_rel_band_high = []

    for e in E0_smooth:
        # evaluate bootstrap mean model
        bias_samples = (lam_samples - 1) * e + delta_samples
        mean_bias_band_low.append(np.percentile(bias_samples, 16))
        mean_bias_band_high.append(np.percentile(bias_samples, 84))

        # evaluate bootstrap width model
        width_samples = np.sqrt(
            (a_samples / np.sqrt(e))**2 +
            (b_samples / e)**2 +
            c_samples**2
        )
        sigma_rel_band_low.append(np.percentile(width_samples, 16))
        sigma_rel_band_high.append(np.percentile(width_samples, 84))

    # ------------------------------------------------------
    # Plot
    # ------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # ------------------ LEFT: Mean Bias ------------------
    ax[0].errorbar(E0, mean_bias, yerr=mean_bias_err, fmt='o', label="Data")
    ax[0].plot(E0_smooth, mean_bias_curve, 'r-', label="Fit")

    ax[0].fill_between(
        E0_smooth, mean_bias_band_low, mean_bias_band_high,
        color='r', alpha=0.3, label="±1σ band"
    )

    ax[0].set_xlabel("$E_0$ [GeV]")
    ax[0].set_ylabel("$\\hat{\\mu} - E_0$ [GeV]")
    ax[0].set_title("Mean Bias vs $E_0$")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    # ------------------ RIGHT: Relative Width ------------------
    ax[1].errorbar(E0, sigma_rel, yerr=sigma_rel_err, fmt='o', label="Data")
    ax[1].plot(E0_smooth, sigma_rel_curve, 'r-', label="Fit")

    ax[1].fill_between(
        E0_smooth, sigma_rel_band_low, sigma_rel_band_high,
        color='r', alpha=0.3, label="±1σ band"
    )

    ax[1].set_xlabel("$E_0$ [GeV]")
    ax[1].set_ylabel("$\\hat{\\sigma}/E_0$")
    ax[1].set_title("Relative Width vs $E_0$")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    fig.tight_layout()
    save_dir = os.path.dirname(savepath)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(savepath)

    return fig, ax
