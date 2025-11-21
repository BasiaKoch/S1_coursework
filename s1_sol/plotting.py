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

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

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
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))

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


def plot_individual_fits_overlaid(data_dict, indiv_fits_df, savepath='figs/Figure2.1.pdf'):
    """
    Plot individual normal fits overlaid on residual histograms (Figure 2.1)

    Parameters
    ----------
    data_dict : dict
        Dictionary mapping E0 -> array of E measurements
    indiv_fits_df : pd.DataFrame
        Results from fit_individual_energies with columns: E0, mu, sigma
    savepath : str
        Path to save the figure

    Returns
    -------
    fig, axes : matplotlib Figure and Axes
    """
    from scipy.stats import norm

    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))

    e0_values = sorted(data_dict.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(e0_values)))

    bin_edges = np.linspace(-15, 15, 50)
    bin_width = bin_edges[1] - bin_edges[0]
    x_curve = np.linspace(-15, 15, 200)

    # LEFT: Overlaid histograms with fitted curves
    for i, e0 in enumerate(e0_values):
        residuals = data_dict[e0] - e0
        ax[0].hist(residuals, bins=bin_edges, alpha=0.4, color=colors[i],
                   label=f'$E_0$ = {e0} GeV', edgecolor='none')

        row = indiv_fits_df[indiv_fits_df['E0'] == e0].iloc[0]
        mu_res = row['mu'] - e0
        y_curve = len(residuals) * bin_width * norm.pdf(x_curve, mu_res, row['sigma'])
        ax[0].plot(x_curve, y_curve, color=colors[i], linewidth=2)

    ax[0].set_xlabel('$E - E_0$ [GeV]')
    ax[0].set_ylabel('Counts')
    ax[0].legend(fontsize=8)
    ax[0].grid(True, alpha=0.3)

    # RIGHT: Combined with sum of fits
    all_residuals = np.concatenate([data_dict[e0] - e0 for e0 in e0_values])
    ax[1].hist(all_residuals, bins=bin_edges, alpha=0.7, color='gray',
               edgecolor='black', label='All data')

    y_total = np.zeros_like(x_curve)
    for e0 in e0_values:
        row = indiv_fits_df[indiv_fits_df['E0'] == e0].iloc[0]
        mu_res = row['mu'] - e0
        y_total += len(data_dict[e0]) * bin_width * norm.pdf(x_curve, mu_res, row['sigma'])

    ax[1].plot(x_curve, y_total, 'r-', linewidth=2, label='Sum of fits')
    ax[1].set_xlabel('$E - E_0$ [GeV]')
    ax[1].set_ylabel('Counts')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_dir = os.path.dirname(savepath)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(savepath)

    return fig, ax


def plot_simultaneous_fit_with_bands(data_dict, params, errors,
                                      boot_lb, boot_dE, boot_a, boot_b, boot_c,
                                      savepath="figs/Figure3.1.pdf"):
    """
    Plot simultaneous fit curves with bootstrap error bands (Figure 3.1)

    Parameters
    ----------
    data_dict : dict
        Dictionary mapping E0 -> array of E measurements
    params : dict
        Fitted parameters {'lb', 'dE', 'a', 'b', 'c'}
    errors : dict
        Parameter uncertainties {'lb', 'dE', 'a', 'b', 'c'}
    boot_lb, boot_dE, boot_a, boot_b, boot_c : np.ndarray
        Bootstrap samples for each parameter (length n_boot)
    savepath : str
        Path to save the figure

    Returns
    -------
    fig, axes : matplotlib Figure and array of Axes
    """

    # Extract central values
    lam = params["lb"]
    delta = params["dE"]
    a = params["a"]
    b = params["b"]
    c = params["c"]

    # Get E0 values and compute sample statistics
    E0_vals = np.array(sorted(data_dict.keys()))
    means = np.array([np.mean(data_dict[e0]) for e0 in E0_vals])
    stds = np.array([np.std(data_dict[e0], ddof=1) for e0 in E0_vals])

    # Smooth x-axis for curves
    E0_smooth = np.linspace(E0_vals.min(), E0_vals.max(), 300)

    # --- Central fitted curves ---
    mu_fit = lam * E0_smooth + delta
    sigma_fit = E0_smooth * np.sqrt((a/np.sqrt(E0_smooth))**2 + (b/E0_smooth)**2 + c**2)

    # Rescaled for plotting
    mu_rescaled = mu_fit - E0_smooth
    sigma_rescaled = sigma_fit / E0_smooth

    # --- Bootstrap bands ---
    n_boot = len(boot_lb)
    mu_band = np.zeros((n_boot, len(E0_smooth)))
    sigma_band = np.zeros((n_boot, len(E0_smooth)))

    for k in range(n_boot):
        mu_k = boot_lb[k] * E0_smooth + boot_dE[k]
        mu_band[k] = mu_k - E0_smooth

        sigma_k = E0_smooth * np.sqrt(
            (boot_a[k]/np.sqrt(E0_smooth))**2 +
            (boot_b[k]/E0_smooth)**2 +
            boot_c[k]**2
        )
        sigma_band[k] = sigma_k / E0_smooth

    mu_lo, mu_hi = np.percentile(mu_band, [16, 84], axis=0)
    sigma_lo, sigma_hi = np.percentile(sigma_band, [16, 84], axis=0)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: μ_E - E0
    axes[0].errorbar(E0_vals, means - E0_vals, fmt='o', label='Data', capsize=3)
    axes[0].plot(E0_smooth, mu_rescaled, 'r-', lw=2, label='Fit')
    axes[0].fill_between(E0_smooth, mu_lo, mu_hi, color='r', alpha=0.3, label='±1σ band')
    axes[0].set_xlabel(r'$E_0$ [GeV]')
    axes[0].set_ylabel(r'$\mu_E - E_0$ [GeV]')
    axes[0].set_title('Simultaneous Fit: Mean Bias')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right panel: σ_E / E0
    axes[1].errorbar(E0_vals, stds / E0_vals, fmt='o', label='Data', capsize=3)
    axes[1].plot(E0_smooth, sigma_rescaled, 'r-', lw=2, label='Fit')
    axes[1].fill_between(E0_smooth, sigma_lo, sigma_hi, color='r', alpha=0.3, label='±1σ band')
    axes[1].set_xlabel(r'$E_0$ [GeV]')
    axes[1].set_ylabel(r'$\sigma_E / E_0$')
    axes[1].set_title('Simultaneous Fit: Relative Width')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    # Save
    save_dir = os.path.dirname(savepath)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(savepath)

    return fig, axes


def plot_parameter_comparison(ls_params, ls_errors,
                              indiv_params, indiv_errors,
                              sim_params, sim_errors,
                              savepath="figs/Figure3.2.pdf"):

    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # Parameters must appear in consistent order
    keys = ["lb", "dE", "a", "b", "c"]
    display = [r"$\lambda$", r"$\Delta$", "a", "b", "c"]

    x = np.arange(len(keys))
    dx = 0.20   # horizontal offset between methods

    fig, ax = plt.subplots(figsize=(10, 5))

    def vals(d): return [d[k] for k in keys]
    def errs(d): return [d[k] for k in keys]

    # --- Sample LS ---
    ax.errorbar(
        x - dx,
        vals(ls_params),
        yerr=errs(ls_errors),
        fmt="o", color="C0", capsize=4, label="Sample LS"
    )

    # --- Individual MLE ---
    ax.errorbar(
        x,
        vals(indiv_params),
        yerr=errs(indiv_errors),
        fmt="o", color="C1", capsize=4, label="Individual MLE"
    )

    # --- Simultaneous MLE ---
    ax.errorbar(
        x + dx,
        vals(sim_params),
        yerr=errs(sim_errors),
        fmt="o", color="C2", capsize=4, label="Simultaneous MLE"
    )

    # Cosmetics
    ax.set_xticks(x)
    ax.set_xticklabels(display, fontsize=12)
    ax.set_ylabel("Parameter value", fontsize=12)
    ax.set_title("Comparison of Parameter Estimates", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    # Save
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, bbox_inches="tight")

    return fig, ax


####4.1




def plot_bootstrap_histograms(boot_ls, boot_ind, boot_sim,
                              savepath="figs/Figure4.1.pdf"):
    """
    Figure 4.1:
      For each parameter {lb, dE, a, b, c} plot histograms of
      the bootstrap distributions for all three methods.

    Parameters
    ----------
    boot_ls, boot_ind, boot_sim : dict
        Output of bootstrap_all_methods. Each maps
        "lb", "dE", "a", "b", "c" -> 1D np.ndarray of length n_boot.
    savepath : str
        Where to save the figure.
    """

    keys    = ["lb", "dE", "a", "b", "c"]
    labels  = [r"$\lambda$", r"$\Delta$", "a", "b", "c"]

    # Use 2x3 layout as specified in instructions
    fig, axes = plt.subplots(2, 3, figsize=(19.2, 9.6))

    # Hide the upper-right subplot (position [0,2])
    axes[0, 2].set_visible(False)

    # Map parameters to subplot positions: (0,0), (0,1), (1,0), (1,1), (1,2)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]

    for i, (k, lab) in enumerate(zip(keys, labels)):
        row, col = positions[i]
        ax = axes[row, col]

        vals_ls  = np.asarray(boot_ls[k])
        vals_ind = np.asarray(boot_ind[k])
        vals_sim = np.asarray(boot_sim[k])

        all_vals = np.concatenate([vals_ls, vals_ind, vals_sim])
        bins = np.linspace(all_vals.min(), all_vals.max(), 40)

        ax.hist(vals_ls,  bins=bins, histtype="step", density=True,
                color="C0", label="Sample LS", linewidth=1.5)
        ax.hist(vals_ind, bins=bins, histtype="step", density=True,
                color="C1", label="Individual MLE", linewidth=1.5)
        ax.hist(vals_sim, bins=bins, histtype="step", density=True,
                color="C2", label="Simultaneous MLE", linewidth=1.5)

        ax.set_xlabel(lab, fontsize=12)
        ax.grid(alpha=0.3)

        if i == 0:
            ax.set_ylabel("Density", fontsize=12)

    # Add legend to the last subplot
    axes[1, 2].legend(loc="best", fontsize=10)

    fig.tight_layout()
    save_dir = os.path.dirname(savepath)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(savepath, bbox_inches="tight")

    return fig, axes



#4.2

def plot_bootstrap_comparison(ls_params, ls_errors,
                              indiv_params, indiv_errors,
                              sim_params, sim_errors,
                              boot_ls, boot_ind, boot_sim,
                              savepath="figs/Figure4.2.pdf"):
    """
    Figure 4.2:
      Same style as Figure 3.2, but overlay:
        - original estimates + errors (parts 1–3)
        - bootstrap mean ± std from part 4.

    Parameters
    ----------
    ls_params, ls_errors : dict
        Sample LS central values and 1σ errors {"lb","dE","a","b","c"}.
    indiv_params, indiv_errors : dict
        Individual MLE → LS results.
    sim_params, sim_errors : dict
        Simultaneous MLE results.
    boot_ls, boot_ind, boot_sim : dict
        Output of bootstrap_all_methods.
    """

    keys    = ["lb", "dE", "a", "b", "c"]
    labels  = [r"$\lambda$", r"$\Delta$", "a", "b", "c"]

    x = np.arange(len(keys))
    dx = 0.20  # horizontal offset per method

    fig, ax = plt.subplots(figsize=(10, 5))

    def vals(d): return [d[k] for k in keys]
    def errs(d): return [d[k] for k in keys]

    # Bootstrap means + stds
    def boot_stats(boot_dict):
        means = [np.mean(boot_dict[k]) for k in keys]
        stds  = [np.std(boot_dict[k], ddof=1) for k in keys]
        return means, stds

    ls_boot_mean,   ls_boot_std   = boot_stats(boot_ls)
    ind_boot_mean,  ind_boot_std  = boot_stats(boot_ind)
    sim_boot_mean,  sim_boot_std  = boot_stats(boot_sim)

    # ---------------- Sample LS ----------------
    x_ls = x - dx
    ax.errorbar(x_ls, vals(ls_params), yerr=errs(ls_errors),
                fmt="o", color="C0", capsize=4,
                label="Sample LS (orig)")
    ax.errorbar(x_ls, ls_boot_mean, yerr=ls_boot_std,
                fmt="^", color="C0", capsize=4, alpha=0.8,
                label="Sample LS (boot)")

    # ---------------- Individual MLE ----------------
    x_ind = x
    ax.errorbar(x_ind, vals(indiv_params), yerr=errs(indiv_errors),
                fmt="o", color="C1", capsize=4,
                label="Individual MLE (orig)")
    ax.errorbar(x_ind, ind_boot_mean, yerr=ind_boot_std,
                fmt="^", color="C1", capsize=4, alpha=0.8,
                label="Individual MLE (boot)")

    # ---------------- Simultaneous MLE ----------------
    x_sim = x + dx
    ax.errorbar(x_sim, vals(sim_params), yerr=errs(sim_errors),
                fmt="o", color="C2", capsize=4,
                label="Simultaneous MLE (orig)")
    ax.errorbar(x_sim, sim_boot_mean, yerr=sim_boot_std,
                fmt="^", color="C2", capsize=4, alpha=0.8,
                label="Simultaneous MLE (boot)")

    # Cosmetics
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Parameter value", fontsize=12)
    ax.set_title("Parameter Estimates: Original vs Bootstrap", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=2, fontsize=9)

    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, bbox_inches="tight")

    return fig, ax
