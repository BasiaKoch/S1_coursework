"""Data loading and preprocessing utilities for photon detector analysis"""

import pandas as pd
import numpy as np


def load_sample_data(filepath='sample.csv'):
    """
    Load calibration sample data from CSV file

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing columns 'E_true' and 'E_rec'

    Returns
    -------
    df : pd.DataFrame
        Full dataset with columns E_true (true energy) and E_rec (reconstructed energy)
    data_dict : dict
        Dictionary mapping E0 (true energy) -> numpy array of E (measured energies)
    e0_values : np.ndarray
        Sorted array of unique true energy values

    Examples
    --------
    >>> df, data_dict, e0_values = load_sample_data('../sample.csv')
    >>> print(f"Loaded {len(df)} measurements at {len(e0_values)} energy points")
    """
    # Load the CSV file
    df = pd.read_csv(filepath)

    # Validate columns
    required_cols = ['E_true', 'E_rec']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # Get unique energy points (sorted)
    e0_values = np.array(sorted(df['E_true'].unique()))

    # Group data by true energy
    data_dict = {}
    for e0 in e0_values:
        data_dict[e0] = df[df['E_true'] == e0]['E_rec'].values

    return df, data_dict, e0_values


def get_residuals(df):
    """
    Compute energy residuals (E_rec - E_true)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'E_true' and 'E_rec' columns

    Returns
    -------
    residuals : np.ndarray
        Array of residuals (E_rec - E_true)
    """
    return df['E_rec'].values - df['E_true'].values


def get_residuals_by_energy(data_dict):
    """
    Compute residuals for each energy point

    Parameters
    ----------
    data_dict : dict
        Dictionary mapping E0 -> array of E measurements

    Returns
    -------
    residuals_dict : dict
        Dictionary mapping E0 -> array of residuals (E - E0)
    """
    residuals_dict = {}
    for e0, e_measurements in data_dict.items():
        residuals_dict[e0] = e_measurements - e0

    return residuals_dict
