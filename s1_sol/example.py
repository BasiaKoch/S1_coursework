"""Dummy functionality of s1_sol"""

import numpy as np
import matplotlib.pyplot as plt
import json
import requests


def hello_world():
    """A function to print the Hello World message"""

    print("Hello world!")


def make_me_a_plot(size, bins=50):
    """
    A a function to demonstrate making a plot

    Parameters
    ----------
    size : int
        Size of random sample to produce
    bins : int, optional
        Number of bins for the plotted histogram

    Returns
    -------
    fig, ax
        The matplotlib Figure and Axes
    """

    x = np.random.normal(size=size)

    fig, ax = plt.subplots()
    ax.hist(x, bins=bins)

    return fig, ax


def make_results_json(filename):
    """
    Make an example of the results.json output file

    Parameters
    ----------
    filename : str
        Path to save the file to

    """

    ex_dict = {
        "sample_ests": {
            "values": {
                "lb": np.nan,  # estimated value of lambda
                "dE": np.nan,  # estimated value of DeltaE
                "a": np.nan,  # estimated value of a
                "b": np.nan,  # estimated value of b
                "c": np.nan,  # estimated value of c
            },
            "errors": {
                "lb": np.nan,  # estimated error of lambda
                "dE": np.nan,  # estimated error of DeltaE
                "a": np.nan,  # estimated error of a
                "b": np.nan,  # estimated error of b
                "c": np.nan,  # estimated error of c
            },
        },
        "individual_fits": {
            "values": {
                "lb": np.nan,  # estimated value of lambda
                "dE": np.nan,  # estimated value of DeltaE
                "a": np.nan,  # estimated value of a
                "b": np.nan,  # estimated value of b
                "c": np.nan,  # estimated value of c
            },
            "errors": {
                "lb": np.nan,  # estimated error of lambda
                "dE": np.nan,  # estimated error of DeltaE
                "a": np.nan,  # estimated error of a
                "b": np.nan,  # estimated error of b
                "c": np.nan,  # estimated error of c
            },
        },
        "simultaneous_fit": {
            "values": {
                "lb": np.nan,  # estimated value of lambda
                "dE": np.nan,  # estimated value of DeltaE
                "a": np.nan,  # estimated value of a
                "b": np.nan,  # estimated value of b
                "c": np.nan,  # estimated value of c
            },
            "errors": {
                "lb": np.nan,  # estimated error of lambda
                "dE": np.nan,  # estimated error of DeltaE
                "a": np.nan,  # estimated error of a
                "b": np.nan,  # estimated error of b
                "c": np.nan,  # estimated error of c
            },
        },
    }
    with open(filename, "w") as f:
        json.dump(ex_dict, f, indent=4)

