"""Manage results.json file for the assignment"""

import json
import numpy as np


class ResultsManager:
    """
    Helper class to build and manage the results.json file

    This class helps you progressively build up the results as you complete
    each section of the assignment.

    Examples
    --------
    >>> results = ResultsManager()
    >>> results.update('sample_ests',
    ...                {'lb': 1.05, 'dE': 2.3, 'a': 0.5, 'b': 0.3, 'c': 0.02},
    ...                {'lb': 0.01, 'dE': 0.1, 'a': 0.05, 'b': 0.03, 'c': 0.002})
    >>> results.save('results.json')
    """

    def __init__(self):
        """Initialize empty results structure"""
        self.results = {
            "sample_ests": {
                "values": {"lb": np.nan, "dE": np.nan, "a": np.nan, "b": np.nan, "c": np.nan},
                "errors": {"lb": np.nan, "dE": np.nan, "a": np.nan, "b": np.nan, "c": np.nan},
            },
            "individual_fits": {
                "values": {"lb": np.nan, "dE": np.nan, "a": np.nan, "b": np.nan, "c": np.nan},
                "errors": {"lb": np.nan, "dE": np.nan, "a": np.nan, "b": np.nan, "c": np.nan},
            },
            "simultaneous_fit": {
                "values": {"lb": np.nan, "dE": np.nan, "a": np.nan, "b": np.nan, "c": np.nan},
                "errors": {"lb": np.nan, "dE": np.nan, "a": np.nan, "b": np.nan, "c": np.nan},
            },
        }

    def update(self, method, param_dict, error_dict):
        """
        Update results for a given method

        Parameters
        ----------
        method : str
            One of 'sample_ests', 'individual_fits', 'simultaneous_fit'
        param_dict : dict
            Parameter values with keys {'lb', 'dE', 'a', 'b', 'c'}
            lb = lambda, dE = Delta
        error_dict : dict
            Parameter errors with keys {'lb', 'dE', 'a', 'b', 'c'}

        Examples
        --------
        >>> results = ResultsManager()
        >>> results.update('sample_ests',
        ...                {'lb': 1.05, 'dE': 2.3, 'a': 0.5, 'b': 0.3, 'c': 0.02},
        ...                {'lb': 0.01, 'dE': 0.1, 'a': 0.05, 'b': 0.03, 'c': 0.002})
        """
        if method not in self.results:
            raise ValueError(f"Method must be one of {list(self.results.keys())}")

        # Validate parameter keys
        required_keys = {'lb', 'dE', 'a', 'b', 'c'}
        if set(param_dict.keys()) != required_keys:
            raise ValueError(f"param_dict must have keys: {required_keys}")
        if set(error_dict.keys()) != required_keys:
            raise ValueError(f"error_dict must have keys: {required_keys}")

        # Update results
        self.results[method]['values'] = param_dict.copy()
        self.results[method]['errors'] = error_dict.copy()

    def save(self, filepath='results.json'):
        """
        Save results to JSON file

        Parameters
        ----------
        filepath : str
            Path where to save the JSON file
        """
        # Convert numpy types to Python native types for JSON serialization
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        with open(filepath, 'w') as f:
            json.dump(convert(self.results), f, indent=4)

        print(f"Results saved to {filepath}")

    def load(self, filepath='results.json'):
        """
        Load results from JSON file

        Parameters
        ----------
        filepath : str
            Path to the JSON file to load
        """
        with open(filepath, 'r') as f:
            self.results = json.load(f)

        print(f"Results loaded from {filepath}")

    def display(self):
        """Pretty print the current results"""
        print(json.dumps(self._convert_for_display(), indent=2))

    def _convert_for_display(self):
        """Convert results for display (handle NaN properly)"""
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, float) and np.isnan(obj):
                return "NaN"
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            else:
                return obj

        return convert(self.results)

    def get_values(self, method):
        """
        Get parameter values for a specific method

        Parameters
        ----------
        method : str
            One of 'sample_ests', 'individual_fits', 'simultaneous_fit'

        Returns
        -------
        values : dict
            Dictionary of parameter values
        """
        return self.results[method]['values']

    def get_errors(self, method):
        """
        Get parameter errors for a specific method

        Parameters
        ----------
        method : str
            One of 'sample_ests', 'individual_fits', 'simultaneous_fit'

        Returns
        -------
        errors : dict
            Dictionary of parameter errors
        """
        return self.results[method]['errors']
