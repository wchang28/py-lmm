import numpy as np


class TimeStructure:
    """Class representing a discrete set of points in time T_0, T_1, ..., T_n."""

    """Args:

    times: List of floats representing points in time with unit [1/year].
    """
    def __init__(self, times):
        self._times = np.array(times)

    @property
    def num_times(self):
        """Number of points in time."""
        return self._times.size

    @property
    def tenors(self):
        """Returns 1D array [T_1 - T_0, T_2 - T_1, ..., T_n - T_n-1]."""
        return np.diff(self._times)

    def time(self, timeindex):
        """Returns time value for index i"""
        return self._times[timeindex]

    def tenor(self, timeindex):
        """Returns length of the time period [T_i, T_i+1] for index i."""
        return self._times[timeindex + 1] - self._times[timeindex]


class MonteCarloConfig:
    """Container class bundling configuration for a Monte Carlo simulation."""

    """Args:
    num_paths: Number of paths in Monte Carlo simulation.
    num_factors: Number of factors for correlation matrix in Monte Carlo simulation.
    """
    def __init__(self, num_paths, num_factors):
        self._num_paths = num_paths
        self._num_factors = num_factors

    @property
    def num_paths(self):
        return self._num_paths

    @property
    def num_factors(self):
        return self._num_factors