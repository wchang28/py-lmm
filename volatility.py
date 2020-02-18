from abc import ABC, abstractmethod
import numpy as np


class Volatility(ABC):
    """Abstract class providing interface for volatility functions."""

    @abstractmethod
    def volatility(self):
        """Returns 2D array representing volatility functions.
        First index is forward rate, second index is time.
        """
        pass

    @abstractmethod
    def calibrate(self):
        """Calibrates to market data. Needs to be called before other methods."""
        pass

class PiecewiseConstVolatility(Volatility):
    """Implements time-homogeneous, piecewise constant volatility functions.
    The volatility functions will be calibrated to implied caplet volatilities.
    """
    """Args:
    timestruct: Instance of TimeStructure class containing times
    at which volatility functions should be computed.
    capletvolas: 1D array containing caplet volatilities for calibration.
    """
    def __init__(self, timestruct, capletvolas):
        self._timestruct = timestruct
        self._capletvolas = capletvolas

        num_rates = timestruct.num_times - 1
        self._volas = np.zeros((num_rates, timestruct.num_times))

    @property
    def volatility(self):
        """Returns 2D array of piecewise constant volatilities."""
        return self._volas

    def calibrate(self):
        """Calibrates volatility functions to given caplet volatilities."""

        # Bootstrap caplet volatilities
        T_1 = self._timestruct.time(1)
        bs_volas = []

        for i, capletvola in enumerate(self._capletvolas, 2):
            bs_vola = capletvola**2 * self._timestruct.time(i-1)

            for j, vola in enumerate(bs_volas, 2):
                bs_vola -= vola**2 * self._timestruct.tenor(j-1)

            bs_volas.append(np.sqrt(bs_vola/T_1))

        # Insert bootstrapped volatilities into 2D array
        self._volas[1:, 1] = bs_volas
        for i in range(2, self._timestruct.num_times):
            self._volas[i:, i] = bs_volas[:-i+1]