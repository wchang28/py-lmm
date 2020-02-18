from abc import ABC, abstractmethod
import numpy as np


class InterestRateModel(ABC):
    """Abstract class providing interface for a interest rate model."""

    @abstractmethod
    def rate(self, rateindex, timeindex):
        """Returns 1D array containing all realizations of simulation for given
        forward rate and time.
        """
        pass

    @abstractmethod
    def numeraire(self, rateindex, timeindex):
        """Returns 1D array containing numeraires corresponding to realizations
        of simulation for given forward rate and time.
        """
        pass

    @abstractmethod
    def simulate(self):
        """Simulates forward rates. Needs to be called before other methods."""
        pass


class LiborMarketModel(InterestRateModel):
    """Implements LIBOR market model using an Euler-Maruyama scheme."""
    """Args:

    mc_config: Instance of MonteCarloConfig class.
    timestruct: Instance of TimeStructure class containing times
    at which forward rates should be simulated.
    volatility: Instance implementing Volatility interface.
    correlation: Instance implementing Correlation interface.
    forwardcurve: Initial forward curve as 1D array of floats.
    """
    def __init__(self, mc_config, timestruct, volatility, correlation, forwardcurve):
        self._timestruct = timestruct
        self._volatility = volatility
        self._correlation = correlation
        self._forwardcurve = forwardcurve
        self._mc_config = mc_config

        shape = (timestruct.num_times - 1, timestruct.num_times, mc_config.num_paths)
        self._rates = np.zeros(shape)
        self._numeraire = np.zeros(shape)

    def rate(self, rateindex, timeindex):
        """Returns realizations for given forward rate and time."""
        return self._rates[rateindex, timeindex]

    def numeraire(self, rateindex, timeindex):
        """Returns numeraire for given forward rate and time."""
        return self._numeraire[rateindex, timeindex]

    def simulate(self):
        """Simulates forward rates using Euler-Maruyama time-stepping procedure."""

        # Pre-compute standard normal increments for all time steps and rates
        mean = np.zeros(self._mc_config.num_factors)
        cov = np.identity(self._mc_config.num_factors)
        increments = np.random.multivariate_normal(mean, cov, (self._timestruct.num_times, self._mc_config.num_paths))

        # Declare variables for convenience
        num_rates = self._timestruct.num_times - 1
        rho = self._correlation.correlation
        vols = self._volatility.volatility
        tau = self._timestruct.tenors
        sqrt_tau = np.sqrt(tau)

        # Simulate forward rates
        for k in range(self._mc_config.num_paths):
            L = self._rates[:, :, k]
            P = self._numeraire[:, :, k]

            L[:, 0] = self._forwardcurve
            P[:, 0] = np.cumprod(1/(1 + tau*L[:, 0]))

            for j in range(1, self._timestruct.num_times):
                # Compute drift
                drift = np.zeros(num_rates)
                for m in range(j+1, num_rates):
                    drift[m] = ((rho[m, j]*tau[m-1]*L[m, j-1]*vols[m, j])/(1 + tau[m-1]*L[m, j-1]))

                # Compute forward rates and numeraire
                for i in range(j, num_rates):
                    dW = self._correlation.corr_sqrt[i, :] @ increments[j, k]

                    L[i, j] = L[i, j-1] * np.exp((-vols[i, j] * np.sum(drift[i+1:]) - 0.5 * vols[i, j]**2) * tau[j] + vols[i, j] * sqrt_tau[j] * dW)
                    P[i, j] = (1 if i==j else P[i-1, j])/(1 + tau[j]*L[i, j])

