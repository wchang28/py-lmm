from abc import ABC, abstractmethod
import numpy as np


class Correlation(ABC):
    """Abstract class providing interface for a correlation matrix."""

    @abstractmethod
    def compute(self):
        """Computes correlation matrix. Needs to be called before other methods."""
        pass

    @abstractmethod
    def correlation(self):
        """Returns correlation matrix as 2D array."""
        pass

    @abstractmethod
    def corr_sqrt(self):
        """Returns pseudo-square root of correlation matrix as 2D array."""
        pass


class ReducedFactorCorrelation(Correlation):
    """Implements n-dimensional reduced-factor correlation matrix.
    Correlation matrix will be of rank F, where F is the given number of factors.
    """

    """Args:

    correlation: Correlation matrix as 2D array to approximate.
    num_factors: Number of factors to use for approximation.
    """
    def __init__(self, correlation, num_factors):
        self._correlation = correlation
        self._num_factors = num_factors

        corr_dim = correlation.shape[0]
        self._corr_sqrt = np.zeros((corr_dim, num_factors))
        self._corr_reduced = np.zeros(correlation.shape)

    @property
    def correlation(self):
        """Returns approximation to correlation matrix with rank F."""
        return self._corr_reduced

    @property
    def corr_sqrt(self):
        """Returns (n, F)-dimensional pseudo-square root of correlation matrix."""
        return self._corr_sqrt

    def compute(self):
        """Computes a rank F approximation of given correlation matrix."""

        # Compute eigenvalues and vectors from correlation matrix
        eigvals, eigvecs = np.linalg.eig(self._correlation)

        # Sort eigenvalues and corresponding eigenvectors descendingly
        sorted_indices = eigvals.argsort()[::-1][:self._num_factors]
        eigvals = eigvals[sorted_indices]
        eigvecs = eigvecs.T[sorted_indices]

        # Create covariance matrix with reduced rank
        sqrt = np.column_stack([np.sqrt(l)*v for l, v in zip(eigvals, eigvecs)])
        covariance = sqrt @ sqrt.T

        # Create correlation matrix with reduced rank by normalizing covariance matrix
        normalize = lambda i, j: sqrt[i, j]/np.sqrt(covariance[i, i])
        self._corr_sqrt = np.fromfunction(normalize, self._corr_sqrt.shape, dtype=int)
        self._corr_reduced = self._corr_sqrt @ self._corr_sqrt.T