from abc import ABC, abstractmethod
import numpy as np

class InterestRateProduct(ABC):
    """Abstract class providing interface interest rate product."""

    @abstractmethod
    def price(self, ratemodel):
        """Returns price(s) using the given interest rate model."""
        pass

class Cap(InterestRateProduct):
    """Implementation of a cap."""

    """Args:

    caprate: Cap rate.
    notional: Notional of cap.
    timestruct: Instance of TimeStructure class containing cap maturities.
    """
    def __init__(self, caprate, notional, timestruct):
        self._caprate = caprate
        self._notional = notional
        self._timestruct = timestruct

    def price(self, ratemodel):
        """Returns price of cap"""
        N = self._notional
        K = self._caprate

        caplet_prices = []
        num_rates = self._timestruct.num_times - 1
        for i in range(1, num_rates):
            tau = self._timestruct.tenor(i-1)
            L = ratemodel.rate(i, i)
            P = ratemodel.numeraire(num_rates-1, i+1)
            D = ratemodel.numeraire(num_rates-1, 0)

            payoff = tau * N * (L - K)
            payoff[payoff < 0] = 0.0 # Maximum with 0

            # Compute expectation under numeraire measure
            caplet_price = np.mean(D*payoff/(1 if i==num_rates-1 else P),
            dtype=np.float64)

            caplet_prices.append(caplet_price)

        return np.sum(caplet_prices)


class RatchetFloater(InterestRateProduct):
    """Implementation of a ratchet floater"""

    """Args:

    X: Constant spread for forward rate.
    Y: Constant spread for coupons.
    alpha: Fixed cap.
    notional: Notional of ratchet floater.
    timestruct: Instance of TimeStructure class containing forward rate maturities.
    """
    def __init__(self, X, Y, alpha, notional, timestruct):
        self._X = X
        self._Y = Y
        self._alpha = alpha
        self._notional = notional
        self._timestruct = timestruct

    def price(self, ratemodel):
        """Returns price of ratchet floater"""
        N = self._notional
        X = self._X
        Y = self._Y
        alpha = self._alpha

        num_rates = self._timestruct.num_times - 1
        coupons = []
        payoffs = []
        for i in range(1, num_rates):
            tau = self._timestruct.tenor(i-1)
            L = ratemodel.rate(i, i)
            P = ratemodel.numeraire(num_rates-1, i+1)
            D = ratemodel.numeraire(num_rates-1, 0)

            if i == 1:
                coupon = tau * (L + Y)
            else:
                coupon = tau * (L + Y) - coupons[i-2]
                coupon[coupon < 0] = 0.0 # Maximum with 0
                coupon[coupon > alpha] = alpha # Minimum with alpha
                coupon = coupons[i-2] + coupon

            coupons.append(coupon)

            payoff = N * (tau * (L + X) - coupon)
            # Compute expectation under numeraire measure
            payoffs.append(np.mean(D*payoff/(1 if i==num_rates-1 else P), dtype=np.float64))

        return np.sum(payoffs)