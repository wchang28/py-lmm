import numpy as np
from utilities import TimeStructure, MonteCarloConfig
from volatility import PiecewiseConstVolatility
from correlation import ReducedFactorCorrelation
from interestratemodel import LiborMarketModel
from product import Cap, RatchetFloater


# Set-up hypothetical market data
termstruct = TimeStructure([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
forwardcurve = np.array([0.0112, 0.0118, 0.0123, 0.0127, 0.0132, 0.0137, 0.0145, 0.0154, 0.0163, 0.0174])
capletvolas = np.array([0.2366, 0.2487, 0.2573, 0.2564, 0.2476, 0.2376, 0.2252, 0.2246, 0.2223])

# Correlation matrix generated by using parametric form with beta = 0.2
# Full correlation matrix initialization removed due to space restrictions
# - CODE WILL NOT WORK WITHOUT FULL CORRELATION MATRIX -
correlation_matrix = np.array([[1, 0.904837, ...], ..., [..., 0.904837, 1]])

# Simulate forward rates with 100000 sample paths and 4 factors
config = MonteCarloConfig(10, 4)

# Initialize LIBOR market model
volatility = PiecewiseConstVolatility(termstruct, capletvolas)
correlation = ReducedFactorCorrelation(correlation_matrix, config.num_factors)
libor_marketmodel = LiborMarketModel(config, termstruct, volatility, correlation, forwardcurve)

volatility.calibrate()
correlation.compute()
libor_marketmodel.simulate()

# Price cap with cap rate 1.1% and notional of 10000000
cap = Cap(0.0110, 10000000, termstruct)
cap_price = cap.price(libor_marketmodel)

# Price ratchet floater with spreads both 0.15%, fixed rate 0.01%
# and notional of 10000000
ratchetfloater = RatchetFloater(0.0015, 0.0015, 0.0001, 10000000, termstruct)
ratchetfloater_price = ratchetfloater.price(libor_marketmodel)