from src.pricers.lspi import LSPIPricer
from src.pricers.american_monte_carlo import AmericanMonteCarloPricer
from src.pricers.binomial_tree import BinomialTreePricer

NAME_TO_PRICER = {
    "LSPIPricer": LSPIPricer,
    "AmericanMonteCarloPricer": AmericanMonteCarloPricer,
    "BinomialTreePricer": BinomialTreePricer
}
