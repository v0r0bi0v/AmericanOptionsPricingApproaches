from pricers.lspi import LSPIPricer
from pricers.american_monte_carlo import AmericanMonteCarloPricer
from pricers.binomial_tree import BinomialTreePricer

NAME_TO_PRICER = {
    "LSPIPricer": LSPIPricer,
    "AmericanMonteCarloPricer": AmericanMonteCarloPricer,
    "BinomialTreePricer": BinomialTreePricer
}
