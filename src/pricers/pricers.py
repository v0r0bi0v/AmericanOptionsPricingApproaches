from src.pricers.lspi_book import LSPIBookPricer
from src.pricers.lspi_semi_gradient import LSPISemiGradientPricer
from src.pricers.lspi_full_gradient import LSPIFullGradientPricer
from src.pricers.lspi_n_steps import LSPINStepsPricer
from src.pricers.american_monte_carlo import AmericanMonteCarloPricer
from src.pricers.binomial_tree import BinomialTreePricer

NAME_TO_PRICER = {
    "LSPIBookPricer": LSPIBookPricer,
    "LSPISemiGradientPricer": LSPISemiGradientPricer,
    "AmericanMonteCarloPricer": AmericanMonteCarloPricer,
    "BinomialTreePricer": BinomialTreePricer
}
