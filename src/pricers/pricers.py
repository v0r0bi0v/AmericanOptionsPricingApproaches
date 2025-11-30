from src.pricers.lspi_book import LSPIBookPricer
from src.pricers.lspi_semi_gradient import LSPISemiGradientPricer
from src.pricers.lspi_full_gradient import LSPIFullGradientPricer
from src.pricers.lspi_n_steps import LSPINStepsPricer
from src.pricers.american_monte_carlo import AmericanMonteCarloPricer
from src.pricers.binomial_tree import BinomialTreePricer
from src.pricers.fqi_pricer import FQIPricerDiploma

from src.pricers.lspi_orig_diploma import LSPIPricerDiploma

NAME_TO_PRICER = {
    "FQIPricerDiploma": FQIPricerDiploma,
    "LSPIPricerDiploma" : LSPIPricerDiploma,
    "LSPIBookPricer": LSPIBookPricer,
    "LSPISemiGradientPricer": LSPISemiGradientPricer,
    "AmericanMonteCarloPricer": AmericanMonteCarloPricer,
    "BinomialTreePricer": BinomialTreePricer
}
