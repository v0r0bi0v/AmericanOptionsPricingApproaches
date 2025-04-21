from abc import abstractmethod
import numpy as np


class PricerAbstract:
    @abstractmethod
    def price(
        self, 
        *args, 
        **kwargs
    ) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def plot_expected_prices(self):
        raise NotImplementedError
    
    @abstractmethod
    def plot_sample(self):
        raise NotImplementedError
