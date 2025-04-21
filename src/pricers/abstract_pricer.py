from abc import abstractmethod
from utils.plotting_utils import plot_trajectories
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
