import numpy as np
import matplotlib.pyplot as plt
from warnings import warn

from src.pricers.abstract_pricer import PricerAbstract
from src.samplers.abstract_sampler import SamplerAbstract


class BinomialTreePricer(PricerAbstract):
    def __init__(self, sampler: SamplerAbstract):
        self.sampler = sampler
        
        if type(sampler).__name__ != "GeometricBrownianMotionPutSampler":
            raise ValueError("Sampler must be a GeometricBrownianMotionPutSampler for BinomialTreePricer")

    
    def plot_expected_prices(self):
        pass

    def plot_sample(self):
        pass

    def _generate_tree(self):
        self._dt = self.sampler.time_grid[-1] / (self.sampler.cnt_times - 1)

        self._tree = np.zeros((self.sampler.cnt_times, self.sampler.cnt_times), dtype=float)

        det_comp = (self.sampler.r - 0.5 * self.sampler.sigma * self.sampler.sigma) * self._dt  
        stoc_comp = self.sampler.sigma * np.sqrt(self._dt)

        self._up = np.exp(det_comp + stoc_comp)
        self._down = np.exp(det_comp - stoc_comp)

        f = np.vectorize(
            lambda i, j: self._up**(j - i) * self._down**i if i <= j else 0
        )
        self._tree = self.sampler.asset0 * np.fromfunction(f, (self.sampler.cnt_times, self.sampler.cnt_times))

    
    def _plot(self, tree):
        plt.figure(figsize=(10, 6))
        
        for i in range(self.sampler.cnt_times):
            for j in range(i + 1):
                plt.plot(self.sampler.time_grid[i], tree[j, i], 'bo')
        
        label_added = False
        for i in range(self.sampler.cnt_times - 1):
            for j in range(i + 1):
                if not label_added:
                    plt.plot([self.sampler.time_grid[i], self.sampler.time_grid[i + 1]], [tree[j, i], tree[j, i + 1]], 'b-', alpha=0.5, label="tree")
                    label_added = True
                else:
                    plt.plot([self.sampler.time_grid[i], self.sampler.time_grid[i + 1]], [tree[j, i], tree[j, i + 1]], 'b-', alpha=0.5)
                plt.plot([self.sampler.time_grid[i], self.sampler.time_grid[i + 1]], [tree[j, i], tree[j + 1, i + 1]], 'b-', alpha=0.5)

        plt.plot([0, self.sampler.time_grid[-1]], [self.sampler.strike, self.sampler.strike], "--", color="red", label="strike")
        
        plt.legend()
        plt.title("Tree of asset prices")
        plt.xlabel("Time steps")
        plt.ylabel("Asset price")
        plt.grid(True)
        plt.show()


    def _price(self, payoff_function):
        p_tilde = (np.exp(self.sampler.r * self._dt) - self._down) / (self._up - self._down)  
        price = np.zeros((self.sampler.cnt_times, self.sampler.cnt_times))
        price[:, -1] = payoff_function(self._tree[:, -1])

        price_history = np.zeros(self.sampler.cnt_times)
        price_history[-1] = price[:, -1].mean()

        payoff = payoff_function(self._tree)
        for time_index in range(self.sampler.cnt_times - 2, -1, -1):
            for traj_index in range(time_index + 1):
                price[traj_index, time_index] = np.maximum(
                    payoff[traj_index, time_index],
                    np.exp(-self.sampler.r * self._dt) * (  
                        p_tilde * price[traj_index, time_index + 1] + 
                        (1 - p_tilde) * price[traj_index + 1, time_index + 1]
                    )
                )
            
            price_history[time_index] = price[:, time_index][
                self._tree[:, time_index] != 0.
            ].mean()

        return price[0, 0], price_history
    
    def price(self, test=False, quiet=False):
        if test:
            warn("Test mode is not suitable for BinomialTreePricer")
        self._generate_tree()
        if not quiet:
            self._plot(self._tree)
        return self._price(lambda x: np.maximum(0, self.sampler.strike - x))
