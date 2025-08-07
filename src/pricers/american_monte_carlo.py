import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from IPython.display import display, clear_output
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

from src.pricers.abstract_pricer import PricerAbstract
from src.samplers.abstract_sampler import SamplerAbstract


def _plot_progress(sampler, bar, price_history, lower_bound, upper_bound, ax=None):
    clear_output(wait=True)
    display(bar.container)
    if ax is None:
        ax = plt.gca()
    ax.ticklabel_format(style='plain', useOffset=False)
    ax.plot(sampler.time_grid, price_history)
    ax.plot(sampler.time_grid, lower_bound, "--")
    ax.plot(sampler.time_grid, upper_bound, "-.")
    ax.legend()
    ax.set_title("$Option_t$")
    ax.set_xlabel("$t$")
    ax.set_ylabel("price")
    ax.grid()


@dataclass
class AmericanMonteCarloResult:
    price_history: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray

class AmericanMonteCarloPricer(PricerAbstract):
    def __init__(
            self,
            sampler: SamplerAbstract,
            degree: int = 3,
            regularization_alpha: float = 1e-4
    ):
        self.sampler = sampler
        self.regularization_alpha = regularization_alpha
        self.basis_functions_transformer = PolynomialFeatures(degree=degree)
        self.price_history: np.ndarray | None = None
        self.option_price: np.ndarray | None = None
        self.result = {}
        self.weights: list = []
        self.scalers: list = []

    def price(self, test=False, quiet=False, ax=None):
        self.sampler.sample()
        discounted_payoff = self.sampler.payoff * self.sampler.discount_factor
        # TODO: Вот так к сожалению в общем случае лучше не делать (если DiscountFactor стохастический)
        
        # if not quiet:
        #     self.sampler.plot(cnt=10, plot_mean=True, y="payoff, discount_factor, markov_state")

        # option_price - 1-d array of size (cnt_trajectories). Copies the last payoff from each trajectory
        self.option_price = discounted_payoff[:, -1].copy()

        if not test:
            self.weights = [None] * self.sampler.cnt_times
            self.scalers = [None] * self.sampler.cnt_times
        self.price_history = [None] * (self.sampler.cnt_times - 1) + [self.option_price.mean()]

        lower_bound = np.zeros(self.sampler.cnt_times)
        upper_bound = np.zeros(self.sampler.cnt_times)
        for i in range(self.sampler.cnt_times):
            lower_bound[i] = discounted_payoff[:, i:].mean(axis=0).max()
            upper_bound[i] = discounted_payoff[:, i:].max(axis=1).mean()

        bar = tqdm(range(self.sampler.cnt_times - 2, -1, -1), desc=f"AMC price {'test' if test else 'train'}")
        for time_index in bar:
            if time_index == 0:
                continuation_value = np.ones(self.sampler.cnt_trajectories) * np.mean(self.option_price)
                in_the_money_indices = np.arange(self.sampler.cnt_trajectories, dtype=int)
            else:
                in_the_money_indices = np.where(discounted_payoff[:, time_index] > 1e-9)[0]
                if (len(in_the_money_indices) / self.sampler.cnt_trajectories < 1e-2 or
                        len(in_the_money_indices) < 2 or test and self.weights[time_index] is None):
                    self.price_history[time_index] = self.option_price.mean()
                    continue
                features = self.sampler.markov_state[in_the_money_indices, time_index].copy()

                if not test:
                    self.scalers[time_index] = StandardScaler()
                    self.scalers[time_index].fit(features)
                features = self.scalers[time_index].transform(features)
                
                transformed = self.basis_functions_transformer.fit_transform(features)
                
                if not test:
                    regularization = np.eye(transformed.shape[1], dtype=float) * self.regularization_alpha
                    inv = np.linalg.pinv((transformed.T @ transformed + regularization), rcond=1e-10)
                    self.weights[time_index] = inv @ transformed.T @ self.option_price[in_the_money_indices]
                
                continuation_value = transformed @ self.weights[time_index]

            indicator = discounted_payoff[in_the_money_indices, time_index] > continuation_value
            self.option_price[in_the_money_indices] = \
                (indicator * discounted_payoff[in_the_money_indices, time_index].copy() +
                 ~indicator * self.option_price[in_the_money_indices]) # тут неправильно использовать CV. CV получен как предикт и поэтому будет получаться накоп ошибок
            
            self.price_history[time_index] = self.option_price.mean()
            if not quiet and time_index % 10 == 0:
                _plot_progress(self.sampler, bar, self.price_history, lower_bound, upper_bound, ax=ax)  # Передаём ax

        key = "test" if test else "train"
        self.result[key] = {
            "price": float(self.option_price.mean()),
            "upper_bound": float(discounted_payoff.max(axis=1).mean()),
            "lower_bound": float(discounted_payoff.mean(axis=0).max()),
            "std": float(self.option_price.std())
        }

        return self.price_history
    
    def continuation_value(self, state, t):
        assert isinstance(t, float | int)
        if isinstance(state, float | int):
            state = np.array([[state]])
        elif isinstance(state, list | np.ndarray):
            state = np.array([state])
            assert len(state.shape) == 2
        else:
            raise ValueError()
        assert ((self.sampler.time_grid - t)**2).min() < 1e-6
        i = np.argmin((self.sampler.time_grid - t)**2)
        if self.weights[i] is None: 
            return np.nan
        return (
            self.basis_functions_transformer.transform(
                self.scalers[i].transform(state)
            ) @ self.weights[i]
        ).flatten()[0] / self.sampler.discount_factor[0][i]
