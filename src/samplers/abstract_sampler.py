from abc import abstractmethod
from utils.plotting_utils import plot_trajectories
import warnings
import numpy as np


class SamplerAbstract:
    def __init__(
            self,
            cnt_trajectories: int,
            cnt_times: int | None = None,
            t: float | None = None,
            time_grid: np.ndarray | None = None,
            seed: int | None = None,
            *args,
            **kwargs
    ):
        if (cnt_times is None or t is None) and time_grid is None:
            raise ValueError("You must specify time and cnt_times or time_grid")
        if time_grid is None:
            self.time_grid = np.linspace(0, t, cnt_times)
        else:
            self.time_grid = time_grid
        self.time_deltas = np.diff(self.time_grid)
        self.cnt_trajectories = cnt_trajectories
        self.cnt_times = len(self.time_grid)
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.markov_state = None
        self.payoff = None
        self.discount_factor = None

    @abstractmethod
    def sample(self) -> None:
        pass

    def plot(
            self,
            cnt: int,
            plot_mean: False = False,
            y: str = "payoff"
    ):
        if "markov_state" in y:
            if self.markov_state.shape[2] != 1:
                warnings.warn("We cannot plot >=2d processes")
            plot_trajectories(self.time_grid, self.markov_state[:, :, 0], cnt, "Markov state", "Markov State",
                              plot_mean)
        if "payoff" in y:
            plot_trajectories(self.time_grid, self.payoff, cnt, "payoff", "Payoff", plot_mean)
        if "discount_factor" in y:
            plot_trajectories(self.time_grid, self.discount_factor, cnt, "discount factor",
                              "Discount Factor", plot_mean)
