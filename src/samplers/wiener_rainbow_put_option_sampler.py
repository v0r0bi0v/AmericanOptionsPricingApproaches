import numpy as np
from tqdm.auto import tqdm
from .abstract_sampler import SamplerAbstract


class WienerRainbowPutOptionSampler(SamplerAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigmas = kwargs.get("sigmas")
        self.dim = len(self.sigmas)
        self.strike = kwargs.get("strike")
        self.asset0 = kwargs.get("asset0") 
        self.r = kwargs.get("r")

    def sample(self) -> None:
        normals = self.random_state.normal(0, 1, (self.dim, self.cnt_trajectories, self.cnt_times - 1))
        self.markov_state = np.zeros((self.dim, self.cnt_trajectories, self.cnt_times), dtype=float)
        
        for i in range(self.dim):
            self.markov_state[i, :, 0] = self.asset0[i]  # Установка начальной цены
            for j in tqdm(range(self.cnt_trajectories), desc="WienerRainbow sampling with asset0"):
                self.markov_state[i][j][1:] = self.asset0[i] * np.exp(
                    (-0.5 * self.sigmas[i]**2) * self.time_deltas +
                    self.sigmas[i] * np.cumsum(normals[i][j] * np.sqrt(self.time_deltas))
                )
        
        self.payoff = np.clip(self.strike - np.min(self.markov_state, axis=0), 0, 1e20)
        self.discount_factor = np.repeat(
            np.exp(-self.r * self.time_grid).reshape((1, -1)), self.cnt_trajectories, axis=0
        )
        self.markov_state = np.transpose(self.markov_state, (1, 2, 0))