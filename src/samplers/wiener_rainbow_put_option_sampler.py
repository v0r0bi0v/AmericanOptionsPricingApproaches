import numpy as np
from tqdm.notebook import tqdm_notebook as tqdm
from abstract_sampler import SamplerAbstract


class WienerRainbowPutOptionSampler(SamplerAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigmas = kwargs.get("sigmas") # список волатильностей каждого
        # актива (это радужный, поэтому их тут мб много)
        self.strike = kwargs.get("strike")     # скаляр
        self.r = kwargs.get("r", 0.05)          #  скаляр
        self.dim = len(self.sigmas) # размерность радужного вектора

    def sample(self) -> None:
        normals = self.random_state.normal(0, 1, (self.dim, self.cnt_trajectories, self.cnt_times - 1))
        self.markov_state = np.zeros((self.dim, self.cnt_trajectories, self.cnt_times), dtype=float)
        
        for i in range(self.dim):
            for j in tqdm(range(self.cnt_trajectories)):
                self.markov_state[i][j] = np.zeros(len(self.time_deltas) + 1, dtype=float)
                self.markov_state[i][j][1:] = self.sigmas[i] * np.cumsum(normals[i][j] * np.sqrt(self.time_deltas))
        
        self.payoff = np.clip(self.strike - np.min(self.markov_state, axis=0), 0, 1e20)
        
        self.discount_factor = np.repeat(
            np.exp(-self.r * self.time_grid).reshape((1, -1)), self.cnt_trajectories, axis=0
        )

        self.markov_state = np.transpose(self.markov_state, (1, 2, 0))