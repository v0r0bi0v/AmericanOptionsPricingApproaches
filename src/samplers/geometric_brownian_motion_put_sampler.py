import numpy as np
from tqdm.notebook import tqdm_notebook as tqdm
from abstract_sampler import SamplerAbstract


class GeometricBrownianMotionPutSampler(SamplerAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = kwargs.get("sigma")        # 
        self.strike = kwargs.get("strike")      # 
        self.r = kwargs.get("r", 0.05)          # 
        self.asset0 = kwargs.get("asset0")      # 
        self.t = kwargs.get("t", 1)             

    def sample(self) -> None:
        normals = self.random_state.normal(0, 1, (self.cnt_trajectories, self.cnt_times - 1))
        self.markov_state = np.zeros((self.cnt_trajectories, self.cnt_times, 1), dtype=float)
        self.markov_state[:, 0, 0] = self.asset0
        
        # 
        for t in tqdm(range(1, self.cnt_times)):
            dt = self.time_deltas[t - 1]
            self.markov_state[:, t, 0] = self.markov_state[:, t - 1, 0] * np.exp(
                (self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * normals[:, t - 1]
            )
        
        self.payoff = np.clip(self.strike - self.markov_state[:, :, 0], 0, 1e20)

        self.discount_factor = np.repeat(
            np.exp(-self.r * self.time_grid).reshape((1, -1)), self.cnt_trajectories, axis=0
        )
