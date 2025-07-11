import numpy as np
from tqdm import tqdm_notebook as tqdm
from abstracts import SamplerAbstract

class WienerRainbowPutOptionSampler(SamplerAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigmas = kwargs.get("sigmas")      # 
        self.strike = kwargs.get("strike")     
        self.r = kwargs.get("r", 0.05)          # 
        self.dim = len(self.sigmas)

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


class GeometricBrownianMotionPutSampler(SamplerAbstract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = kwargs.get("sigma")        # 
        self.strike = kwargs.get("strike")      # 
        self.r = kwargs.get("r", 0.05)          # 
        self.asset0 = kwargs.get("asset0")      # 
        self.t = kwargs.get("t", 1)             

    def sample(self) -> None:
        # normals - 2-d array of size (cnt_trajectories, cnt_times - 1)
        #           with values from normal distribution in each cell.
        normals = self.random_state.normal(0, 1, (self.cnt_trajectories, self.cnt_times - 1))
        # markov_state - 2-d array (only one row in third dimension) of size (cnt_trajectories, cnt_times). Each row is
        #                a separate trajectory of the markov process based on the GBM. First cell in each trajectory
        #                equals the price of asset.
        self.markov_state = np.zeros((self.cnt_trajectories, self.cnt_times, 1), dtype=float)
        self.markov_state[:, 0, 0] = self.asset0

        for t in tqdm(range(1, self.cnt_times)):
            dt = self.time_deltas[t - 1]
            # markov_state[:, t, 0] - a 1-d vector of size (cnt_trajectories). Every cell is a combination of
            #                         the previous state from the trajectory and GBM coefficient
            self.markov_state[:, t, 0] = self.markov_state[:, t - 1, 0] * np.exp(
                (self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * normals[:, t - 1]
            )

        # payoff - 2-d array of size (cnt_trajectories, cnt_times). Every cell shows the difference between
        #          the option's strike price and the generated markov_state
        self.payoff = np.clip(self.strike - self.markov_state[:, :, 0], 0, 1e20)

        # time_grid - 1-d vector of size (cnt_times).
        #             By default, equals [0, 1, 2, ..., t]
        # discount_factor - 2-d array of size (cnt_trajectories, cnt_times). All rows are equal. Every row equals
        #                   [e^0, e^(-r), e^(-2r) ..., e^(-tr)]
        self.discount_factor = np.repeat(
            np.exp(-self.r * self.time_grid).reshape((1, -1)), self.cnt_trajectories, axis=0
        )

MODELS_TO_SAMPLER = {
    "WienerRainbowPutOptionSampler": WienerRainbowPutOptionSampler,
    "GeometricBrownianMotionPutSampler": GeometricBrownianMotionPutSampler
}
