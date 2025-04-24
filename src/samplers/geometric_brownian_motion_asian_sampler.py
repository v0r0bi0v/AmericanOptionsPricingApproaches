import numpy as np
from tqdm.auto import tqdm
from src.samplers.geometric_brownian_motion_put_sampler import GeometricBrownianMotionPutSampler


class GeometricBrownianMotionAsianPutSampler(GeometricBrownianMotionPutSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(self) -> None:
        normals = self.random_state.normal(0, 1, (self.cnt_trajectories, self.cnt_times - 1))
        self.markov_state = np.zeros((self.cnt_trajectories, self.cnt_times, 1), dtype=float)
        self.markov_state[:, 0, 0] = self.asset0

        for t in tqdm(range(1, self.cnt_times), desc="GBM sampling for Asian put"):
            dt = self.time_deltas[t - 1]
            self.markov_state[:, t, 0] = self.markov_state[:, t - 1, 0] * np.exp(
                (self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * normals[:, t - 1]
            )

        average_price = np.mean(
            self.markov_state[:, :, 0], axis = 1
            )

        self.payoff = np.zeros((self.cnt_trajectories, self.cnt_times), dtype = float)
        self.payoff[:, -1] = np.clip(
            self.strike - average_price, 0, 1e20
            )

        self.discount_factor = np.repeat(
            np.exp(-self.r * self.time_grid).reshape((1, -1)), self.cnt_trajectories, axis=0
        )