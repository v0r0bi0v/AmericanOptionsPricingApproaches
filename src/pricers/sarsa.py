import random
from typing import TypeVar, Set

import numpy as np
from src.pricers.abstract_pricer import PricerAbstract
from src.samplers.abstract_sampler import SamplerAbstract
from src.samplers.geometric_brownian_motion_put_sampler import GeometricBrownianMotionPutSampler

A = TypeVar('A')

class State:
    def __init__(self, t: int, price: float):
        self.t = t
        self.price = price


class Sarsa(PricerAbstract):
    def __init__(
            self,
            gamma: float,
            strike: float,
            sampler: SamplerAbstract,
            alpha: float = 0.1
    ):
        self.gamma = gamma # костыль, должен быть в sampler'е
        self.strike = strike # костыль, должен быть в sampler'е
        self.sampler = sampler
        self.alpha = alpha
        self.T = sampler.cnt_times


    def feature_functions(self, s: State) -> np.ndarray:
        m = s.price / self.strike
        exp_term = np.exp(-m / 2.0)
        t = s.t
        T = self.T

        return np.array([
            1.0,
            exp_term,
            exp_term * (1.0 - m),
            exp_term * (1.0 - 2.0 * m + 0.5 * m * m),
            np.sin(np.pi * (T - t) / (2.0 * T)),
            np.log(np.maximum(T - t, 1e-10)),
            (t / T) * (t / T)
        ])


    def epsilon_greedy_action(
            self,
            theta: np.ndarray,
            s: State,
            epsilon: float
    ) -> A:
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(0, 1)
        else:
            features = self.feature_functions(s)
            not_execute_reward = theta[0] @ features
            execute_reward = theta[1] @ features
            if not_execute_reward > execute_reward:
                return 0
            return 1


    def step(self, s: State, action: A) -> (float, bool):
        if action == 0:
            return 0, False
        return max(self.strike - s.price, 0), True


    def price(self, test=False, quiet=None):
        self.sampler.sample()
        paths = self.sampler.markov_state[:, :, 0]
        num_paths, num_times = paths.shape

        q = {}
        theta = np.zeros((2, 7))
        for path in range(num_paths):
            state = State(0, paths[path, 0])
            epsilon = 1 / (path + 2)

            action = self.epsilon_greedy_action(theta, state, epsilon)
            for time in range(num_times - 1):
                reward, finish = self.step(state, action)

                next_state = State(time + 1, paths[path, time + 1])
                next_action = self.epsilon_greedy_action(theta, next_state, epsilon)

                features = self.feature_functions(state)
                q_current = theta[action] @ features

                if finish:
                    q_next = 0.0
                else:
                    next_features = self.feature_functions(next_state)
                    q_next = theta[next_action] @ next_features

                delta = reward + self.gamma * q_next - q_current
                theta[action] += self.alpha * delta * features

                state = next_state
                action = next_action
        return q


if __name__ == "__main__":
    sp = {
        "asset0": 100,
        "sigma": 0.2,
        "r": 0.05,
        "strike": 100,
        "t": 1.0,
        "cnt_times": 365,
        "seed": None,
        "cnt_trajectories": 1000
    }
    sarsa = Sarsa(gamma=0.98, strike=100, sampler=GeometricBrownianMotionPutSampler(**sp))
    sarsa.price()