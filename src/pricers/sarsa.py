import random
from typing import TypeVar, Set

import numpy as np
from src.pricers.abstract_pricer import PricerAbstract
from src.samplers.abstract_sampler import SamplerAbstract
from src.samplers.geometric_brownian_motion_put_sampler import GeometricBrownianMotionPutSampler

A = TypeVar('A')

class State:
    def __init__(self, t: int, price: int):
        self.t = t
        self.price = price


class Sarsa(PricerAbstract):
    def __init__(
            self,
            gamma: float,
            strike: int,
            sampler: SamplerAbstract,
            alpha: float = 0.1
    ):
        self.gamma = gamma # костыль, должен быть в sampler'е
        self.strike = strike * 100 # костыль, должен быть в sampler'е
        self.sampler = sampler
        self.alpha = alpha


    @staticmethod
    def epsilon_greedy_action(
            q: dict,
            s: State,
            epsilon: float
    ) -> A:
        optimal_a, max_q = None, 0
        for action in range(2):
            if not ((s.t, s.price), action) in q.keys():
                q[((s.t, s.price), action)] = 0
            if q[((s.t, s.price), action)] >= max_q:
                max_q = q[((s.t, s.price), action)]
                optimal_a = action

        if np.random.uniform(0, 1) < epsilon:
            return random.randint(0, 1)
        else:
            return optimal_a


    def step(self, s: State, action: A) -> (float, bool):
        if action == 0:
            return 0, False
        return max(self.strike - s.price, 0), True


    def price(self, test=False, quiet=None):
        self.sampler.sample()
        paths = (self.sampler.markov_state[:, :, 0] * 100).astype(int)
        num_paths, num_times = paths.shape

        q = {}
        for path in range(num_paths):
            state = State(0, paths[path, 0])
            epsilon = 1 / (path + 1)

            action = self.epsilon_greedy_action(q, state, epsilon)
            for time in range(num_times):
                reward, finish = self.step(state, action)

                next_price = paths[path, time + 1]
                next_state = State(time + 1, next_price)
                next_action = self.epsilon_greedy_action(q, next_state, epsilon)

                if not finish:
                    reward += self.gamma * q[((next_state.t, next_state.price), next_action)]

                q[((state.t, state.price), action)] += self.alpha * (reward - q[((state.t, state.price), action)])

                if finish:
                    break # ??? вот тут можно не прерывать обработку траектории, а идти дальше по ней до конца ???
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
        "cnt_trajectories": 100000
    }
    sarsa = Sarsa(gamma=0.98, strike=100, sampler=GeometricBrownianMotionPutSampler(**sp))
    sarsa.price()