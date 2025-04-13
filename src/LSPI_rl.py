import numpy as np
from abstracts import PricerAbstract

class LSPIPricer(PricerAbstract):
    def __init__(self, iterations: int = 5, tol: float = 1e-6):
        self.iterations = iterations # количество итераций GPI
        self.tol = tol # epsilon для сходимости политики:
        self.w = None  # веса
        self.basis_functions = None  # To store basis function values if needed (not required to keep globally).

    def _basis_functions(
            self, S: float, K: float, t: float, T: float
            ) -> np.ndarray:
        """
        Feature функции из книжки, они используются как фичи в приближении
        Q-value функции линрегом
        """
        M = S / K  # price ratio
        exp_term = np.exp(-M / 2.0)
        phi0 = 1.0
        phi1 = exp_term
        phi2 = exp_term * (1.0 - M)
        phi3 = exp_term * (1.0 - 2.0 * M + 0.5 * M * M)
        phi4 = np.sin(-np.pi * t / (2.0 * T) + np.pi / 2.0)
        phi5 = np.log(
            max(T - t, 1e-10)
            )
        phi6 = (t / T) ** 2
        return np.array([phi0, phi1, phi2, phi3, phi4, phi5, phi6],
                        dtype = float)

    def price(self, option, sampler) -> np.ndarray:
        """
        Compute the American option price using LSPI.

        :param option: An AmericanOption instance (with option.strike_price and other details).
        :param sampler: A GeometricBrownianMotionPutSampler instance (must be initialized with the option parameters and sampled).
        :return: Numpy array of present values of the option payoff for each simulated path under the optimal policy.
                 The mean of this array is the estimated option price at time 0.
        """
        # Ensure we have sample paths generated
        if sampler.markov_state is None or sampler.payoff is None or sampler.discount_factor is None:
            sampler.sample()

        K = option.strike_price  # страйк
        time_grid = sampler.time_grid
        T = time_grid[-1]
        r = sampler.r
        S_paths = sampler.markov_state[:, :,
                  0]  # sampler.markov_state это массив размера (cnt_traj,
        # cnt_times, 1)
        payoff_matrix = sampler.payoff  # shape (num_paths, num_time_steps): immediate exercise payoff max(K - S, 0)
        disc_matrix = sampler.discount_factor  # shape (num_paths, num_time_steps): discount factor to time 0 for each time

        num_paths, num_times = S_paths.shape

        d = 7
        w = np.zeros(d, dtype=float) if self.w is None else np.array(
            self.w, dtype=float
            ) # базовая инициализация накапливаемых весов

        # LSPI main loop: тут делаем GPI
        prev_w = None
        for it in range(self.iterations): # колво итераций GPI
            prev_w = w.copy()
            # инициализация накопа матрицы A и вектора b
            A = np.zeros((d, d), dtype = float)
            b = np.zeros(d, dtype = float)

            # **Policy evaluation step**: use current w to evaluate Q and update w via LSTDQ
            # Loop over all sample transitions (state -> next state) where action "continue" was taken in simulation
            # (The behavior policy for sample generation is to always continue to the next time step, until final.)
            for p in range(num_paths):
                # идём по пути p с текущей (fixed) политикой и делаем действия a

                exercised = False  # продали ли опцион

                for t_idx in range(num_times - 1):
                    if exercised:
                        break
                    S_t = S_paths[p, t_idx]  # цена базового актива в данном
                    # сечении
                    t_curr = time_grid[t_idx]  # текущее время
                    payoff_now = payoff_matrix[p, t_idx]  # immediate exercise payoff now (max(K-S,0))
                    Q_ex = disc_matrix[p, t_idx] * payoff_now

                    # payoff нашли, теперь ищем continuation value
                    features = self._basis_functions(S_t, K, t_curr, T)
                    Q_cont = float(np.dot(w, features))

                    if Q_ex >= Q_cont:
                        # Policy would exercise at this state (because immediate payoff >= continuation value).
                        # Under policy, episode ends here with reward Q_ex, so no "continue" transition from this state.
                        # We mark exercised and do NOT add this transition to A (since no continue action taken).
                        exercised = True
                        continue

                    # если мы здесь, то значит policy решила продолжать
                    # то есть из стейта s_t = S_t сделали action = continue,
                    # получили reward = 0 и перешли в стейт s_{t + 1} = S_{t
                    # + 1}
                    reward = 0.0

                    ####  TODO
                    ###$ TODO

    def plot_expected_prices(self):
        if self.w is None:
            raise RuntimeError(
                "запустите price()"
                )

        raise NotImplementedError(
            "не реализован"
            )

    def plot_sample(self):
        if self.w is None:
            raise RuntimeError(
                "запустите price()"
                )
        raise NotImplementedError(
            "не реализован"
            )
