import numpy as np
from abstracts import PricerAbstract

class LSPIPricer(PricerAbstract):
    def __init__(self, iterations: int = 5, tol: float = 1e-6):
        self.iterations = iterations # количество итераций GPI
        self.tol = tol # epsilon для сходимости политики:
        self.w = None  # веса
        self.basis_functions = None

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

        option: объект AmericanOption
        sampelr: объект GBM
        return: массив payoff'ов для каждой траектории GBM. среднее по этому массиву это payoff
        в момент времени t = 0

        """

        if sampler.markov_state is None or sampler.payoff is None or sampler.discount_factor is None:
            sampler.sample()

        K = option.strike_price  # страйк
        time_grid = sampler.time_grid
        T = time_grid[-1]
        # r = sampler.r
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

            # Policy Evaluation часть: используем текущие веса w для получения Q и обновления весов
            for p in range(num_paths):

                exercised = False  # продали ли опцион

                for t_idx in range(num_times - 1):
                    if exercised:
                        break
                    S_t = S_paths[p, t_idx]  # цена базового актива в данном сечении
                    t_curr = time_grid[t_idx]  # текущее время
                    payoff_now = payoff_matrix[p, t_idx]  # immediate exercise payoff now (max(K-S,0))
                    Q_ex = disc_matrix[p, t_idx] * payoff_now

                    # payoff нашли, теперь ищем continuation value
                    features = self._basis_functions(S_t, K, t_curr, T)
                    Q_cont = float(np.dot(w, features))

                    if Q_ex >= Q_cont: # жадная политика
                        exercised = True
                        continue

                    # если мы здесь, то значит policy решила продолжать
                    # то есть из стейта s_t = S_t сделали action = continue,
                    # получили reward = 0 и перешли в стейт s_{t + 1} = S_{t + 1}
                    reward = 0.0

                    # получаем следующее состояние и действие, действуя политикой
                    next_idx = t_idx + 1
                    S_next = S_paths[p, next_idx]
                    t_next = time_grid[next_idx]
                    payoff_next = payoff_matrix[p, next_idx]

                    Q_ex_next = disc_matrix[p, next_idx] * payoff_next # payoff в след состоянии

                    # начинаем считать cv для след. состояния
                    features_next = None
                    if next_idx < num_times - 1: # если нетерминальное
                        features_next = self._basis_functions(
                            S_next, K, t_next, T
                            )
                        Q_cont_next = float(np.dot(w, features_next))
                    else:
                        # находимся в моменте экспирации, поэтому можем только продать, cv = 0
                        features_next = None
                        Q_cont_next = 0.0

                    if next_idx == num_times - 1 or Q_ex_next >= Q_cont_next:
                        phi_next = np.zeros(
                            d, dtype=float
                            )
                        future_reward = Q_ex_next
                    else:
                        phi_next = features_next
                        future_reward = 0.0

                    # накопление матрицы A и вектора b. это можно ускорить до O(n^2), (см книгу)
                    phi_curr = features
                    diff_phi = phi_curr - phi_next
                    A += np.outer(phi_curr, diff_phi)
                    b += phi_curr * (reward + future_reward)

                    if next_idx == num_times - 1 or Q_ex_next >= Q_cont_next:
                        exercised = True
                        continue

            # обновляем веса
            try:
                w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(A, b, rcond = None)[0]

            # проверка на сходимость. если уже сошлись, то выходим
            if np.allclose(w, prev_w, atol = self.tol, rtol = 0):
                break

        self.w = w.copy()

        # непосредственно сам pricing
        pv_payoffs = np.zeros(num_paths, dtype = float)
        for p in range(num_paths):

            for t_idx in range(num_times):

                S_t = S_paths[p, t_idx]
                payoff_now = payoff_matrix[p, t_idx]

                if t_idx == num_times - 1:
                    pv_payoffs[p] = disc_matrix[p, t_idx] * payoff_now
                    break

                t_curr = time_grid[t_idx]

                features = self._basis_functions(S_t, K, t_curr, T)
                Q_cont = float(np.dot(w, features))
                Q_ex = disc_matrix[p, t_idx] * payoff_now

                if Q_ex >= Q_cont:
                    pv_payoffs[p] = disc_matrix[p, t_idx] * payoff_now
                    break

        return pv_payoffs

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