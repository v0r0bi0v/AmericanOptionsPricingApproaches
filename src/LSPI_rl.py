import numpy as np
from abstracts import PricerAbstract

class LSPIPricer(PricerAbstract):
    def __init__(self, iterations: int = 5, tol: float = 1e-6, lambda_reg: float = 1e-6, epsilon: float = 1e-6):
        self.iterations = iterations   # количество итераций GPI
        self.tol = tol                 # порог сходимости для вектора w
        self.lambda_reg = lambda_reg   # кф-т регуляризации для матрицы A
        self.epsilon = epsilon         # порог при сравнении Q_ex и Q_cont
        self.w = None

    def _basis_functions(self, S: float, K: float, t: float, T: float) -> np.ndarray:
        M = S / K
        exp_term = np.exp(-M / 2.0)
        phi0 = 1.0
        phi1 = exp_term
        phi2 = exp_term * (1.0 - M)
        phi3 = exp_term * (1.0 - 2.0 * M + 0.5 * M * M)
        phi4 = np.sin(-np.pi * t / (2.0 * T) + np.pi / 2.0)
        phi5 = np.log(max(T - t, 1e-10))
        phi6 = (t / T) ** 2
        return np.array([phi0, phi1, phi2, phi3, phi4, phi5, phi6], dtype=float)

    def price(self, sampler, **option_params) -> np.ndarray:

        if sampler.markov_state is None or sampler.payoff is None or sampler.discount_factor is None:
            sampler.sample()

        K = option_params.get("strike")
        time_grid = sampler.time_grid
        T = time_grid[-1]
        S_paths = sampler.markov_state[:, :, 0]   # матрица траекторий, shape (num_paths, num_times)
        payoff_matrix = sampler.payoff            # матрица payoffs = max(K - S, 0) того же размера
        disc_matrix = sampler.discount_factor     # матрица дисконт-факторов exp(-r * t)
        num_paths, num_times = S_paths.shape
        d = 7  # dim базиса

        # инициализация вектора весов w (если уже есть self.w с предыдущего прогона, используем
        # его как старт)
        if self.w is None:
            w = np.zeros(d, dtype=float)
        else:
            w = np.array(self.w, dtype=float)

        for it in range(1, self.iterations + 1):
            prev_w = w.copy()

            A = np.zeros((d, d), dtype=float)
            b = np.zeros(d, dtype=float)

            for p in range(num_paths):

                exercised = False

                for t_idx in range(num_times - 1):

                    if exercised:
                        break

                    S_t = S_paths[p, t_idx]
                    t_curr = time_grid[t_idx]
                    payoff_now = payoff_matrix[p, t_idx]
                    Q_ex = disc_matrix[p, t_idx] * payoff_now  # исполнить сейчас

                    # вычисляем continuation
                    phi_curr = self._basis_functions(S_t, K, t_curr, T)
                    Q_cont = float(np.dot(w, phi_curr))

                    if Q_ex >= Q_cont + self.epsilon:
                        exercised = True
                        continue

                    # переход (s -> s') при действии continue:
                    reward = 0.0
                    next_idx = t_idx + 1
                    S_next = S_paths[p, next_idx]
                    t_next = time_grid[next_idx]
                    payoff_next = payoff_matrix[p, next_idx]

                    Q_ex_next = disc_matrix[p, next_idx] * payoff_next  # дисконтированный payoff в следующем состоянии

                    if next_idx < num_times - 1:
                        phi_next = self._basis_functions(S_next, K, t_next, T)
                        Q_cont_next = float(np.dot(w, phi_next))
                    else:
                        phi_next = None
                        Q_cont_next = 0.0

                    # разбор случаев C1 и C2 из книги с порогом epsilon
                    if next_idx == num_times - 1 or Q_ex_next >= Q_cont_next + self.epsilon:
                        phi_next_vec = np.zeros(d, dtype=float)
                        future_reward = Q_ex_next
                    else:
                        phi_next_vec = phi_next
                        future_reward = 0.0

                    diff_phi = phi_curr - phi_next_vec  # phi(s) - gamma * phi(s') (gamma = 1)
                    A += np.outer(phi_curr, diff_phi)
                    b += phi_curr * (reward + future_reward)

                    # если следующий стейт был терминальным или exercise-состоянием, завершаем
                    # траекторию
                    if next_idx == num_times - 1 or Q_ex_next >= Q_cont_next + self.epsilon:
                        exercised = True
                        continue

            A += self.lambda_reg * np.eye(d) # регуляризация матрицы A
            w = np.linalg.solve(A, b) # новые веса

            # Проверяем сходимость по изменению весов
            diff_w = np.linalg.norm(w - prev_w, ord=2)
            if diff_w < self.tol:
                print(f"сошлись на {it}-ой итерации")
                break

        self.w = w.copy()  # сохраняем веса с последней итерации

        pv_payoffs = np.zeros(num_paths, dtype = float)
        for p in range(num_paths):
            for t_idx in range(num_times):

                S_t = S_paths[p, t_idx]
                payoff_now = payoff_matrix[p, t_idx]

                if t_idx == num_times - 1:
                    pv_payoffs[p] = disc_matrix[p, t_idx] * payoff_now
                    break

                t_curr = time_grid[t_idx]
                phi_curr = self._basis_functions(S_t, K, t_curr, T)

                Q_cont = float(np.dot(w, phi_curr))
                Q_ex = disc_matrix[p, t_idx] * payoff_now

                if Q_ex >= Q_cont + self.epsilon:

                    pv_payoffs[p] = Q_ex
                    break

        return pv_payoffs