import numpy as np
from typing import Optional
from abstracts import PricerAbstract

class LSPIPricer(PricerAbstract):
    def __init__(
            self,
            iterations: int = 5,
            tol: float = 1e-6,
            lambda_reg: float = 1e-6,
            epsilon: float = 1e-6
    ):
        self.iterations = iterations
        self.tol = tol
        self.lambda_reg = lambda_reg # кф-т регуляризации для решения Aw = b
        self.epsilon = epsilon
        self.w: Optional[np.ndarray] = None

    def _basis_functions(self, S: np.ndarray, K: float, t: np.ndarray, T: float) -> np.ndarray:

        '''
        Список phi трехмерный, так как Q(s, a, w) = phi(s)^T @ w, для каждого состояния s.
        состояние определяется парой (время, цена), поэтому для каждой координаты вектора phi(s)
        нужны значения, т.е. получаем |cnt_times| * |cnt_traj| * d размерность
        phi[p, t, i] - значение i-ой базисной функции для траектории p в момент времени t
        '''

        M = S / K

        exp_term = np.exp(-M / 2.0)

        phi = np.zeros((*S.shape, 7), dtype=float)

        phi[:, :, 0] = 1.0
        phi[:, :, 1] = exp_term
        phi[:, :, 2] = exp_term * (1.0 - M)
        phi[:, :, 3] = exp_term * (1.0 - 2.0 * M + 0.5 * M * M)
        phi[:, :, 4] = np.sin(np.pi * (T - t) / (2.0 * T))
        phi[:, :, 5] = np.log(np.maximum(T - t, 1e-10))
        phi[:, :, 6] = (t / T) ** 2

        return phi

    def price(self, sampler, **option_params) -> np.ndarray:

        if sampler.markov_state is None or sampler.payoff is None or sampler.discount_factor is None:
            sampler.sample()

        K = option_params.get("strike")
        r = option_params.get("r")
        time_grid = sampler.time_grid
        T = time_grid[-1]
        dt = time_grid[1] - time_grid[0]
        gamma = np.exp(-r * dt)
        S_paths = sampler.markov_state[:, :, 0]
        payoff_matrix = sampler.payoff
        disc_matrix = sampler.discount_factor
        num_paths, num_times = S_paths.shape
        d = 7

        if self.w is None:
            w = np.zeros(d, dtype=float)
        else:
            w = self.w.copy()

        time_grid_expanded = np.tile(time_grid, (num_paths, 1))
        phi_all = self._basis_functions(S_paths, K, time_grid_expanded, T)
        phi_curr_all = phi_all[:, :-1, :]  # [num_paths, num_times-1, d]
        phi_next_all = phi_all[:, 1:, :]  # [num_paths, num_times-1, d]

        for it in range(self.iterations):
            prev_w = w.copy()
            A = np.zeros((d, d), dtype=float)
            b = np.zeros(d, dtype=float)

            # ptg,d -> pt эквивалентно sum_{d} (phi[p, t, d] * w[d])

            Q_cont_next = np.einsum('ptd,d->pt', phi_next_all, w)
            Q_ex_next = disc_matrix[:, 1:] * payoff_matrix[:, 1:]

            non_terminal_s_prime = np.tile((np.arange(num_times - 1) < num_times - 2)[np.newaxis, :], (num_paths, 1))

            next_cont_cond = non_terminal_s_prime & (Q_cont_next >= Q_ex_next - self.epsilon)
            diff_phi = phi_curr_all - next_cont_cond[:, :, None] * gamma * phi_next_all

            A += np.einsum('ptd,pte->de', phi_curr_all, diff_phi)
            b += np.einsum('ptd,pt->d', phi_curr_all, (~next_cont_cond) * gamma * payoff_matrix[:, 1:])

            A += self.lambda_reg * np.eye(d)
            w = np.linalg.solve(A, b)

            if np.linalg.norm(w - prev_w) < self.tol:
                print(f"сошлись на {it + 1}-й итерации")
                break

        self.w = w.copy()

        pv_payoffs = np.zeros(num_paths, dtype=float)
        for p in range(num_paths):
            for t_idx in range(num_times):
                payoff_now = payoff_matrix[p, t_idx]
                disc_factor = disc_matrix[p, t_idx]
                if t_idx == num_times - 1:
                    pv_payoffs[p] = disc_factor * payoff_now
                    break
                Q_cont = np.dot(phi_all[p, t_idx], w)
                continuation_value = Q_cont / disc_factor if disc_factor > 0 else 0
                if payoff_now >= continuation_value + self.epsilon:
                    pv_payoffs[p] = disc_factor * payoff_now
                    break

        return pv_payoffs