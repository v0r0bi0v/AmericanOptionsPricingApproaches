import numpy as np
from typing import Optional
from sklearn.preprocessing import StandardScaler
from ..pricers.abstract_pricer import PricerAbstract
from ..samplers.abstract_sampler import SamplerAbstract


class LSPIPricer(PricerAbstract):
    def __init__(
            self,
            sampler: SamplerAbstract,
            iterations: int = 20,
            tol: float = 1e-5,
            lambda_reg: float = 1e-1,
    ):
        self.iterations = iterations
        self.tol = tol
        self.lambda_reg = lambda_reg
        self.w: Optional[np.ndarray] = None
        self.sampler = sampler

    def _basis_functions(self, S: np.ndarray, K: float, t: np.ndarray, T: float) -> np.ndarray:
        M = S / K
        exp_term = np.exp(-M / 2.0)

        phi = np.zeros((*S.shape, 7), dtype = float)
        phi[:, :, 0] = 1.0
        phi[:, :, 1] = exp_term
        phi[:, :, 2] = exp_term * (1.0 - M)
        phi[:, :, 3] = exp_term * (1.0 - 2.0 * M + 0.5 * M * M)
        phi[:, :, 4] = np.sin(np.pi * (T - t) / (2.0 * T))
        phi[:, :, 5] = np.log(np.maximum(T - t, 1e-10)) # 1e-10 чтобы не было log(0)
        phi[:, :, 6] = (t / T) ** 2

        scaler = StandardScaler()
        phi_reshaped = phi[:, :, 1:].reshape(-1, 6)
        phi_scaled = scaler.fit_transform(phi_reshaped)
        phi_scaled = phi_scaled.reshape(phi.shape[0], phi.shape[1], 6)


        phi_final = np.zeros_like(phi)
        phi_final[:, :, 0] = phi[:, :, 0]
        phi_final[:, :, 1:] = phi_scaled

        return phi_final

    def price(self, test=False, quiet=None) -> np.ndarray:

        self.sampler.sample()

        K = self.sampler.strike
        r = self.sampler.r
        time_grid = self.sampler.time_grid
        T = time_grid[-1]
        dt = time_grid[1] - time_grid[0]
        gamma = np.exp(-r * dt)
        S_paths = self.sampler.markov_state[:, :, 0]
        payoff_matrix = self.sampler.payoff
        disc_matrix = self.sampler.discount_factor
        num_paths, num_times = S_paths.shape
        d = 7

        time_grid_expanded = np.tile(time_grid, (num_paths, 1))
        phi_all = self._basis_functions(S_paths, K, time_grid_expanded, T)

        if not test:

            w = np.zeros(d, dtype=float) if self.w is None else self.w.copy()

            phi_curr_all = phi_all[:, :-1, :]# базисные ф-ии кроме терминального состояния, phi(s_i)
            phi_next_all = phi_all[:, 1:, :]# базисные ф-ии кроме начального состояния, phi(s_i')


            for it in range(self.iterations):
                prev_w = w.copy()
                A = np.zeros((d, d), dtype=float)
                b = np.zeros(d, dtype=float)

                # phi_next_all размера num_paths × (num_times-1) × d) умножается на w размера d
                # и получается результат размера num_paths × (num_times-1)
                # соответствует phi(s_i') * w для каждой траектории p и для каждого времени t
                Q_cont_next = np.einsum('ptd,d->pt', phi_next_all, w)

                Q_ex_next = disc_matrix[:, 1:] * payoff_matrix[:, 1:]

                non_terminal_s_prime = np.tile(
                    (np.arange(num_times - 1) < num_times - 2)[np.newaxis, :], (num_paths, 1)
                )

                next_cont_cond = non_terminal_s_prime & (Q_cont_next >= Q_ex_next)

                # diff_phi - это phi(s_i) - Indicator(C_1) * gamma * phi(s_i')
                diff_phi = phi_curr_all - next_cont_cond[:, :, None] * gamma * phi_next_all

                A += np.einsum('ptd,pte->de', phi_curr_all, diff_phi)
                b += np.einsum('ptd,pt->d', phi_curr_all, (~next_cont_cond) * Q_ex_next)

                A += self.lambda_reg * np.eye(d)
                w, _, _, _ = np.linalg.lstsq(A, b, rcond = None)

                diff = np.linalg.norm(w - prev_w)
                print(f"на {it}-ой итерации норма разности весов составила {round(diff, 4)}")
                
                if diff < self.tol:
                    break
            self.w = w.copy()
        else:

            w = self.w

        pv_payoffs = np.zeros(num_paths, dtype=float)
        for p in range(num_paths):
            for t_idx in range(num_times):
                payoff_now = payoff_matrix[p, t_idx]
                disc_factor = disc_matrix[p, t_idx]
                if t_idx == num_times - 1:
                    pv_payoffs[p] = disc_factor * payoff_now
                    break
                Q_cont = np.dot(phi_all[p, t_idx], w)
                continuation_value = Q_cont
                if payoff_now >= continuation_value:
                    pv_payoffs[p] = disc_factor * payoff_now
                    break

        return pv_payoffs