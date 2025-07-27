import numpy as np
from typing import Optional
from sklearn.preprocessing import StandardScaler
from abstracts import PricerAbstract
from abstracts import SamplerAbstract

class LSPIPricer(PricerAbstract):
    def __init__(
            self,
            sampler: SamplerAbstract,
            iterations: int = 20,
            tol: float = 1e-5,
            lambda_reg: float = 1e-1
    ):
        self.iterations = iterations
        self.tol = tol
        self.lambda_reg = lambda_reg
        self.w: Optional[np.ndarray] = None
        self.sampler = sampler
        self.scaler = StandardScaler()

    def _basis_functions_raw(self, S: np.ndarray, K: float, t: np.ndarray, T: float) -> np.ndarray:
        M = S / K
        exp_term = np.exp(-M / 2.0)

        phi = np.zeros((*S.shape, 7), dtype=float)
        phi[..., 0] = 1.0 
        phi[..., 1] = exp_term
        phi[..., 2] = exp_term * (1.0 - M)
        phi[..., 3] = exp_term * (1.0 - 2.0 * M + 0.5 * M * M)
        phi[..., 4] = np.sin(np.pi * (T - t) / (2.0 * T))
        phi[..., 5] = np.log(np.maximum(T - t, 1e-10))
        phi[..., 6] = (t / T) ** 2
        return phi

    def _scale_features(self, phi: np.ndarray, fit: bool = False) -> np.ndarray:
        
        features_to_scale = phi[..., 1:]
        original_shape = features_to_scale.shape
        
        features_reshaped = features_to_scale.reshape(-1, original_shape[-1])
        
        if fit:
            self.scaler.fit(features_reshaped)

        if not self.scaler.mean_.any():
            raise RuntimeError("Скейлер не обучен. Вызовите price(test=False) перед использованием.")

        scaled_features = self.scaler.transform(features_reshaped)
        scaled_features = scaled_features.reshape(original_shape)
        
        phi_final = np.copy(phi)
        phi_final[..., 1:] = scaled_features
        return phi_final

    def price(self, test=False, quiet=False) -> np.ndarray:
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
        
        phi_all_raw = self._basis_functions_raw(S_paths, K, time_grid_expanded, T)

        if not test:
            
            phi_all = self._scale_features(phi_all_raw, fit=True)
            
            w = np.zeros(d, dtype=float) if self.w is None else self.w.copy()
            phi_curr_all = phi_all[:, :-1, :]
            phi_next_all = phi_all[:, 1:, :]
            
            for it in range(self.iterations):
                prev_w = w.copy()
                A = np.zeros((d, d), dtype=float)
                b = np.zeros(d, dtype=float)
                
                Q_cont_next = np.tensordot(phi_next_all, w, axes=(2, 0))
                
                Q_ex_next = disc_matrix[:, 1:] * payoff_matrix[:, 1:]
                
                non_terminal_s_prime = np.tile(
                    (np.arange(num_times - 1) < num_times - 2)[np.newaxis, :], (num_paths, 1)
                )
                next_cont_cond = non_terminal_s_prime & (Q_cont_next >= Q_ex_next)
                
                diff_phi = phi_curr_all - next_cont_cond[..., np.newaxis] * gamma * phi_next_all
                
                A += np.tensordot(phi_curr_all, diff_phi, axes=([0, 1], [0, 1]))
                b += np.sum(phi_curr_all * (~next_cont_cond)[:, :, np.newaxis] * Q_ex_next[:, :, np.newaxis], axis=(0, 1))
                
                A += self.lambda_reg * np.eye(d)
                w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                
                diff = np.linalg.norm(w - prev_w)
                if not quiet:
                    print(f"Итерация {it}: норма разности весов = {diff:.4f}")
                if diff < self.tol:
                    break
            self.w = w.copy()
        
        w = self.w

        phi_all = self._scale_features(phi_all_raw, fit=False)
        
        pv_payoffs = np.zeros(num_paths, dtype=float)
        for p in range(num_paths):
            for t_idx in range(num_times):
                payoff_now = payoff_matrix[p, t_idx]
                disc_factor = disc_matrix[p, t_idx]
                
                if t_idx == num_times - 1:
                    pv_payoffs[p] = disc_factor * payoff_now
                    break
                
                Q_cont = np.dot(phi_all[p, t_idx, :], w)
                
                if payoff_now > 0 and payoff_now >= Q_cont:
                    pv_payoffs[p] = disc_factor * payoff_now
                    break
                    
        return np.array([np.mean(pv_payoffs)])
