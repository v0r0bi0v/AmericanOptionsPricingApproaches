from copy import deepcopy
import numpy as np
from typing import Optional
from sklearn.preprocessing import StandardScaler
from numba import njit
from src.pricers.abstract_pricer import PricerAbstract
from src.samplers.abstract_sampler import SamplerAbstract

class LSPIFullGradientPricer(PricerAbstract):
    def __init__(
        self,
        sampler: SamplerAbstract,
        iterations: int = 20,
        tol: float = 1e-5,
        lambda_reg: float = 1e-1,
    ):
        self.sampler = sampler
        self.iterations = iterations
        self.tol = tol
        self.lambda_reg = lambda_reg
        self.w: np.ndarray | None = None
        self.scaler = StandardScaler()
        self.is_fitted_ = False

        # Проверка равномерности временной сетки
        dt = self.sampler.time_deltas[0]
        assert np.allclose(np.diff(sampler.time_grid), dt), "Time grid must be uniform"
        self.dt = dt
        self.T = self.sampler.time_grid[-1]

    def _basis_functions_raw(self, markov_state: np.ndarray, t: np.ndarray) -> np.ndarray:
        t = np.tile(t, (markov_state.shape[0], 1))
        S = markov_state[:, :, 0] / 100
        exp_term = np.exp(-S / 2.0)

        phi = np.zeros((*S.shape, 7), dtype=float)
        phi[..., 0] = 1.0 
        phi[..., 1] = exp_term
        phi[..., 2] = exp_term * (1.0 - S)
        phi[..., 3] = exp_term * (1.0 - 2.0 * S + 0.5 * S * S)
        phi[..., 4] = np.sin(np.pi * (self.T - t) / (2.0 * self.T))
        phi[..., 5] = np.log(np.maximum(self.T - t, 1e-10))
        phi[..., 6] = (t / self.T) ** 2
        return phi

    def _scale_features(self, phi: np.ndarray, fit: bool = False) -> np.ndarray:
        """Скейлинг признаков (кроме константы)"""
        features_to_scale = phi[..., 1:]
        original_shape = features_to_scale.shape
        
        features_reshaped = features_to_scale.reshape(-1, original_shape[-1])
        
        if fit:
            self.scaler.fit(features_reshaped)
            self.is_fitted_ = True
        else:
            if not self.is_fitted_:
                raise RuntimeError("Scaler not fitted. Call price(test=False) first.")

        scaled_features = self.scaler.transform(features_reshaped)
        scaled_features = scaled_features.reshape(original_shape)
        
        phi_final = np.copy(phi)
        phi_final[..., 1:] = scaled_features
        return phi_final

    def price(self, test: bool = False, quiet: bool = False) -> np.ndarray:
        self.sampler.sample()
        
        K = self.sampler.strike
        T = self.sampler.time_grid[-1]
        r = -np.log(self.sampler.discount_factor[0, 1] / self.sampler.discount_factor[0, 0]) / self.dt
        
        # Проверка детерминированности discount factor
        assert np.allclose(
            self.sampler.discount_factor,
            np.repeat(
                np.exp(-r * self.sampler.time_grid)[np.newaxis],
                repeats=self.sampler.discount_factor.shape[0],
                axis=0
            )
        ), "Discount factor cannot be stochastic yet"
        
        # Параметры путей
        n_paths, n_times, _ = self.sampler.markov_state.shape
        gamma = np.exp(-r * self.dt)
        
        # Создаем базисные функции
        phi_all_raw = self._basis_functions_raw(self.sampler.markov_state, self.sampler.time_grid)
        
        # Применяем скейлинг
        if not test:
            phi_all = self._scale_features(phi_all_raw, fit=True)
        else:
            phi_all = self._scale_features(phi_all_raw, fit=False)
        
        n_features = phi_all.shape[-1]

        # Инициализация весов
        if self.w is None or not test:
            w = np.zeros(n_features) if self.w is None else self.w.copy()
        else:
            w = self.w

        # Режим обучения (обновление весов)
        if not test:
            # Подготовка данных
            phi_curr = phi_all[:, :-1, :]  # Текущие состояния (t)
            phi_next = phi_all[:, 1:, :]   # Следующие состояния (t+1)
            payoff_next = self.sampler.payoff[:, 1:]  # Выплаты в t+1
            
            # Выравнивание в 1D
            phi_curr_flat = phi_curr.reshape(-1, n_features)
            phi_next_flat = phi_next.reshape(-1, n_features)
            payoff_next_flat = payoff_next.reshape(-1)
            
            # Флаг нетерминальных состояний
            non_terminal_next = np.tile(np.arange(n_times-1) < n_times-2, (n_paths, 1))
            non_terminal_next_flat = non_terminal_next.reshape(-1)
            
            # Итерации LSPI
            for it in range(self.iterations):
                prev_w = deepcopy(w)
                
                Q_cont_next = phi_next_flat @ w
                continue_cond = non_terminal_next_flat & ((Q_cont_next >= payoff_next_flat) | (payoff_next_flat < 0))
                
                X = phi_curr_flat - gamma * continue_cond[:, np.newaxis] * phi_next_flat
                y = gamma * (
                    ~continue_cond * payoff_next_flat
                )
                w = np.linalg.pinv(X.T @ X + np.eye(n_features) * self.lambda_reg, rcond=1e-10) \
                    @ (X.T @ y)

                # Проверка сходимости
                diff_norm = np.linalg.norm(w - prev_w)
                if not quiet:
                    print(f"Iteration {it}: ||Δw|| = {diff_norm:.9f}")
                if diff_norm < self.tol:
                    if not quiet:
                        print(f"Converged after {it} iterations")
                    break
            
            self.w = w

        # Оценка политики
        pv_payoffs = np.zeros(n_paths)
        for p in range(n_paths):
            for t in range(n_times):
                # Признаки текущего состояния
                phi_t = phi_all[p, t]
                Q_cont = phi_t @ self.w
                payoff_t = self.sampler.payoff[p, t]
                
                # Условие исполнения
                disc_factor = self.sampler.discount_factor[p, t]
                if payoff_t >= Q_cont or t == n_times - 1:
                    pv_payoffs[p] = disc_factor * payoff_t
                    break

        if not quiet:
            print("q-value at 0:", phi_all[0, 0] @ self.w)
            print("Option price:", np.mean(pv_payoffs))
        
        return np.array([np.mean(pv_payoffs)])
    
    def continuation_value(self, state, t):
        if isinstance(state, float | int | list | np.ndarray) and isinstance(t, float | int):
            if isinstance(state, list | np.ndarray):
                state = np.array(state)[np.newaxis, np.newaxis, :]
                assert len(state.shape) == 3
            else:
                state = np.array([[[state]]])
            t = np.array([t])
        else:
            raise ValueError()

        return (
            self._scale_features(
                self._basis_functions_raw(state, t), 
                fit=False
            ) @ self.w
        ).flatten()[0]
