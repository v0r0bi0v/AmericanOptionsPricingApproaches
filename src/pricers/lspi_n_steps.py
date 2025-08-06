from copy import deepcopy
import numpy as np
from typing import Optional
from sklearn.preprocessing import StandardScaler
from numba import njit
from src.pricers.abstract_pricer import PricerAbstract
from src.samplers.abstract_sampler import SamplerAbstract

class LSPINStepsPricer(PricerAbstract):
    def __init__(
        self,
        sampler: SamplerAbstract,
        iterations: int = 20,
        tol: float = 1e-5,
        lambda_reg: float = 1e-1
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
        S_paths = self.sampler.markov_state[:, :, 0]  # Берем только цену актива

        # Создаем расширенную временную сетку
        time_grid_expanded = np.tile(self.sampler.time_grid, (n_paths, 1))
        
        # Создаем базисные функции
        phi_all_raw = self._basis_functions_raw(S_paths, K, time_grid_expanded, T)
        
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
            payoff = self.sampler.payoff
            payoff_next = self.sampler.payoff[:, 1:]  # Выплаты в t+1
            
            # Выравнивание в 1D
            phi_all_flat = phi_all.reshape(-1, n_features)
            phi_curr_flat = phi_curr.reshape(-1, n_features)
            phi_next_flat = phi_next.reshape(-1, n_features)
            payoff_flat = payoff.reshape(-1)
            payoff_next_flat = payoff_next.reshape(-1)
            
            # Флаг нетерминальных состояний
            non_terminal_next = np.tile(np.arange(n_times-1) < n_times-2, (n_paths, 1))
            non_terminal_next_flat = non_terminal_next.reshape(-1)

            non_terminal = np.tile(np.arange(n_times) < n_times-1, (n_paths, 1))
            non_terminal_flat = non_terminal.reshape(-1)
            
            # Итерации LSPI
            for it in range(self.iterations):
                prev_w = deepcopy(w)
                
                rewards = (
                    (
                        ~non_terminal_flat | 
                        (
                        ((phi_all_flat @ w).reshape(-1) < payoff_flat)
                        & 
                        ~(payoff_flat < 0)
                        )) * \
                    payoff_flat
                ).reshape(n_paths, n_times)

                Q_cont_next = spread_with_gamma(rewards, gamma)[:, 1:].reshape(-1)

                X = phi_curr_flat
                y = gamma * Q_cont_next
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


@njit
def spread_with_gamma(arr, gamma):
    result = np.zeros_like(arr)
    rows, cols = np.where(np.abs(arr) > 1e-6)
    
    for i, j in zip(rows, cols):
        if arr[i, j] == 0:
            continue
            
        # Создаем диапазон для влияния
        k_values = np.arange(j + 1)
        powers = j - k_values
        influences = (gamma ** powers) * arr[i, j]
        
        # Находим позиции, которые еще не заполнены
        target_pos = k_values[result[i, k_values] == 0]
        result[i, target_pos] = influences[target_pos]
        
        # Обнуляем где power == 0
        result[i, j] = 0
        
    return result

# bellman_opt_eq(
#     phi_curr_flat, 
#     phi_next_flat, 
#     payoff_next_flat, 
#     pricer.w * 0.99, 
#     gamma,
#     pricer.w
# )