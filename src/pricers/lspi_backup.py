from copy import deepcopy
import numpy as np
from typing import Optional
from sklearn.preprocessing import StandardScaler
from src.pricers.abstract_pricer import PricerAbstract
from src.samplers.abstract_sampler import SamplerAbstract


import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tqdm.auto import tqdm

class LSPIPricer(PricerAbstract):
    def __init__(
        self,
        sampler: SamplerAbstract,
        degree: int = 3,
        iterations: int = 20,
        tol: float = 1e-5,
        lambda_reg: float = 1e-1,
        use_paper_basis = True
    ):
        self.sampler = sampler
        self.degree = degree
        self.iterations = iterations
        self.tol = tol
        self.lambda_reg = lambda_reg
        self.w: np.ndarray | None = None
        self.is_fitted_ = False  # Флаг обученности трансформеров

        # Проверка равномерности временной сетки
        dt = self.sampler.time_deltas[0]
        assert np.allclose(np.diff(sampler.time_grid), dt), "Time grid must be uniform"
        self.dt = dt
        self.T = self.sampler.time_grid[-1]
        self.use_paper_basis = use_paper_basis

    def _create_features(self, markov_state: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
        """Создает полиномиальные признаки из марковского состояния и времени"""

        if not self.use_paper_basis:
            n_paths, n_times, state_dim = markov_state.shape
            
            # Нормализованное время до экспирации
            time_to_exp = (self.T - time_grid) / self.T
            time_to_exp = np.tile(time_to_exp, (n_paths, 1))[..., np.newaxis]
            
            # Объединяем время и состояние
            features = np.concatenate([time_to_exp, markov_state], axis=-1)
            flat_features = features.reshape(-1, features.shape[-1])
            
            # Первый вызов - инициализируем и фитим трансформеры
            if not self.is_fitted_:
                self.scaler_ = StandardScaler()
                self.poly_ = PolynomialFeatures(degree=self.degree, include_bias=True)
                
                scaled = self.scaler_.fit_transform(flat_features)
                poly_features = self.poly_.fit_transform(scaled)
                self.n_features_ = poly_features.shape[1]
                self.is_fitted_ = True
            else:
                # Используем обученные трансформеры
                scaled = self.scaler_.transform(flat_features)
                poly_features = self.poly_.transform(scaled)
            
            return poly_features.reshape(n_paths, n_times, -1)
        
        assert markov_state.shape[-1] == 1, "Only 1d basis functions from paper is implemented"

        M = markov_state[:, :, 0] / 100.
        exp_term = np.exp(-M / 2.0)
        t = time_grid

        phi = np.zeros((*markov_state[:, :, 0].shape, 7), dtype = float)
        phi[:, :, 0] = 1.0
        phi[:, :, 1] = exp_term
        phi[:, :, 2] = exp_term * (1.0 - M)
        phi[:, :, 3] = exp_term * (1.0 - 2.0 * M + 0.5 * M * M)
        phi[:, :, 4] = np.sin(np.pi * (self.T - t) / (2.0 * self.T))
        phi[:, :, 5] = np.log(np.maximum(self.T - t, 1e-10)) # 1e-10 чтобы не было log(0)
        phi[:, :, 6] = (t / self.T) ** 2

        phi_reshaped = phi[:, :, 1:].reshape(-1, 6)

        if not self.is_fitted_:
            self.scaler = StandardScaler()
            self.scaler.fit(phi_reshaped)
            self.is_fitted_ = True

        phi_scaled = self.scaler.transform(phi_reshaped)
        phi_scaled = phi_scaled.reshape(phi.shape[0], phi.shape[1], 6)


        phi_final = np.zeros_like(phi)
        phi_final[:, :, 0] = phi[:, :, 0]
        phi_final[:, :, 1:] = phi_scaled

        return phi_final


    def price(self, test: bool = False, quiet: bool = False) -> np.ndarray:
        # Генерация путей
        self.sampler.sample()

        r = -np.log(self.sampler.discount_factor[0, 1] / self.sampler.discount_factor[0, 0]) / (self.sampler.time_deltas[0])
        
        assert np.allclose(
            self.sampler.discount_factor,
            np.repeat(
                np.exp(-r * self.sampler.time_grid)[np.newaxis],
                repeats=self.sampler.discount_factor.shape[0],
                axis=0
            )
        ), "Discount factor cannot be stochastic yet"
        
        # Параметры
        n_paths, n_times, _ = self.sampler.markov_state.shape
        gamma = np.exp(-r * self.dt)  # Коэффициент дисконтирования
        
        # Создаем признаки
        phi_all = self._create_features(
            self.sampler.markov_state, 
            self.sampler.time_grid
        )
        n_features = phi_all.shape[-1]
        
        # Инициализация весов
        if self.w is None or not test:
            w = np.zeros(n_features) if self.w is None else self.w.copy()
        else:
            w = self.w

        # Режим обучения
        if not test:
            # Подготовка данных
            phi_curr = phi_all[:, :-1, :]  # Текущие состояния (t)
            phi_next = phi_all[:, 1:, :]    # Следующие состояния (t+1)
            payoff_next = self.sampler.payoff[:, 1:]  # Выплаты в t+1
            
            # Выравнивание в 1D
            phi_curr_flat = phi_curr.reshape(-1, n_features)
            phi_next_flat = phi_next.reshape(-1, n_features)
            payoff_next_flat = payoff_next.reshape(-1)
            
            # Флаг нетерминальных состояний
            non_terminal = np.tile(np.arange(n_times-1) < n_times-2, (n_paths, 1))
            non_terminal_flat = non_terminal.reshape(-1)
            
            # Итерации LSPI
            for it in range(self.iterations):
                prev_w = deepcopy(w)
                
                # Вычисляем Q-значения продолжения
                Q_cont_next = phi_next_flat @ w
                
                # Условие продолжения
                continue_cond = non_terminal_flat & (Q_cont_next >= payoff_next_flat)
                
                # Формируем систему уравнений
                diff_phi = phi_curr_flat - gamma * continue_cond[:, None] * phi_next_flat
                A = phi_curr_flat.T @ diff_phi
                b = phi_curr_flat.T @ (gamma * (1 - continue_cond) * payoff_next_flat)
                
                A += self.lambda_reg * np.eye(n_features)
                w = np.linalg.solve(A, b)
                
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
                Q_cont = phi_t @ w
                payoff_t = self.sampler.payoff[p, t]
                
                # Условие исполнения
                if payoff_t >= Q_cont or t == n_times - 1:
                    disc_factor = self.sampler.discount_factor[p, t]
                    pv_payoffs[p] = disc_factor * payoff_t
                    break
        
        return np.array([pv_payoffs.mean()])
