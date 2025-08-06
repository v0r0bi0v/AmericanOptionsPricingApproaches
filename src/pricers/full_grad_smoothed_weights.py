import numpy as np
from typing import Optional
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from .abstract_pricer import PricerAbstract
from ..samplers.abstract_sampler import SamplerAbstract

class LSPIPricer(PricerAbstract):
    def __init__(
        self,
        sampler: SamplerAbstract,
        degree: int = 3,
        iterations: int = 20,
        tol: float = 1e-5,
        lambda_reg: float = 1e-1,
    ):
        self.sampler = sampler
        self.degree = degree
        self.iterations = iterations
        self.tol = tol
        self.lambda_reg = lambda_reg
        self.w: np.ndarray | None = None
        self.is_fitted_ = False  # Флаг обученности трансформеров
        self.alpha = 0.7

        # Проверка равномерности временной сетки
        dt = self.sampler.time_deltas[0]
        assert np.allclose(np.diff(sampler.time_grid), dt), "Time grid must be uniform"
        self.dt = dt

    def _create_features(self, markov_state: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
        """Создает полиномиальные признаки из марковского состояния и времени"""
        n_paths, n_times, state_dim = markov_state.shape
        T = time_grid[-1]
        
        # Нормализованное время до экспирации
        time_to_exp = (T - time_grid) / T
        time_to_exp = np.tile(time_to_exp, (n_paths, 1))[..., None]
        
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
 
            phi_curr_flat = phi_curr.reshape(-1, n_features)
            phi_next_flat = phi_next.reshape(-1, n_features)
            payoff_next_flat = payoff_next.reshape(-1)
            
            non_terminal = np.tile(np.arange(n_times-1) < n_times-2, (n_paths, 1))
            non_terminal_flat = non_terminal.reshape(-1)
            
            for it in range(self.iterations):
                prev_w = deepcopy(w)
                
                Q_cont_next = phi_next_flat @ w
                
                continue_cond = non_terminal_flat & (Q_cont_next >= payoff_next_flat)
                
                diff_phi = phi_curr_flat - gamma * continue_cond[:, None] * phi_next_flat
                # A = phi_curr_flat.T @ diff_phi
                
                # полный градиент 
                A = diff_phi @ diff_phi.T
                
                b = phi_curr_flat.T @ (gamma * (1 - continue_cond) * payoff_next_flat)
                
                A += self.lambda_reg * np.eye(n_features)
                
                w_new = np.linalg.lstsq(A, b, rcond = 1e-10)[0]
                
                # w = np.linalg.lstsq(A, b, rcond = 1e-10)[0]
                
                w = (1 - self.alpha) * prev_w + self.alpha * w_new
                
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
                
        if not quiet:
            print(f"q функция равна {phi_all[0, 0] @ w}")
            print(f"цена равна {pv_payoffs.mean()}")
        
        return np.array([pv_payoffs.mean()])