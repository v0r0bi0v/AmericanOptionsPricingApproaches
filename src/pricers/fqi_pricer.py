import numpy as np
from tqdm.auto import tqdm

from src.pricers.abstract_pricer import PricerAbstract
from src.samplers.abstract_sampler import SamplerAbstract


class FQIPricerDiploma(PricerAbstract):

    def __init__(
        self,
        sampler: SamplerAbstract,
        iterations: int = 25,
        lambda_reg: float = 0.0,
        tol: float = 1e-6,
        debug: bool = False,
    ):
        self.sampler = sampler
        self.iterations = iterations
        self.lambda_reg = lambda_reg
        self.tol = tol
        self.debug = debug

        self.gamma: float | None = None
        self.w: np.ndarray | None = None  # веса для 7 базисных функций

        self.price_history: np.ndarray | None = None
        self.option_price: np.ndarray | None = None
        self.result: dict = {}

    # ---------- базисные функции φ(S,t) (как в Li 2009) ----------

    def _basis_ST(self, S: np.ndarray, t: np.ndarray) -> np.ndarray:
        S = np.asarray(S, dtype=float)
        t = np.asarray(t, dtype=float)

        K = getattr(self.sampler, "strike")
        T = getattr(self.sampler, "t", self.sampler.time_grid[-1])

        S_norm = S / K  # S' = S / K

        # Базис по цене (лагерровские)
        phi0_S = np.ones_like(S_norm)
        exp_term = np.exp(-S_norm / 2.0)
        phi1_S = exp_term
        phi2_S = exp_term * (1.0 - S_norm)
        phi3_S = exp_term * (1.0 - 2.0 * S_norm + 0.5 * S_norm**2)

        # Базис по времени
        phi0_t = np.sin(-t * np.pi / (2.0 * T) + np.pi / 2.0)
        phi1_t = np.log(np.clip(T - t, 1e-12, None))
        phi2_t = (t / T) ** 2

        return np.stack(
            [phi0_S, phi1_S, phi2_S, phi3_S,
             phi0_t, phi1_t, phi2_t],
            axis=-1,
        )

    # ---------- подготовка датасета переходов ----------

    def _build_dataset(self):

        S = self.sampler.markov_state[:, :, 0]  # (n_traj, n_times)
        payoff = self.sampler.payoff           # (n_traj, n_times)
        time_grid = self.sampler.time_grid
        n_traj, n_times = payoff.shape

        dt = time_grid[1] - time_grid[0]
        self.gamma = float(np.exp(-self.sampler.r * dt))

        t_curr = np.broadcast_to(time_grid[:-1], (n_traj, n_times - 1))
        t_next = np.broadcast_to(time_grid[1:], (n_traj, n_times - 1))

        S_curr = S[:, :-1].reshape(-1)
        S_next = S[:, 1:].reshape(-1)
        t_curr_flat = t_curr.reshape(-1)
        t_next_flat = t_next.reshape(-1)
        g_next = payoff[:, 1:].reshape(-1)  # payoff в следующем состоянии

        # next-шаг терминален, если это переход к последнему времени
        terminal = np.zeros((n_traj, n_times - 1), dtype=bool)
        terminal[:, -1] = True
        terminal = terminal.reshape(-1)

        X = np.stack([S_curr, t_curr_flat], axis=1)
        X_next = np.stack([S_next, t_next_flat], axis=1)

        return X, X_next, g_next, terminal

    def _features(self, X: np.ndarray):
        return self._basis_ST(S=X[:, 0], t=X[:, 1])

    def _fit_features(self, X: np.ndarray, X_next: np.ndarray):
        Phi = self._features(X)
        Phi_next = self._features(X_next)
        return Phi, Phi_next

    # ---------- одна итерация FQI ----------

    def _fqi_iteration(self, Phi, Phi_next, g_next, terminal, w):

        gamma = self.gamma
        n_samples, n_features = Phi.shape

        # Q^{(i)}(s') = текущая continuation value в next-состояниях
        Q_next = Phi_next @ w

        # для терминальных next-состояний continuation невозможен → берём g_next
        max_term = np.maximum(g_next, Q_next)
        target_next = np.where(terminal, g_next, max_term)

        # итоговый таргет y = γ * target_next
        y = gamma * target_next

        if self.debug:
            frac_terminal = terminal.mean()
            frac_exercise_like = (g_next >= Q_next).mean()
            print(f"[FQI] terminal fraction: {frac_terminal:.3f}")
            print(f"[FQI] g_next >= Q_next fraction: {frac_exercise_like:.3f}")

        # A и b для линейной регрессии
        A = Phi.T @ Phi
        if self.lambda_reg > 0:
            A += self.lambda_reg * np.eye(n_features)
        b = Phi.T @ y

        if self.debug:
            try:
                svals = np.linalg.svd(A, compute_uv=False)
                cond_A = float(svals[0] / (svals[-1] + 1e-18))
            except Exception:
                cond_A = np.inf
            print(f"[FQI] cond(A): {cond_A:.3e}")

        w_new = np.linalg.pinv(A, rcond=1e-10) @ b

        if self.debug:
            residual = np.linalg.norm(A @ w_new - b) / (np.linalg.norm(b) + 1e-12)
            print(f"[FQI] relative residual ||Aw-b||/||b||: {residual:.3e}")

        return w_new

    # ---------- оценка политики (как в LSPI) ----------

    def _evaluate_policy(self):

        S = self.sampler.markov_state[:, :, 0]
        payoff = self.sampler.payoff
        disc = self.sampler.discount_factor
        time_grid = self.sampler.time_grid

        n_traj, n_times = payoff.shape
        w = self.w

        # continuation value во всех (S,t) кроме терминального времени
        S_curr = S[:, :-1]
        t_curr = np.broadcast_to(time_grid[:-1], S_curr.shape)

        Phi_curr = self._basis_ST(
            S_curr.reshape(-1),
            t_curr.reshape(-1),
        )
        cont_flat = Phi_curr @ w
        cont = cont_flat.reshape(n_traj, n_times - 1)

        # правило exercise: payoff >= continuation
        exercise_cond = payoff[:, :-1] >= cont

        # диагностика
        if self.debug:
            mask_bad = (payoff[:, :-1] == 0.0) & (cont < 0.0) & exercise_cond
            frac_bad = mask_bad.mean()
            print(f"[policy] payoff==0 & cont<0 & exercise fraction: {frac_bad:.3f}")
            if frac_bad > 0:
                frac_bad_by_time = mask_bad.mean(axis=0)
                print("[policy] first 10 time steps bad fraction:",
                      np.array2string(frac_bad_by_time[:10], precision=3))

            has_early_ex = exercise_cond.any(axis=1)
            first_ex_idx = exercise_cond.argmax(axis=1)
            exercise_time_idx_dbg = np.where(
                has_early_ex,
                first_ex_idx,
                n_times - 1,
            )
            mean_exercise_time = time_grid[exercise_time_idx_dbg].mean()
            frac_early = np.mean(exercise_time_idx_dbg < (n_times - 1))
            print(f"[policy] mean exercise time (years): {mean_exercise_time:.4f}")
            print(f"[policy] fraction exercised before maturity: {frac_early:.3f}")

        has_early_ex = exercise_cond.any(axis=1)
        first_ex_idx = exercise_cond.argmax(axis=1)
        exercise_time_idx = np.where(
            has_early_ex,
            first_ex_idx,
            n_times - 1,
        )

        idx = np.arange(n_traj)
        chosen_payoff = payoff[idx, exercise_time_idx]
        chosen_disc = disc[idx, exercise_time_idx]
        values = chosen_payoff * chosen_disc
        return values

    def debug_evaluate_policy(self):

        S = self.sampler.markov_state[:, :, 0]
        payoff = self.sampler.payoff
        disc = self.sampler.discount_factor
        time_grid = self.sampler.time_grid

        n_traj, n_times = payoff.shape
        w = self.w

        S_curr = S[:, :-1]
        t_curr = np.broadcast_to(time_grid[:-1], S_curr.shape)

        Phi_curr = self._basis_ST(
            S_curr.reshape(-1),
            t_curr.reshape(-1),
        )
        cont_flat = Phi_curr @ w
        cont = cont_flat.reshape(n_traj, n_times - 1)

        exercise_cond = payoff[:, :-1] >= cont
        has_early_ex = exercise_cond.any(axis=1)
        first_ex_idx = exercise_cond.argmax(axis=1)
        exercise_time_idx = np.where(
            has_early_ex,
            first_ex_idx,
            n_times - 1,
        )

        idx = np.arange(n_traj)
        chosen_payoff = payoff[idx, exercise_time_idx]
        chosen_disc = disc[idx, exercise_time_idx]
        values = chosen_payoff * chosen_disc

        return values, exercise_time_idx, cont, exercise_cond

    # ---------- внешний интерфейс ----------

    def price(self, test: bool = False, quiet: bool = False, ax=None):
        if not test:
            # TRAIN
            self.sampler.sample()

            X_raw, X_next_raw, g_next, terminal = self._build_dataset()
            Phi, Phi_next = self._fit_features(X_raw, X_next_raw)

            n_features = Phi.shape[1]
            w = np.zeros(n_features)

            iters = tqdm(range(self.iterations), desc="FQI iterations") if not quiet else range(self.iterations)

            for it in iters:
                if self.debug:
                    print(f"\n=== FQI iteration {it} ===")
                w_new = self._fqi_iteration(Phi, Phi_next, g_next, terminal, w)
                if np.linalg.norm(w_new - w) < self.tol:
                    if self.debug:
                        print(f"[FQI] converged at iter {it}")
                    w = w_new
                    break
                w = w_new

            self.w = w

            # оценка политики
            values = self._evaluate_policy()
            self.option_price = values

            price0 = float(values.mean())
            history = np.full(self.sampler.cnt_times, np.nan)
            history[0] = price0
            self.price_history = history

            self.result["train"] = {
                "price": price0,
                "std": float(values.std()),
            }

            return history

        # TEST
        if self.w is None:
            raise RuntimeError("Train first!")

        self.sampler.sample()
        values = self._evaluate_policy()
        self.option_price = values

        price0 = float(values.mean())
        history = np.full(self.sampler.cnt_times, np.nan)
        history[0] = price0
        self.price_history = history

        self.result["test"] = {
            "price": price0,
            "std": float(values.std()),
        }

        return history

    def plot_expected_prices(self):
        pass

    def plot_sample(self):
        pass
