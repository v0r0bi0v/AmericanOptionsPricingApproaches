import numpy as np
from tqdm.auto import tqdm

from src.pricers.abstract_pricer import PricerAbstract
from src.samplers.abstract_sampler import SamplerAbstract


class LSPIPricerDiploma(PricerAbstract):

    def __init__(
        self,
        sampler: SamplerAbstract,
        iterations: int = 25,
        lambda_reg: float = 1e-2,
        tol: float = 1e-6,
        debug: bool = False,
        basis_version: str = "time_buckets",
        n_time_buckets: int = 4,
    ):
        self.sampler = sampler
        self.iterations = iterations
        self.lambda_reg = lambda_reg
        self.tol = tol
        self.debug = debug
        self.basis_version = basis_version
        self.n_time_buckets = n_time_buckets

        self.gamma: float | None = None
        self.w: np.ndarray | None = None
        self.price_history: np.ndarray | None = None
        self.option_price: np.ndarray | None = None
        self.result: dict = {}

        # std нормировка для feature-ов
        self.feature_std: np.ndarray | None = None


    def _basis_ST(self, S: np.ndarray, t: np.ndarray) -> np.ndarray:
        S = np.asarray(S, dtype=float)
        t = np.asarray(t, dtype=float)

        K = getattr(self.sampler, "strike")
        T = getattr(self.sampler, "t", self.sampler.time_grid[-1])

        S_norm = S / K

        # базовый Laguerre-блок по S (Li 2009)
        f0 = np.ones_like(S_norm)
        exp_term = np.exp(-S_norm / 2.0)
        f1 = exp_term
        f2 = exp_term * (1.0 - S_norm)
        f3 = exp_term * (1.0 - 2.0 * S_norm + 0.5 * S_norm**2)

        # payoff-подобные фичи для пута
        intrinsic = np.maximum(K - S, 0.0)    # (K - S)+
        intrinsic2 = intrinsic**2 / K        # квадратичная версия

        base_S = [f0, f1, f2, f3, intrinsic, intrinsic2]

        if self.basis_version == "original":
            tau = t / T
            phi0_t = np.sin(-t * np.pi / (2.0 * T) + np.pi / 2.0)
            phi1_t = np.log(np.clip(T - t, 1e-12, None))
            phi2_t = tau ** 2
            return np.stack(
                [f0, f1, f2, f3, intrinsic, intrinsic2, phi0_t, phi1_t, phi2_t],
                axis=-1,
            )

        if self.basis_version == "time_buckets":
            tau = t / T
            u = np.linspace(0.0, 1.0, self.n_time_buckets + 1)
            edges = 1.0 - (1.0 - u) ** 2

            features = []
            for k in range(self.n_time_buckets):
                left, right = edges[k], edges[k + 1]
                mask_k = ((tau >= left) & (tau < right)).astype(float)
                if k == self.n_time_buckets - 1:
                    mask_k = ((tau >= left) & (tau <= right)).astype(float)

                for fi in base_S:
                    features.append(fi * mask_k)

            return np.stack(features, axis=-1)

        raise ValueError(f"Unknown basis_version: {self.basis_version}")


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
        g_next = payoff[:, 1:].reshape(-1)

        # terminal: когда next index — последний шаг
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

    def _lspi_iteration(self, Phi, Phi_next, g_next, terminal, w):

        gamma = self.gamma
        n_samples, n_features = Phi.shape

        # continuation value с текущей политикой w в s'
        cont_next = Phi_next @ w
        ex_next = g_next

        # --- отладка по next-состояниям ---
        if self.debug:
            mask_otm_next = (g_next == 0.0) & (~terminal)
            mask_itm_next = (g_next > 0.0) & (~terminal)

            def stats(x, mask, name):
                if mask.any():
                    vals = x[mask]
                    print(
                        f"[LSPI] {name}: "
                        f"min={vals.min():.4f}, "
                        f"p1={np.percentile(vals,1):.4f}, "
                        f"p5={np.percentile(vals,5):.4f}, "
                        f"mean={vals.mean():.4f}, "
                        f"p95={np.percentile(vals,95):.4f}, "
                        f"max={vals.max():.4f}"
                    )
                else:
                    print(f"[LSPI] {name}: no samples")

            stats(cont_next, mask_otm_next, "cont_next | OTM")
            stats(cont_next, mask_itm_next, "cont_next | ITM")

            mask_neg = (cont_next < 0)
            print(f"[LSPI] fraction cont_next < 0 overall: {mask_neg.mean():.4f}")
            print(
                "[LSPI] fraction cont_next < 0 in OTM: "
                f"{(mask_otm_next & (cont_next < 0)).mean():.4f}"
            )

        # C1: не terminal и политика выбирает continue в s'
        C1 = (~terminal) & (cont_next >= ex_next)

        # C2: terminal или политика выбирает exercise в s'
        C2 = terminal | (cont_next < ex_next)

        if self.debug:
            print(f"[LSPI] C1 (continue at s') fraction: {C1.mean():.3f}")
            print(f"[LSPI] C2 (exercise/terminal at s') fraction: {C2.mean():.3f}")

        # Матрица Y
        Y = Phi.copy()
        Y[C1] -= gamma * Phi_next[C1]

        # Вектор v
        v = np.zeros(n_samples, dtype=float)
        v[C2] = gamma * g_next[C2]

        # Строим A и b
        A = Phi.T @ Y
        if self.lambda_reg > 0:
            A += self.lambda_reg * np.eye(n_features)
        b = Phi.T @ v

        # --- численная диагностика A,w ---
        if self.debug:
            try:
                s = np.linalg.svd(A, compute_uv=False)
                cond_A = float(s[0] / (s[-1] + 1e-18))
            except Exception:
                cond_A = np.inf
            print(f"[LSPI] cond(A): {cond_A:.3e}")

        # Решаем A w' = b
        w_new = np.linalg.pinv(A, rcond=1e-10) @ b

        if self.debug:
            print(
                f"[LSPI] w_new: norm={np.linalg.norm(w_new):.4f}, "
                f"min={w_new.min():.4f}, max={w_new.max():.4f}"
            )
            col_norms = np.linalg.norm(Phi, axis=0)
            print(
                f"[LSPI] Phi column norms: min={col_norms.min():.4e}, "
                f"max={col_norms.max():.4e}"
            )

        return w_new

    def _evaluate_policy(self):

        S = self.sampler.markov_state[:, :, 0]
        payoff = self.sampler.payoff
        disc = self.sampler.discount_factor
        time_grid = self.sampler.time_grid

        n_traj, n_times = payoff.shape

        # compute continuation value for all states except terminal
        S_curr = S[:, :-1]
        t_curr = np.broadcast_to(time_grid[:-1], S_curr.shape)

        Phi_curr_raw = self._basis_ST(
            S_curr.reshape(-1),
            t_curr.reshape(-1),
        )

        if self.feature_std is None:
            raise RuntimeError("Features std not set. Train LSPI first.")

        Phi_curr = Phi_curr_raw / self.feature_std

        cont_flat = Phi_curr @ self.w
        cont = cont_flat.reshape(n_traj, n_times - 1)

        # условие раннего exercise
        exercise_cond = payoff[:, :-1] >= cont

        # ---- диагностика "плохих" упражнений: payoff == 0 и cont < 0 ----
        if self.debug:
            mask_bad = (payoff[:, :-1] == 0.0) & (cont < 0.0) & exercise_cond
            frac_bad = mask_bad.mean()
            print(f"[policy] payoff==0 & cont<0 & exercise fraction: {frac_bad:.3f}")
            if frac_bad > 0:
                frac_bad_by_time = mask_bad.mean(axis=0)
                print(
                    "[policy] first 10 time steps bad fraction:",
                    np.array2string(frac_bad_by_time[:10], precision=3),
                )

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

        # если никогда не exercise → упражняем в maturity
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
        """
        Возвращает:
        - values: дисконтированные стоимости на траекториях
        - exercise_time_idx: индекс времени упражнения на каждой траектории
        - cont: continuation value (матрица [n_traj, n_times-1])
        - exercise_cond: bool-маска exercise по всем (S,t)
        """
        S = self.sampler.markov_state[:, :, 0]
        payoff = self.sampler.payoff
        disc = self.sampler.discount_factor
        time_grid = self.sampler.time_grid

        n_traj, n_times = payoff.shape

        S_curr = S[:, :-1]
        t_curr = np.broadcast_to(time_grid[:-1], S_curr.shape)

        Phi_curr_raw = self._basis_ST(
            S_curr.reshape(-1),
            t_curr.reshape(-1),
        )

        if self.feature_std is None:
            raise RuntimeError("Features std not set. Train LSPI first.")

        Phi_curr = Phi_curr_raw / self.feature_std
        cont_flat = Phi_curr @ self.w
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

    def price(self, test: bool = False, quiet: bool = False, ax=None):

        if not test:
            self.sampler.sample()

            X_raw, X_next_raw, g_next, terminal = self._build_dataset()
            Phi_raw, Phi_next_raw = self._fit_features(X_raw, X_next_raw)

            # нормировка только по std
            Phi_std = Phi_raw.std(axis=0)
            Phi_std[Phi_std < 1e-8] = 1.0  # чтобы не делить на ноль

            Phi = Phi_raw / Phi_std
            Phi_next = Phi_next_raw / Phi_std

            self.feature_std = Phi_std

            n_features = Phi.shape[1]
            w = np.zeros(n_features)

            iters = tqdm(range(self.iterations), desc="LSPI iterations") if not quiet else range(self.iterations)

            for it in iters:
                if self.debug:
                    print(f"\n=== LSPI iteration {it} ===")
                w_new = self._lspi_iteration(Phi, Phi_next, g_next, terminal, w)
                if np.linalg.norm(w_new - w) < self.tol:
                    if self.debug:
                        print(f"[LSPI] converged at iter {it}")
                    w = w_new
                    break
                w = w_new

            self.w = w

            # evaluate policy
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

        # --- TEST ---
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

    def compute_exercise_boundary(
        self,
        S_min: float | None = None,
        S_max: float | None = None,
        n_S: int = 200,
    ):

        if self.w is None or self.feature_std is None:
            raise RuntimeError("Run LSPI training (price(test=False)) before calling compute_exercise_boundary().")

        K = getattr(self.sampler, "strike")
        time_grid = self.sampler.time_grid
        T = time_grid[-1]
        n_times = len(time_grid)

        # диапазон по S, если не задан
        if S_min is None:
            S_min = 0.0
        if S_max is None:
            # что-то разумное, например 2.5 * K
            S_max = 2.5 * K

        S_grid = np.linspace(S_min, S_max, n_S)
        t_grid = time_grid

        S_star = np.full(n_times, np.nan, dtype=float)

        # просматриваем все моменты времени, кроме последнего
        for j, t in enumerate(time_grid[:-1]):
            t_vec = np.full_like(S_grid, t, dtype=float)

            # фичи и continuation
            Phi_raw = self._basis_ST(S_grid, t_vec)
            Phi = Phi_raw / self.feature_std
            cont = Phi @ self.w

            # payoff пута
            payoff = np.maximum(K - S_grid, 0.0)

            # policy: exercise, если payoff >= cont
            exercise_mask = payoff >= cont

            if not exercise_mask.any():
                # никогда не упражняем в этот момент
                S_star[j] = np.nan
                continue

            if exercise_mask.all():
                # всегда упражняем: граница — правый край сетки
                S_star[j] = S_grid[-1]
                continue

            # последний индекс, где ещё exercise=True
            idx_last_ex = np.where(exercise_mask)[0][-1]

            # для красоты можно сделать линейную интерполяцию
            if idx_last_ex == n_S - 1:
                S_star[j] = S_grid[idx_last_ex]
            else:
                s_left = S_grid[idx_last_ex]
                s_right = S_grid[idx_last_ex + 1]
                p_left = payoff[idx_last_ex] - cont[idx_last_ex]
                p_right = payoff[idx_last_ex + 1] - cont[idx_last_ex + 1]

                if p_right == p_left:
                    S_star[j] = s_left
                else:
                    # корень по секущей
                    frac = p_left / (p_left - p_right)
                    S_star[j] = s_left + frac * (s_right - s_left)

        # в maturity граница для американского пута — примерно strike
        S_star[-1] = K

        return t_grid, S_star

    def plot_exercise_boundary(
        self,
        S_min: float | None = None,
        S_max: float | None = None,
        n_S: int = 200,
        ax=None,
    ):
        """
        Рисует S*(t) для текущей политики LSPI.
        """
        import matplotlib.pyplot as plt

        t_grid, S_star = self.compute_exercise_boundary(
            S_min=S_min,
            S_max=S_max,
            n_S=n_S,
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(t_grid, S_star, marker="o")
        ax.set_title("Exercise boundary $S^*(t)$ from LSPI policy")
        ax.set_xlabel("t")
        ax.set_ylabel("S*(t)")
        ax.grid(True)

        return ax
    
    
    def continuation_value(self, S, t):
        """
        Continuation value C(S,t) = φ(S,t)^T w для текущей политики LSPI.
        Работает и для скаляров, и для numpy-массивов.
        """
        if self.w is None or self.feature_std is None:
            raise RuntimeError(
                "Run LSPI training (price(test=False)) before calling continuation_value()."
            )

        S_arr = np.asarray(S, dtype=float)
        t_arr = np.asarray(t, dtype=float)

        if t_arr.ndim == 0:
            t_arr = np.full_like(S_arr, t_arr, dtype=float)
        else:
            t_arr = np.broadcast_to(t_arr, S_arr.shape)

        Phi_raw = self._basis_ST(S_arr, t_arr)
        Phi = Phi_raw / self.feature_std
        return Phi @ self.w

