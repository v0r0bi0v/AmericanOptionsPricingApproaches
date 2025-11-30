import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from IPython.display import display, clear_output

from src.pricers.abstract_pricer import PricerAbstract
from src.samplers.abstract_sampler import SamplerAbstract


def _plot_progress(sampler, bar, price_history, lower_bound, upper_bound, ax=None):
    clear_output(wait=True)

    container = getattr(bar, "container", None)
    if container is not None:
        display(container)
    else:
        bar.refresh()

    if ax is None:
        ax = plt.gca()
    ax.ticklabel_format(style='plain', useOffset=False)
    ax.plot(sampler.time_grid, price_history)
    ax.plot(sampler.time_grid, lower_bound, "--")
    ax.plot(sampler.time_grid, upper_bound, "-.")
    ax.legend()
    ax.set_title("$Option_t$")
    ax.set_xlabel("$t$")
    ax.set_ylabel("price")
    ax.grid()


class AmericanMonteCarloPricer(PricerAbstract):
    def __init__(
            self,
            sampler: SamplerAbstract,
            degree: int = 3,
            regularization_alpha: float = 1e-4,
            debug: bool = False,                 # <-- добавим флаг
    ):
        self.sampler = sampler
        self.regularization_alpha = regularization_alpha
        self.basis_functions_transformer = PolynomialFeatures(degree=degree)
        self._poly_fitted = False
        self.price_history: np.ndarray | None = None
        self.option_price: np.ndarray | None = None
        self.result = {}
        self.weights: list = []
        self.scalers: list = []

        self.debug = debug                     # <-- сохраним
        self.exercise_time_idx: np.ndarray | None = None  # <-- сюда запишем τ по траекториям

    def price(self, test=False, quiet=False, ax=None):
        self.sampler.sample()
        discounted_payoff = self.sampler.payoff * self.sampler.discount_factor

        self.option_price = discounted_payoff[:, -1].copy()

        if not test:
            self.weights = [None] * self.sampler.cnt_times
            self.scalers = [None] * self.sampler.cnt_times
            self._poly_fitted = False

        n_traj = self.sampler.cnt_trajectories
        n_times = self.sampler.cnt_times

        # изначально считаем, что все исполняются в T
        exercise_time_idx = np.full(n_traj, n_times - 1, dtype=int)

        self.price_history = [None] * (n_times - 1) + [self.option_price.mean()]

        lower_bound = np.zeros(n_times)
        upper_bound = np.zeros(n_times)
        for i in range(n_times):
            lower_bound[i] = discounted_payoff[:, i:].mean(axis=0).max()
            upper_bound[i] = discounted_payoff[:, i:].max(axis=1).mean()

        bar = tqdm(range(n_times - 2, -1, -1),
                   desc=f"AMC price {'test' if test else 'train'}")
        for time_index in bar:
            if time_index == 0:
                continuation_value = (
                    np.ones(n_traj) * np.mean(self.option_price)
                )
                in_the_money_indices = np.arange(n_traj, dtype=int)
            else:
                in_the_money_indices = np.where(discounted_payoff[:, time_index] > 1e-9)[0]
                if (len(in_the_money_indices) / n_traj < 1e-2 or
                        len(in_the_money_indices) < 2 or
                        (test and self.weights[time_index] is None)):
                    self.price_history[time_index] = self.option_price.mean()
                    continue

                features = self.sampler.markov_state[in_the_money_indices, time_index].copy()

                # StandardScaler: train -> fit, test -> только transform
                if not test:
                    self.scalers[time_index] = StandardScaler()
                    self.scalers[time_index].fit(features)
                features = self.scalers[time_index].transform(features)

                # PolynomialFeatures: один раз fit, дальше только transform
                if (not self._poly_fitted) and (not test):
                    self.basis_functions_transformer.fit(features)
                    self._poly_fitted = True
                if not self._poly_fitted:
                    raise RuntimeError(
                        "PolynomialFeatures is not fitted: call price(test=False) first"
                    )

                transformed = self.basis_functions_transformer.transform(features)

                if not test:
                    regularization = (
                        np.eye(transformed.shape[1], dtype=float)
                        * self.regularization_alpha
                    )
                    inv = np.linalg.pinv(
                        (transformed.T @ transformed + regularization),
                        rcond=1e-10
                    )
                    self.weights[time_index] = (
                        inv @ transformed.T @ self.option_price[in_the_money_indices]
                    )

                continuation_value = transformed @ self.weights[time_index]

            # --- ключевой шаг: решаем exercise vs continue ---
            indicator = discounted_payoff[in_the_money_indices, time_index] > continuation_value

            # у тех, кто сейчас exercise, время остановки = текущий индекс
            exercised_idx = in_the_money_indices[indicator]
            exercise_time_idx[exercised_idx] = time_index

            self.option_price[in_the_money_indices] = (
                indicator * discounted_payoff[in_the_money_indices, time_index].copy()
                + ~indicator * self.option_price[in_the_money_indices]
            )

            self.price_history[time_index] = self.option_price.mean()
            if not quiet and time_index % 10 == 0:
                _plot_progress(
                    self.sampler, bar, self.price_history, lower_bound, upper_bound, ax=ax
                )

        # сохраним времена исполнения
        self.exercise_time_idx = exercise_time_idx

        if self.debug:
            t_ex = self.sampler.time_grid[exercise_time_idx]
            frac_early = np.mean(exercise_time_idx < (n_times - 1))
            print(f"[AMC] mean exercise time (years): {t_ex.mean():.4f}")
            print(f"[AMC] fraction exercised before maturity: {frac_early:.3f}")

        key = "test" if test else "train"
        self.result[key] = {
            "price": float(self.option_price.mean()),
            "upper_bound": float(discounted_payoff.max(axis=1).mean()),
            "lower_bound": float(discounted_payoff.mean(axis=0).max()),
            "std": float(self.option_price.std())
        }

        return self.price_history
