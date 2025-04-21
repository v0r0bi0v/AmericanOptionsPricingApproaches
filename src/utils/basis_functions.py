from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import itertools
import typing as tp


# @njit
def jitted_transform(x: np.ndarray, combinations: tp.List[tp.Tuple[int, ...]]):
    x_ = np.hstack((np.ones((x.shape[0], 1)), x))

    transformed_data = []
    for combination in combinations:
        product = np.prod(x_[:, list(combination)], axis=1).reshape(-1, 1)
        transformed_data.append(product)

    return np.hstack(transformed_data)


class PolynomialTransformer(TransformerMixin, BaseEstimator) :
    def __init__ (
        self,
        deg=2
    ):
        self.deg = deg

    def fit(self, *_):
        return self

    def transform(
        self,
        x: np.ndarray
    ):
        combinations = itertools.combinations_with_replacement(range(x.shape[1] + 1), self.deg)
        return jitted_transform(x, list(combinations) )
