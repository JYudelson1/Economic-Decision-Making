import pandas as pd
import numpy as np
from math import exp
from ev_based_model import *
import pickle as pkl

pd.options.mode.chained_assignment = None  # default='warn'


class HPTTWModel(EVModel):
    """This model implements the predictions of Prospect Theory, with Arnold Glass's Time Window"""

    def __init__(self):
        super().__init__()
        self.free_params: list[str] = ["xg", "xl", "g", "l", "tw"]

    @lru_cache(maxsize=CACHE_SIZE)
    def get_psalesj(self, j, tw):
        """Finds psales,j that is plugged into Prelec funtion"""
        psalesj: float = 0
        for k in range(0, tw - 1):
            ph: float = 0
            for h in range(1, j):
                ph += p(h)
            psalesj += (ph ** k)
        return psalesj

    @lru_cache(maxsize=CACHE_SIZE)
    def prelec(self, p: float, g: float) -> float:
        """Prelec function"""
        neg_log: float = -np.log(p + np.finfo(float).eps)
        return exp(-(neg_log ** g))

    @lru_cache(maxsize=CACHE_SIZE)
    def expected_value(self, day: int, price: int, n: int, fit: Parameters, cutoffs: Tuple[int]):
        psalesj = 0
        minuend = 0.0
        price = int(price)
        n = int(n)
        for j in range(1, price):
            psalesj += self.get_psalesj(j, fit.tw) * p(j)
            wp = prelec(psalesj, fit.g)
            minuend += (wp * (n * (cutoffs[day - 1] - j))) / (1 + (n * (cutoffs[day - 1] - j)) * fit.xg)

        subtrahend = 0.0
        price = int(price)
        n = int(n)
        for j in range(price + 1, 16):
            psalesj = self.get_psalesj(j, fit.tw) * p(j)
            wp = prelec(psalesj, fit.g)
            subtrahend += (wp * fit.l * (n * (j - cutoffs[day - 1]))) / (1 + (n * (cutoffs[day - 1] - j) * fit.xl))
        ev = minuend - subtrahend
        return ev

    @lru_cache(maxsize=CACHE_SIZE)
    def expected_value_day_2(self, price: int, n: int, fit: Parameters):
        minuend = 0.0
        price = int(price)
        n = int(n)
        for j in range(1, price):
            wp = prelec(p(j), fit.g)
            minuend += (wp * (n * (price - j))) / (1 + (n * (price - j)) * fit.xg)

        subtrahend = 0.0
        price = int(price)
        n = int(n)
        for j in range(price + 1, 16):
            wp = prelec(p(j), fit.g)
            subtrahend += (wp * fit.l * (n * (j - price))) / (1 + (n * (price - j) * fit.xl))
        ev = minuend - subtrahend
        return ev

def main() -> None:
    # model name (to save to data dir)
    version = "exhaustive_0-5_930_prop"

    # Error type can be "absolute" or "proportional"
    error_type = "proportional"

    # Initialize model
    model = HPTTWModel()

    # Run fitting
    model.exhaustive_fit(precision=0.5, verbose=True, error_type=error_type)

    mean_error = model.finalize_and_mean_error(error_type=error_type)
    std_deviation = model.std_dev_of_error(error_type=error_type)

    # Prints
    print(f'mean_error = {mean_error}')
    print(f'std_dev = {std_deviation}')
    print(model.data)

    # Saves data
    with open(f'{DATA_DIR}/hpt_{version}.pkl', "wb") as f:
        pkl.dump(model, f)


if __name__ == '__main__':
    main()
