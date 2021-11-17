## Adding package to PATH
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(dirname(abspath(__file__)))))

## Imports
from EconomicDecisionMaking.models.ev_based_model import *
from EconomicDecisionMaking.models.eut_predictor  import EUTModel

pd.options.mode.chained_assignment = None  # default='warn'

## Classes
class HPTModel(EVModel):
    """This model implements the predictions of Prospect Theory, with Arnold Glass's Time Window"""

    def __init__(self):
        super().__init__()
        self.free_params: list[str] = ["xg", "xl", "g", "l"]

    @lru_cache(maxsize=CACHE_SIZE)
    def get_psalesj(self, j, tw, day, cutoffs):
        """Finds psales,j that is plugged into Prelec funtion"""
        psalesj: float = 0
        for k in range(0, day - 1):
            ph: float = 0
            for h in range(1, j):
                ph += p(h)
            psalesj += (ph ** k)
        return psalesj

    @lru_cache(maxsize=CACHE_SIZE)
    def expected_value(self, day: int, price: int, n: int, fit: Parameters, cutoffs: Tuple[int]):
        minuend = 0.0

        for j in range(1, price):
            psalesj = self.get_psalesj(j, fit.tw, day, cutoffs) * p(j)
            wp = prelec(psalesj, fit.g)
            minuend += (wp * n * (price - j)) / (1 + (n * (price - j)) * fit.xg)

        subtrahend = 0.0

        for j in range(price + 1, 16):
            psalesj = self.get_psalesj(j, fit.tw, day, cutoffs) * p(j)
            wp = prelec(psalesj, fit.g)
            subtrahend += (wp * fit.l * n * (j - price)) / (1 + (n * (j - price) * fit.xl))

        ev = minuend - subtrahend
        return ev

    @lru_cache(maxsize=CACHE_SIZE)
    def expected_value_day_2(self, price: int, n: int, fit: Parameters, cutoffs: Tuple[int]):
        return self.expected_value(1, price, n, fit, (1,))

def main(version: str) -> None:

    ### Initialize model
    model = HPTModel()

    ### Run fitting
    model.exhaustive_fit(precision=0.5, verbose=True)

    mean_error    = model.finalize_and_mean_error()
    std_deviation = model.std_dev_of_error()

    ### Prints
    print(f'mean_error = {mean_error}')
    print(f'std_dev = {std_deviation}')
    print(model.data)

    ### Saves data
    with open(f'{DATA_DIR}/hpt_{version}.pkl', "wb") as f:
        pkl.dump(model, f)

def TEST_check_for_eut() -> None:
    """If HPT is coded properly, when g=l=1.0 & xl=xg=0, it should collapse
    to the predictions of EUT. This function, when run, simply asserts that this is true."""
    # Error type can be "absolute" or "proportional"
    error_type = "proportional"

    # Initialize model
    hpt_model = HPTModel()

    for subject in range(hpt_model.num_subjects):
        hpt_model.best_fits[subject] = Parameters(xl=0, xg=0, g=1.0, l=1.0, tw=68)

    hpt_mean_error    = hpt_model.finalize_and_mean_error(error_type=error_type)
    hpt_std_deviation = hpt_model.std_dev_of_error(error_type=error_type)

    eut_model = EUTModel()
    eut_mean_error    = eut_model.finalize_and_mean_error(error_type=error_type)
    eut_std_deviation = eut_model.std_dev_of_error(error_type=error_type)

    # Prints
    print(f'hpt mean_error = {hpt_mean_error}')
    print(f'hpt std_dev = {hpt_std_deviation}')
    print(f'eut mean_error = {eut_mean_error}')
    print(f'eut std_dev = {eut_std_deviation}')

    eut_errors: List[float] = eut_model.mean_error_all_subjects(error_type,
                                                           False,
                                                           save_predictions=True)

    for subject in trange(hpt_model.num_subjects):
        assert list(hpt_model.data.loc[subject]['prediction']) == list(eut_model.data.loc[subject]['prediction'])

    print("HPT is equivalent to EUT when g==l==1.0 & xl==xg==0!")

if __name__ == '__main__':
    ### Model name (to save to data dir)
    version = "exhaustive_0-5_930_prop"

    TEST_check_for_eut()
    main(version=version)
