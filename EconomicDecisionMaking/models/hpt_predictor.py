## Adding package to PATH
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(dirname(abspath(__file__)))))

## Imports
from EconomicDecisionMaking.models.ev_based_model import *
from EconomicDecisionMaking.models.pt_predictor  import PTModel

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
    def gain(self, n: int, price: int, j: int, xg: float, g: float) -> float:
        """Returns the subjective gain from selling AMOUNT goods at price PRICE rather than price J.
        Inputs:
            n: the amount of good to be hypothetically sold.
            price: the current price.
            j: a hypothetical price that might occur.
            xg: hyperbolic gain variable, xl-.1 <= xg <= xl.
            g: overwighting of low probabilities, 0 <= g <= 1."""
        upper: float = n * (price - j)
        lower: float = 1 + (n * (price -j) * xg)
        return prelec(p(j), g) * (upper / lower)

    @lru_cache(maxsize=CACHE_SIZE)
    def loss(self, n: int, price: int, j: int, xl: float, l: float, g: float) -> float:
        """Returns the subjective loss from selling AMOUNT goods at price PRICE rather than price J.
        Inputs:
            n: the amount of good to be hypothetically sold.
            price: the current price.
            j: a hypothetical price that might occur.
            xl: hyperbolic loss variable, 0 <= xl <= .1.
            l: coefficient of loss sensitivity, l >= 1.
            g: overwighting of low probabilities, 0 <= g <= 1."""
        upper2: float = n * (j - price)
        lower2: float = 1 + (n * (j - price) * xl)
        return prelec(p(j), g) * l * (upper2 / lower2)

    @lru_cache(maxsize=CACHE_SIZE)
    def get_term_3a(self, day: int, k: int, cutoffs: Tuple[int], g: float):
        """Gets a particular term from equation (6).
        Moved to a seperate function to support caching."""
        term_3a = 1.0
        for h in range(1, k + 1):
            # print(cutoffs)
            # print(day, k, h)
            sum_3a = 0.0
            for j in range(1, cutoffs[day - h]):
                sum_3a += prelec(p(j), g)
            term_3a *= sum_3a
        return term_3a
    
    @lru_cache(maxsize=CACHE_SIZE)
    def expected_value(self, day: int, price: int, n: int, fit: Parameters, cutoffs: Tuple[int]):
        ev: float = 0

        # First term
        for j in range(cutoffs[day - 1], price):
            ev += self.gain(n, price, j, fit.xg, fit.g)

        # Second term
        lower_bound = max(price + 1, cutoffs[day - 1])
        for j in range(lower_bound, 16):
            ev -= self.loss(n, price, j, fit.xl, fit.l, fit.g)

        # Third Term
        bound = day + 1
        for k in range(1, bound + 1):
            # Term 3a
            term_3a = self.get_term_3a(day, k - 1, cutoffs, fit.g)

            # Term 3b
            term_3b = 0.0
            for j in range(cutoffs[day - k], price):
                term_3b += self.gain(n, price, j, fit.xg, fit.g)

            # Term 3c
            term_3c = 0.0
            lower_bound_3c = max(price + 1, cutoffs[day - k])
            for j in range(lower_bound, 16):
                term_3c += self.loss(n, price, j, fit.xl, fit.l, fit.g)

            ev += term_3a * (term_3b - term_3c)
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

def TEST_check_for_pt() -> None:
    """If HPT is coded properly, when g=l=1.0 & xl=xg=0, it should collapse
    to the predictions of PT. This function, when run, simply asserts that this is true."""
    # Error type can be "absolute" or "proportional"
    error_type = "proportional"

    # Initialize model
    hpt_model = HPTModel()

    for subject in range(hpt_model.num_subjects):
        hpt_model.best_fits[subject] = Parameters(xl=0, xg=0, g=1.0, l=1.0, tw=68)

    hpt_mean_error    = hpt_model.finalize_and_mean_error(error_type=error_type)
    hpt_std_deviation = hpt_model.std_dev_of_error(error_type=error_type)

    pt_model = PTModel()
    
    for subject in range(pt_model.num_subjects):
        pt_model.best_fits[subject] = Parameters(a=0, b=0, g=1.0, l=1.0, tw=68)
    
    pt_mean_error    = pt_model.finalize_and_mean_error(error_type=error_type)
    pt_std_deviation = pt_model.std_dev_of_error(error_type=error_type)

    # Prints
    print(f'hpt mean_error = {hpt_mean_error}')
    print(f'hpt std_dev = {hpt_std_deviation}')
    print(f'pt mean_error = {pt_mean_error}')
    print(f'pt std_dev = {pt_std_deviation}')

    pt_errors: List[float] = pt_model.mean_error_all_subjects(error_type,
                                                           False,
                                                           save_predictions=True)

    for subject in trange(hpt_model.num_subjects):
        assert list(hpt_model.data.loc[subject]['prediction']) == list(pt_model.data.loc[subject]['prediction'])

    print("HPT is equivalent to PT when g==l==1.0 & xl==xg==0!")

if __name__ == '__main__':
    ### Model name (to save to data dir)
    version = "exhaustive_0-5_930_prop"

    TEST_check_for_pt()
    main(version=version)
