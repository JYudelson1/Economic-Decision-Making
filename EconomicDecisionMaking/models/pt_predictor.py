## Adding package to PATH
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(dirname(abspath(__file__)))))

## Imports
from EconomicDecisionMaking.models.ev_based_model import *
from EconomicDecisionMaking.models.eut_predictor  import EUTModel
from numba import jit

pd.options.mode.chained_assignment = None  # default='warn'

## Classes
class PTModel(EVModel):
    """This model implements the predictions of Prospect Theory"""

    def __init__(self):
        super().__init__()
        self.free_params: List[str] = ["a", "b", "g", "l"]

    @lru_cache(maxsize=CACHE_SIZE)
    def gain(self, amount: int, price: int, j: int, a: float, g: float) -> float:
        """Returns the subjective gain from selling AMOUNT goods at price PRICE rather than price J.
        Inputs:
            amount: the amount of good to be hypothetically sold.
            price: the current price.
            j: a hypothetical price that might occur.
            a: gain sensitivity, 0 <= a <= 1.
            g: overwighting of low probabilities, 0 <= g <= 1."""
        inner_bracket1: float = amount * (price - j)
        inner_bracket1 = inner_bracket1 ** a
        return inner_bracket1 * prelec(p(j), g)

    @lru_cache(maxsize=CACHE_SIZE)
    def loss(self, amount: int, price: int, j: int, b: float, l: float, g: float) -> float:
        """Returns the subjective loss from selling AMOUNT goods at price PRICE rather than price J.
        Inputs:
            amount: the amount of good to be hypothetically sold.
            price: the current price.
            j: a hypothetical price that might occur.
            b: loss sensitivity, 0 <= a <= 1.
            l: coefficient of loss sensitivity, l >= 1.
            g: overwighting of low probabilities, 0 <= g <= 1."""
        inner_bracket2: float = amount * (j - price)
        inner_bracket2 = inner_bracket2 ** b
        return inner_bracket2 * prelec(p(j), g) * l

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
    def expected_value(self, day: int, price: int, amount: int, fit: Parameters, cutoffs: Tuple[int]) -> float:
        """A straightforward implementation of equation (6).
        See ev_based_model.py for more notes."""
        ev: float = 0

        # First term
        for j in range(cutoffs[day - 1], price):
            ev += self.gain(amount, price, j, fit.a, fit.g)

        # Second term
        lower_bound = max(price + 1, cutoffs[day - 1])
        for j in range(lower_bound, 16):
            ev -= self.loss(amount, price, j, fit.b, fit.l, fit.g)

        # Third Term
        bound = day + 1
        for k in range(1, bound + 1):
            # Term 3a
            term_3a = self.get_term_3a(day, k-1, cutoffs, fit.g)

            # Term 3b
            term_3b = 0.0
            for j in range(cutoffs[day - k], price):
                term_3b += self.gain(amount, price, j, fit.a, fit.g)

            # Term 3c
            term_3c = 0.0
            lower_bound_3c = max(price + 1, cutoffs[day - k])
            for j in range(lower_bound, 16):
                term_3c += self.loss(amount, price, j, fit.b, fit.l, fit.g)

            ev += term_3a * (term_3b - term_3c)

        return ev

    @lru_cache(maxsize=CACHE_SIZE)
    def expected_value_day_2(self, price: int, amount: int, fit: Parameters, cutoffs: Tuple[int]) -> float:
        """A straightforward implementation of equation (4).
        See ev_based_model.py for more notes."""

        return self.expected_value(1, price, amount, fit, (1,))

    def error_direction(self, subject: int, fit: Parameters) -> int:
        predictions: List[int] = self.predict_one_subject(subject, fit)
        data: List[int] = list(self.get_data_one_subject(subject)['sold'])
        return sum([p-d for p,d in zip(predictions, data)])

    def fit_l(self, subject: int, precision: float, a: float, b: float, g: float) -> Tuple[Parameters, float]:
        best_fit = Parameters(a=a, b=b, g=g, l=1.0)
        lowest_error = self.error_of_fit(subject, best_fit)
        iterator = tqdm(np.arange(1, 3.5 + EPS, precision), desc=f'Fitting l', leave=False)
        for l in iterator:
            fit = Parameters(a=a, b=b, g=g, l=l)
            error = self.error_of_fit(subject, fit)
            if error < lowest_error:
                lowest_error = error
                best_fit = fit
            if error == 0:
                return (best_fit, lowest_error)
            error_dir = self.error_direction(subject, fit)
            if error_dir <= 0:
                iterator.close()
                break
        return (best_fit, lowest_error)

    def fit_a(self, subject: int, precision: float, b: float, g: float) -> Tuple[Parameters, float]:
        best_fit, lowest_error = self.fit_l(subject, precision, a=b, b=b, g=g)
        iterator = tqdm(np.arange(precision, b + EPS, precision)[::-1], desc=f'Fitting a', leave=False)
        for a in iterator:
            fit, error = self.fit_l(subject, precision, a=a, b=b, g=g)
            if error < lowest_error:
                lowest_error = error
                best_fit = fit
            if error == 0:
                return (best_fit, lowest_error)
            error_dir = self.error_direction(subject, fit)
            #print(a, b, fit.l, error_dir)
            if error_dir <= 0:
                iterator.close()
                break
        return (best_fit, lowest_error)

    def fit_b(self, subject: int, precision: float, g: float) -> Tuple[Parameters, float]:
        best_fit, lowest_error = self.fit_a(subject, precision, b=1.0, g=g)
        iterator = tqdm(np.arange(precision, 1 + EPS, precision)[::-1], desc=f'Fitting b', leave=False)
        for b in iterator:
            fit, error = self.fit_a(subject, precision, b=b, g=g)
            if error < lowest_error:
                lowest_error = error
                best_fit = fit
            if error == 0:
                return (best_fit, lowest_error)
            error_dir = self.error_direction(subject, fit)
            if error_dir >= 0:
                iterator.close()
                break
        return (best_fit, lowest_error)

    def optimal_fit_one_subject(self, subject: int, precision: float, verbose: bool, start_fit: Parameters) -> None:
        lowest_error = self.error_of_fit(subject, start_fit)
        self.best_fits[subject] = start_fit
        for g in tqdm(np.arange(precision, 1 + EPS, precision)[::-1], desc=f'Fitting g', leave=False):
            #for b in tqdm(np.arange(precision, 1 + EPS, precision)[::-1], desc=f'Fitting b', leave=False):
                #for a in tqdm(np.arange(b, 1 + EPS, precision)[::-1], desc=f'Fitting a', leave=False):
            fit, error = self.fit_b(subject=subject, precision=precision, g=g)
            if error < lowest_error:
                lowest_error = error
                self.best_fits[subject] = fit
                print(f'best fit = {fit} || {error}')
            if error == 0:
                return

    def optimal_fit(self, precision: float, verbose: bool, start_fit: Parameters) -> None:
        """Does the optimal fit algorithm for all subjects. Modifies in place.
        Inputs:
            precision: the amount to increment each value when iterating through all possible values.
            verbose: set to True to get progress bars for the fitting.
            start_fit: the first parameters to use when traversing the search space."""
        for subject in trange(self.num_subjects, disable=(not verbose), desc="Optimal Fit"):
            if subject != 9: continue
            self.optimal_fit_one_subject(subject, precision, verbose, start_fit)
            print(subject, self.best_fits[subject], self.error_of_fit(subject, self.best_fits[subject]))

    def linear_fit_one_subject(self, subject: int, precision: float, verbose: bool, start_fit: Parameters, i: int) -> None:
        lowest_error = self.error_of_fit(subject, start_fit)
        self.best_fits[subject] = start_fit
        iter_a = tqdm(np.arange(precision, 1 + EPS, precision)[::-1], desc=f'Fitting a (i={i})', leave=False)
        for a in iter_a:
            fit = self.best_fits[subject].deepcopy()
            fit.a = a
            error = self.error_of_fit(subject, fit)
            if error < lowest_error:
                lowest_error = error
                self.best_fits[subject] = fit
            if error == 0:
                iter_a.close()
                return
        iter_b = tqdm(np.arange(self.best_fits[subject].a, 1 + EPS, precision)[::-1], desc=f'Fitting b (i={i})', leave=False)
        for b in iter_b:
            fit = self.best_fits[subject].deepcopy()
            fit.b = b
            error = self.error_of_fit(subject, fit)
            if error < lowest_error:
                lowest_error = error
                self.best_fits[subject] = fit
            if error == 0:
                iter_b.close()
                return
        iter_l = tqdm(np.arange(1, 3.5 + EPS, precision), desc=f'Fitting l (i={i})', leave=False)
        for l in iter_l:
            fit = self.best_fits[subject].deepcopy()
            fit.l = l
            error = self.error_of_fit(subject, fit)
            if error <= lowest_error:
                lowest_error = error
                self.best_fits[subject] = fit
            if error == 0:
                iter_l.close()
                return
        iter_g = tqdm(np.arange(precision, 1 + EPS, precision)[::-1], desc=f'Fitting g (i={i})', leave=False)
        for g in iter_g:
            fit = self.best_fits[subject].deepcopy()
            fit.g = g
            error = self.error_of_fit(subject, fit)
            if error < lowest_error:
                lowest_error = error
                self.best_fits[subject] = fit
            if error == 0:
                iter_g.close()
                return

    def recursive_linear_fit_one_subject(self, subject: int, precision: float, verbose: bool, start_fit: Parameters):
        i = 1
        lowest_error = self.error_of_fit(subject, start_fit)
        self.linear_fit_one_subject(subject, precision, verbose, start_fit, i)
        new_error = self.error_of_fit(subject, self.best_fits[subject])

        while new_error < lowest_error and new_error > 0:
            i += 1
            lowest_error = new_error
            self.linear_fit_one_subject(subject, precision, verbose, self.best_fits[subject], i)
            new_error = self.error_of_fit(subject, self.best_fits[subject])

    def linear_fit(self, precision: float, verbose: bool, start_fit: Parameters) -> None:
        for subject in trange(self.num_subjects, disable=(not verbose), desc="Linear Fit"):
            self.recursive_linear_fit_one_subject(subject, precision, verbose, start_fit)
            if verbose:
                print(subject, self.best_fits[subject], self.error_of_fit(subject, self.best_fits[subject]))

def main(version: str) -> None:

    ### Initialize model
    model = PTModel()

    ### Run fitting
    start_fit = Parameters(a=1.0, b=1.0, g=1.0, l=1.0)
    model.greedy_fit(verbose=True, precision=0.01, start_fit=start_fit)

    ### Print
    model.print_info()

    ### Saves data
    with open(f'{DATA_DIR}/pt_{version}.pkl', "wb") as f:
        pkl.dump(model, f)

def TEST_check_for_eut() -> None:
    """If PT is coded properly, when a=b=g=l=1.0, it should collapse
    to the predictions of EUT. This function, when run, simply asserts that this is true."""

    # Initialize model
    pt_model = PTModel()

    for subject in range(pt_model.num_subjects):
        pt_model.best_fits[subject] = Parameters(a=1.0, b=1.0, g=1.0, l=1.0)

    pt_mean_error     = pt_model.finalize_and_mean_error()
    pt_std_deviation  = pt_model.std_dev_of_error()

    eut_model = EUTModel()
    eut_mean_error    = eut_model.finalize_and_mean_error()
    eut_std_deviation = eut_model.std_dev_of_error()

    # Prints
    print(f'pt mean_error = {pt_mean_error}')
    print(f'pt std_dev = {pt_std_deviation}')
    print(f'eut mean_error = {eut_mean_error}')
    print(f'eut std_dev = {eut_std_deviation}')

    eut_errors: List[float] = eut_model.mean_error_all_subjects(verbose=False,
                                                           save_predictions=True)

    for subject in trange(pt_model.num_subjects):
        assert list(pt_model.data.loc[subject]['prediction']) == list(eut_model.data.loc[subject]['prediction'])

if __name__ == '__main__':
    ### model name (to save to data dir)
    # version = "v2_exhaustive_iter_full_1029"
    # main(version=version)
    #TEST_check_for_eut()
    model = PTModel()
    start_fit = Parameters(a=1.0, b=1.0, g=1.0, l=1.0)
    #model.greedy_fit(precision=0.01, verbose=True, start_fit=start_fit)
    #model.print_info()
