from ev_based_model import *
from eut_predictor import *

pd.options.mode.chained_assignment = None  # default='warn'

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

def main(version: str) -> None:

    # Error type can be "absolute" or "proportional"
    error_type = "proportional"

    # Initialize model
    model = PTModel()

    # Run fitting
    #start_fit = Parameters(a=1.0, b=1.0, g=1.0, l=1.0)
    #model.minimize_fit(start_fit=start_fit, verbose=True, error_type=error_type, method="Nelder-Mead")
    #model.bfs_fit(verbose=True, precision=0.05, error_type=error_type, start_fit=start_fit)
    #model.greedy_fit(verbose=True, precision=0.1, error_type=error_type, start_fit=start_fit)
    #model.simulated_annealing_fit(start_fit=start_fit, verbose=True, error_type=error_type)
    #model.exhaustive_fit(precision=0.2, verbose=True, error_type=error_type)

    precisions = (0.05, 0.01, 0.001)
    model.iterative_exhaustive_search(precisions, verbose=True, start=True)

    # Print
    model.print_info()

    # Saves data
    with open(f'{DATA_DIR}/pt_{version}.pkl', "wb") as f:
        pkl.dump(model, f)

def TEST_check_for_eut() -> None:
    """If PT is coded properly, when a=b=g=l=1.0, it should collapse
    to the predictions of EUT. This function, when run, simply asserts that this is true."""
    # Error type can be "absolute" or "proportional"
    error_type = "proportional"

    # Initialize model
    pt_model = PTModel()

    for subject in range(pt_model.num_subjects):
        pt_model.best_fits[subject] = Parameters(a=1.0, b=1.0, g=1.0, l=1.0)

    pt_mean_error = pt_model.finalize_and_mean_error(error_type=error_type)
    pt_std_deviation = pt_model.std_dev_of_error(error_type=error_type)

    eut_model = EUTModel()
    eut_mean_error = eut_model.finalize_and_mean_error(error_type=error_type)
    eut_std_deviation = eut_model.std_dev_of_error(error_type=error_type)

    # Prints
    print(f'pt mean_error = {pt_mean_error}')
    print(f'pt std_dev = {pt_std_deviation}')
    print(f'eut mean_error = {eut_mean_error}')
    print(f'eut std_dev = {eut_std_deviation}')

    eut_errors: List[float] = eut_model.mean_error_all_subjects(error_type,
                                                           False,
                                                           save_predictions=True)

    for subject in trange(pt_model.num_subjects):
        assert list(pt_model.data.loc[subject]['prediction']) == list(eut_model.data.loc[subject]['prediction'])

def TEST_check_last_groups_values() -> None:
    """Explanation: The last group to work on this project came up with very different values.
    One explanation is that they seemed to have checked error for each participant by checking
    the absolute value of the sum of the differences between predicted sale amounts and actual sale amounts.

    The correct method, however, is to use sum(abs(diff)), not abs(sum(diff)).
    This function uses abs(sum(diff)), and checks to make sure that this yields the errors the last group found."""

    def mean_error_one_subject_abs_sum(model: PTModel, subject: int, predictions: List[int]) -> float:
        d_0: int = 0 # Number of days for which the participant had no goods stored
        total_error: float = 0
        subject_data: pd.DataFrame = model.get_data_one_subject(subject)

        # Iterate through the days and sum the error
        for day in range(model.num_days):
            if subject_data['stored'][day] == 0:
                d_0 += 1
                continue
            day_error: float = (predictions[day] - subject_data['sold'][day]) / subject_data['stored'][day]
            total_error += day_error

        # Find the mean
        mean_error: float = total_error / (model.num_days - d_0)

        return abs(mean_error)

    model = PTModel()

    # Initialize model with the parameter values found by the previous group
    fits = {
        0: Parameters(a=.806,b=.84,g=1,l=1.4),
        1: Parameters(a=.894,b=1,g=1,l=1)
    }

    # Iterate through each subject and check errors
    for subject in range(model.num_subjects):
        pred = model.predict_one_subject(subject, fits[subject])
        error = mean_error_one_subject_abs_sum(model, subject, pred)
        print(f'For subject {subject}, the abs(sum) error is {error}')

if __name__ == '__main__':
    # model name (to save to data dir)
    # version = "v2_exhaustive_iter_full_1029"
    # main(version=version)
    TEST_check_last_groups_values()
