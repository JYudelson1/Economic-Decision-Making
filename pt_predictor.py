from ev_based_model import *
from math import exp
import pickle as pkl

pd.options.mode.chained_assignment = None  # default='warn'

@lru_cache(maxsize=10000)
def prelec(p: float, g: float) -> float:
    """The subjective probability function.
    Inputs:
        p: the objective probability
        g: gamma, the overweighting constant. g >= 1"""
    # Note: seperated into two lines to avoid numpy warnings
    neg_log: float = -np.log(p)
    return exp(-(neg_log ** g))

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
            g: overwighting of low probabilities, g >= 1."""
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
    def get_term_3a(self, day: int, f: int, cutoffs: Tuple[int], g: float):
        """Gets a particular term from equation (6).
        Moved to a seperate function to support caching."""
        term_3a = 1.0
        for h in range(day - 1, f, -1):
            sum_3a = 0.0
            for k in range(1, cutoffs[h]):
                sum_3a += prelec(p(k), g)
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
        for f in range(day - 2, -1, -1):
            # Term 3a
            term_3a = self.get_term_3a(day, f, cutoffs, fit.g)

            # Term 3b
            term_3b = 0.0
            for j in range(cutoffs[f], price):
                term_3b += self.gain(amount, price, j, fit.a, fit.g)

            # Term 3c
            term_3c = 0.0
            lower_bound_3c = max(price + 1, cutoffs[f])
            for j in range(lower_bound, 16):
                term_3c += self.loss(amount, price, j, fit.b, fit.l, fit.g)

            ev += term_3a * (term_3b - term_3c)

        return ev

    @lru_cache(maxsize=CACHE_SIZE)
    def expected_value_day_2(self, price: int, amount: int, fit: Parameters, cutoffs: Tuple[int]) -> float:
        """A straightforward implementation of equation (4).
        See ev_based_model.py for more notes."""
        ev: float = 0

        # Subjective gains
        for j in range(1, price):
            subjective_prob = prelec(p(j), fit.g)
            subjective_gain = (amount * (price - j)) ** fit.a
            ev += subjective_prob * subjective_gain

        #Subjective losses
        for j in range(price + 1, 16):
            subjective_prob = prelec(p(j), fit.g)
            subjective_gain = (amount * (j - price)) ** fit.b
            ev -= subjective_prob * subjective_gain * fit.l

        return ev

if __name__ == '__main__':

    # Error type can be "absolute" or "proportional"
    error_type = "absolute"

    # Initialize model
    model = PTModel()

    # Run stupid fitting
    start_fit = Parameters(a=1.0, b=1.0, g=1.0, l=1.0)
    model.bfs_fit(verbose=True, precision=0.1, error_type=error_type, start_fit=start_fit)

    # Finalizes predictions
    # Note: error_type = 'absolute' means that the model will use absolute differences
    #       between prediction and sale amounts to determine error. error_type = 'proportional'
    #       would use the difference in proportions of goods sold instead. The second seems to
    #       be what Glass used in the report, but the numbers in Table 3 seem to suggest
    #       the usage of absolute difference.

    mean_error = model.finalize_and_mean_error(error_type=error_type)
    std_deviation = model.std_dev_of_error(error_type=error_type)

    # Prints
    print(f'mean_error = {mean_error}')
    print(f'std_dev = {std_deviation}')
    print(model.data)

    # Saves best cutoff data
    model.save_cutoffs(f'{DATA_DIR}/prices_cutoff_pt.csv')
    with open("pt_all_data.pkl", "wb") as f:
        pkl.dump(model.data, f)
