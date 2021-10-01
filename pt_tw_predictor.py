from ev_based_model import *
import pickle as pkl

pd.options.mode.chained_assignment = None  # default='warn'



class PT_TWModel(EVModel):
    """This model implements the predictions of Prospect Theory, with Arnold Glass's Time Window"""

    def __init__(self):
        super().__init__()
        self.free_params: List[str] = ["a", "b", "g", "l", "tw"]

    @lru_cache(maxsize=CACHE_SIZE)
    def p_sales(self, j: int, tw: int) -> float:
        """A straightforward implementation of Arnold Glass's p_sales definition,
        between equations (12) and (13)."""

        k_sum = 0.0
        for k in range(0, tw - 1):
            h_sum = 0.0
            for h in range(1, j):
                h_sum += p(h)
            k_sum += h_sum ** k

        return k_sum

    @lru_cache(maxsize=CACHE_SIZE)
    def expected_value(self, day: int, price: int, amount: int, fit: Parameters, cutoffs: Tuple[int]) -> float:
        """A straightforward implementation of equation (14).
        See ev_based_model.py for more notes."""
        ev = 0.0
        for j in range(1, price):
            inner_bracket = max((amount * (cutoffs[day - 1] - j)), 0) ** fit.a
            prob = self.p_sales(j, fit.tw) * p(j)
            prob = prelec(prob, fit.g)
            ev += prob * inner_bracket

        for j in range(price + 1, 16):
            inner_bracket = (amount * (j - cutoffs[day - 1])) ** fit.b
            prob = self.p_sales(j, fit.tw) * p(j)
            prob = prelec(prob, fit.g)
            ev -= prob * inner_bracket * fit.l

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

def main() -> None:
    # model name (to save to data dir)
    version = "exhaustive_0-5_930_prop"

    # Error type can be "absolute" or "proportional"
    error_type = "proportional"

    # Initialize model
    model = PT_TWModel()

    # Run fitting
    #start_fit = Parameters(a=1.0, b=1.0, g=1.0, l=1.0, tw=67)
    #model.minimize_fit(start_fit=start_fit, verbose=True, error_type=error_type, method="Nelder-Mead")
    #model.bfs_fit(verbose=True, precision=0.05, error_type=error_type, start_fit=start_fit)
    model.exhaustive_fit(precision=0.5, verbose=True, error_type=error_type)
    #model.greedy_fit(verbose=True, precision=0.05, error_type=error_type, start_fit=start_fit)
    #model.simulated_annealing_fit(start_fit=start_fit, verbose=True, error_type=error_type)

    mean_error = model.finalize_and_mean_error(error_type=error_type)
    std_deviation = model.std_dev_of_error(error_type=error_type)

    # Prints
    print(f'mean_error = {mean_error}')
    print(f'std_dev = {std_deviation}')
    print(model.data)

    # Saves data
    with open(f'{DATA_DIR}/pt_tw_{version}.pkl', "wb") as f:
        pkl.dump(model, f)

if __name__ == '__main__':

    main()
    # with open("data/pt_tw_greedy_0-1_924_abs.pkl", "rb") as f:
    #     m = pkl.load(f)
