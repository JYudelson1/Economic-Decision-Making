from ev_based_model import *
import pickle as pkl

pd.options.mode.chained_assignment = None  # default='warn'



class PT_TWModel(EVModel):
    """This model implements the predictions of Prospect Theory, with Arnold Glass's Time Window"""

    def __init__(self):
        super().__init__()
        self.free_params: List[str] = ["a", "b", "g", "l", "tw"]

    def generate_cutoffs(self, fit: Parameters) -> List[int]:
        return []

    @lru_cache(maxsize=CACHE_SIZE)
    def p_sales(self, j: int, day: int, tw: int) -> float:
        """A straightforward implementation of Arnold Glass's p_sales definition,
        between equations (12) and (13)."""

        k_sum = 0.0
        for k in range(day - 2, -1 + (68 - tw), -1):
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
            inner_bracket = max((amount * (price - j)), 0) ** fit.a
            prob = self.p_sales(j, day, fit.tw) * p(j)
            prob = prelec(prob, fit.g)
            ev += prob * inner_bracket

        for j in range(price + 1, 16):
            inner_bracket = (amount * (j - price)) ** fit.b
            prob = self.p_sales(j, day, fit.tw) * p(j)
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

    @lru_cache(maxsize=CACHE_SIZE)
    def predict_one_subject(self, subject: int, fit: Optional[Parameters] = None) -> List[int]:
        """Returns the predicted sale amounts for each day.
        Inputs:
            subject: the participant's number in the dataframe.
            fit: the parameters to be used to predict the subject's behavior.
                NOTE: The free_params used in the fit will depend on the exact model.
        """
        # Ensure fit is not None
        if not fit:
            raise ValueError("Expected Value based models need parameters!")

        # Get subject data
        subject_data: pd.DataFrame = self.get_data_one_subject(subject)

        # Create predictions list
        predictions: List[int] = []

        # Turn cutoffs into a tuple
        # This is to make it hashable and cacheable, and therefore save compute time
        cutoffs_tuple: Tuple[int, ...] = tuple([])

        # Iterate through each day, backwards:
        for day in trange(self.num_days, leave=False, desc="Predicting...", disable=True):
            # Get stored amount, since sell amount must be less than stored

            stored: int = int(subject_data.loc[str(day)]["stored"])
            price: int = int(subject_data.loc[str(day)]["price"])
            max_expected_value: float = 0
            best_sell_amount: int = 0

            # Sell everything the last day
            if day == 0:
                predictions.append(stored)
                continue

            # Find the sell_amount that maximizes expected value
            for sell_amount in range(stored + 1):
                expected_value: float = 0
                if day == 1: # second to last day
                    expected_value = self.expected_value_day_2(price, sell_amount, fit, cutoffs_tuple)
                elif day >= 2: # all other days
                    expected_value = self.expected_value(day, price, sell_amount, fit, cutoffs_tuple)
                # Save the best value
                if expected_value > max_expected_value:
                    max_expected_value = expected_value
                    best_sell_amount = sell_amount
            predictions.append(best_sell_amount)

        return predictions

def main() -> None:
    # model name (to save to data dir)
    version = "exhaustive_0-5_101_prop_test"

    # Error type can be "absolute" or "proportional"
    error_type = "proportional"

    # Initialize model
    model = PT_TWModel()

    # Run fitting
    #start_fit = Parameters(a=1.0, b=1.0, g=1.0, l=1.0, tw=67)
    #model.minimize_fit(start_fit=start_fit, verbose=True, error_type=error_type, method="Nelder-Mead")
    #model.bfs_fit(verbose=True, precision=0.05, error_type=error_type, start_fit=start_fit)
    model.exhaustive_fit_one_subject(subject=50, precision=0.5, verbose=True, error_type=error_type)
    #model.greedy_fit(verbose=True, precision=0.05, error_type=error_type, start_fit=start_fit)
    #model.simulated_annealing_fit(start_fit=start_fit, verbose=True, error_type=error_type)

    #mean_error = model.finalize_and_mean_error(error_type=error_type)
    #std_deviation = model.std_dev_of_error(error_type=error_type)

    # Prints
    # print(f'mean_error = {mean_error}')
    # print(f'std_dev = {std_deviation}')
    print(model.data)

    # Saves data
    with open(f'{DATA_DIR}/pt_tw_{version}.pkl', "wb") as f:
        pkl.dump(model, f)

if __name__ == '__main__':

    main()
    # version = "exhaustive_0-5_930_prop"
    # with open(f'{DATA_DIR}/pt_tw_{version}.pkl', "rb") as f:
    #     m = pkl.load(f)
