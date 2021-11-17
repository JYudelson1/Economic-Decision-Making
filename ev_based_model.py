from prediction_model import *

class EVModel(PredictionModel):
    """Any model other than EUT can be simplified down to an expected value function.
    The only difference between e.g. PT and HPT is the expected value function
        (As far as I can tell. If this later turns out to be false we can rebase.)
    NOTE: Time windows are implicitly supported, as the only effect is a change in
        in the generate_cutoffs function. This occurs if and only if
        the model has tw in its free_params"""

    def __init__(self):
        super().__init__()

    @lru_cache(maxsize=CACHE_SIZE)
    def generate_cutoffs(self, fit: Parameters) -> List[int]:
        """Generates the i' cutoffs for a given fit.
        # It turns out that cutoff prices depend ONLY on the fit.
        # That is to say, the same fit with the same model should always generate the same cutoff prices.
        Inputs:
            fit: the Parameters used to calculate the cutoff values."""
        cutoffs: List[int] = [1]
        iter = range(1, self.num_days)
        for day in tqdm(iter, desc=f'Cutoffs', leave=False, disable=True):

            price: int = cutoffs[day - 1] # price has to be at least the last cutoff value

            # If the model has a time window, every day after that point
            # is the same. So If fit.tw = 20, every day from 20 to 67 has
            # same cutoff as day 19.
            if fit.tw and day >= fit.tw:
                cutoffs.append(price)
                continue

            # Seperate EV function for day 2 (with index 1)
            if day == 1:
                ev = self.expected_value_day_2(price, 1, fit, tuple(cutoffs))
            else:
                ev = self.expected_value(day, price, 1, fit, tuple(cutoffs))

            while (ev <= 0 and price < 15):
                # Continually increment price until it is worthwhile to sell
                price += 1
                ev = self.expected_value(day, price, 1, fit, tuple(cutoffs))
            cutoffs.append(price)
        return cutoffs

    def save_cutoffs(self, filename: str):
        """Save all i' cutoffs into a .csv file
        Inputs:
            filename: the name of the save file"""
        df = pd.DataFrame(columns=range(self.num_subjects))
        for subject in range(self.num_subjects):
            best_fit = self.best_fits[subject]
            cutoffs = self.generate_cutoffs(best_fit)
            for d in range(self.num_days):
                df.loc[subject, d] = cutoffs[d]
        df.to_csv(filename, index=False)

    @lru_cache(maxsize=CACHE_SIZE)
    def expected_value(self, day: int, price: int, amount: int, fit: Parameters, cutoffs: Tuple[int]) -> float:
        """This should return the expected value based on all these factors.
        NOTE: this should be implemented in each individual model.
        Inputs:
            day: the current day
            price: the price of goods on the current day (in normalized 1-15 form)
            amount: the amount of goods to be hypothetically sold
            fit: the paramaters under which the subject is hypothesized to be acting
            cutoffs: a list of cutoff values for the given subject.
                NOTE: The reason you can't just derive this from the subject is because
                      cutoff values are derived from expected values, and vice versa,
                      so you have to be able to calculate expected values based on
                      an incomplete set of cutoff data. Refer to the README for more."""
        raise NotImplementedError

    @lru_cache(maxsize=CACHE_SIZE)
    def expected_value_day_2(self, price: int, amount: int, fit: Parameters, cutoffs: Tuple[int]) -> float:
        """This returns the expected value for day 2.
        The reason why this is a seperate function is because it is sometimes a different eqn.
            This makes sense, as it is the 'base case' for determining e.g. cutoff values.
        By default, this just checks self.expected_value.
            NOTE: This should be reimplemented if you desire unique day 2 behavior.
        Inputs:
            day: the current day
            price: the price of goods on the current day (in normalized 1-15 form)
            amount: the amount of goods to be hypothetically sold
            fit: the paramaters under which the subject is hypothesized to be acting
            cutoffs: a list of cutoff values for the given subject.
                NOTE: The reason you can't just derive this from the subject is because
                      cutoff values are derived from expected values, and vice versa,
                      so you have to be able to calculate expected values based on
                      an incomplete set of cutoff data. Refer to the README for more."""

        return self.expected_value(1, price, amount, fit, cutoffs)

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

        # Generate cutoff values for this fit
        cutoffs = self.generate_cutoffs(fit)

        # Turn cutoffs into a tuple
        # This is to make it hashable and cacheable, and therefore save compute time
        cutoffs_tuple: Tuple[int, ...] = tuple(cutoffs)

        # Iterate through each day, backwards:
        for day in trange(self.num_days, leave=False, desc="Predicting...", disable=True):
            # Get stored amount, since sell amount must be less than stored

            stored: int = int(subject_data.loc[str(day)]["stored"])
            price: int = int(subject_data.loc[str(day)]["price"])
            max_expected_value: float = 0
            best_sell_amount: int = 0

            # If the price < cutoff price, no chance of selling:
            cutoff_price = cutoffs[day]
            if price < cutoff_price:
                predictions.append(0)
                continue

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
