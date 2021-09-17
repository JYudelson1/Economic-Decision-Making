from prediction_model import *
from math import exp
from tqdm import tqdm, trange
from functools import lru_cache
from typing import Dict

pd.options.mode.chained_assignment = None  # default='warn'

@lru_cache(maxsize=10000)
def prelec(p: float, g: float) -> float:
    return exp(-(-np.log(p) ** g))

class PTModel(PredictionModel):
    """This model implements the prediciton of Prospect Theory"""

    def __init__(self):
        super().__init__()
        self.free_params: List[str] = ["a", "b", "g", "l"]

        # Get probabilities of each price
        self.prices_probabilities: pd.DataFrame = pd.read_csv(f'{DATA_DIR}/prices_probabilities.csv')

        for subject in range(self.num_subjects):
            self.data.loc[subject, "cutoff"] = [1] * self.num_days

    @lru_cache(maxsize=1000)
    def cutoff(self, day: int, subject: int):
        #return self.cutoff_prices.loc[0, str(day)]
        return int(self.data.loc[subject, "cutoff"][day])

    def generate_cutoffs(self, subject: int, fit: Parameters):
        """Generates the i' cutoffs for a given subject"""
        cutoffs: List[int] = [1]
        iter = range(1, 68)
        for i in tqdm(iter, desc=f'Cutoffs_{subject}', leave=False):
            price: int = cutoffs[i - 1] # price has to be at least the last cutoff value
            ev: float = self.expected_value(i, price, 1, fit, subject)
            while (ev <= 0):
                price += 1
                ev = self.expected_value(i, price, 1, fit, subject)
            cutoffs.append(price)
            self.data.loc[subject, "cutoff"][i] = price

    def save_cutoffs(self, filename: str):
        """Save all i' cutoffs"""
        df = pd.DataFrame(columns=range(self.num_subjects))
        for i in range(self.num_subjects):
            for d in range(self.num_days):
                df.loc[i, d] = self.data.loc[i, "cutoff"][d]
        df.to_csv(filename, index=False)

    def load_cutoffs(self, filename: str):
        """Loads i' cutoff values from a csv file"""
        cutoff_prices = pd.read_csv(filename)
        for subject in range(self.num_subjects):
            for day in range(self.num_days):
                self.data.loc[subject, "cutoff"][day] = cutoff_prices.loc[subject][day]

    def finalize_all_cutoffs(self):
        """Generates all i' values for all subjects based on their best fits"""
        for subject in trange(self.num_subjects, desc="Generating Cutoffs", leave=False):
            if subject >= 2:
                return
            best_fit = self.best_fits[subject]
            self.generate_cutoffs(subject, best_fit)

    @lru_cache(maxsize=10000)
    def p(self, price: float) -> float:
        return self.prices_probabilities.loc[price - 1]["probability"]

    @lru_cache(maxsize=10000)
    def gain(self, amount: int, price: int, j: int, a: float, g: float) -> float:
        inner_bracket1 = amount * (price - j)
        inner_bracket1 = inner_bracket1 ** a
        return inner_bracket1 * prelec(self.p(j), g)

    @lru_cache(maxsize=10000)
    def loss(self, amount: int, price: int, j: int, b: float, l: float, g: float) -> float:
        inner_bracket2 = amount * (j - price)
        inner_bracket2 = inner_bracket2 ** b
        return inner_bracket2 * prelec(self.p(j), g) * l

    @lru_cache(maxsize=10000)
    def get_term_3a(self, day, f, subject, g):
        """Gets a particular term from equation (6)"""
        term_3a = 1
        for h in range(day - 1, f, -1):
            sum_3a = 0
            for k in range(1, self.cutoff(h, subject)):
                sum_3a += prelec(self.p(k), g)
            term_3a *= sum_3a
        return term_3a

    @lru_cache(maxsize=10000)
    def expected_value(self, day: int, price: int, amount: int, fit: Parameters, subject: int) -> float:
        """A straightforward implementation of equation (6)"""
        ev: float = 0

        # First term
        for j in range(self.cutoff(day - 1, subject), price):
            ev += self.gain(amount, price, j, fit.a, fit.g)

        # Second term
        lower_bound = max(price + 1, self.cutoff(day - 1, subject))
        for j in range(lower_bound, 16):
            ev -= self.loss(amount, price, j, fit.b, fit.l, fit.g)

        # Third Term
        for f in range(day - 2, -1, -1):
            # Term 3a
            term_3a = self.get_term_3a(day, f, subject, fit.g)

            # Term 3b
            term_3b = 0
            for j in range(self.cutoff(f, subject), price):
                term_3b += self.gain(amount, price, j, fit.a, fit.g)

            # Term 3c
            term_3c = 0
            lower_bound_3c = max(price + 1, self.cutoff(f, subject))
            for j in range(lower_bound, 16):
                term_3c += self.loss(amount, price, j, fit.b, fit.l, fit.g)

            ev += term_3a * (term_3b - term_3c)

        return ev

    @lru_cache(maxsize=10000)
    def expected_value_day_2(self, price: int, amount: int, fit: Parameters) -> float:
        """A straightforward implementation of equation (4)"""
        ev: float = 0

        # Subjective gains
        for j in range(1, price):
            subjective_prob = prelec(self.p(j), fit.g)
            subjective_gain = (n * (price - j)) ** fit.a
            ev += subjective_prob * subjective_gain

        #Subjective losses
        for j in range(price + 1, 16):
            subjective_prob = prelec(self.p(j), fit.g)
            subjective_gain = (n * (j - price)) ** fit.b
            ev -= subjective_prob * subjective_gain * fit.l

        return ev


    def predict_one_subject(self, subject: int, fit: Optional[Parameters] = None) -> List[int]:
        """Returns the predicted sale amounts.
            subject: the participant's number in the dataframe.
            fit: the a, b, g, and l parameters
        """
        # Ensure fit is not None
        if not fit:
            raise ValueError("Prospect Theory needs paramaters!")

        # Get subject data
        subject_data: pd.DataFrame = self.get_data_one_subject(subject)

        # Create predictions list
        predictions: List[int] = []

        # Generate cutoff values for this fit
        self.generate_cutoffs(subject, fit)

        # Iterate through each day, backwards:
        for day in trange(68, leave=False, desc="Predicting..."):
            # Get stored amount, since sell amount must be less than stored

            stored: int = int(subject_data.loc[str(day)]["stored"])
            price: int = int(subject_data.loc[str(day)]["price"])
            max_expected_value: float = 0
            best_sell_amount: int = 0

            # If the price < cutoff price, no chance of selling:
            cutoff_price = self.cutoff(day, subject)
            if price < cutoff_price:
                predictions.append(0)
                continue

            # Sell everything the last day
            if day == 0: # last day
                predictions.append(stored)
                continue

            # Find the sell_amount that maximizes expected value
            for sell_amount in range(stored + 1):
                expected_value: float = 0
                if day == 1: # second to last day
                    expected_value = self.expected_value_day_2(price, sell_amount, fit)
                elif day >= 2: # all other days
                    expected_value = self.expected_value(day, price, sell_amount, fit, subject)
                # Save the best value
                if expected_value > max_expected_value:
                    max_expected_value = expected_value
                    best_sell_amount = sell_amount
            predictions.append(best_sell_amount)

        return predictions

if __name__ == '__main__':

    # Initilize model
    model = PTModel()

    # Run stupid fitting
    model.stupid_fit(verbose=True, precision=0.5, error_type="absolute")

    # Finalizes predictions
    # Note: error_type = 'absolute' means that the model will use absolute differences
    #       between prediction and sale amounts to determine error. error_type = 'proportional'
    #       would use the difference in proportions of goods sold instead. The second seems to
    #       be what Glass used in the report, but the numbers in Table 3 seem to suggest
    #       the usage of absolute difference.
    model.finalize_all_cutoffs()
    mean_error = model.finalize_and_mean_error(error_type="absolute")

    # Prints
    print(f'mean_error = {mean_error}')
    print(model.data)
