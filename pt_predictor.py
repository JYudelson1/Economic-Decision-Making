from prediction_model import *
from math import exp
from tqdm import tqdm, trange
from functools import lru_cache

def prelec(p: float, g: float) -> float:
    return exp(-(-np.log(p) ** g))

class PTModel(PredictionModel):
    """This model implements the prediciton of Prospect Theory"""

    def __init__(self):
        super().__init__()
        self.free_params: List[str] = ["a", "b", "g", "l"]

        # Get cutoff prices
        self.cutoff_prices: pd.DataFrame = pd.read_csv(f'{DATA_DIR}/prices_cutoff_eut.csv')

        # Get probabilities of each price
        self.prices_probabilities: pd.DataFrame = pd.read_csv(f'{DATA_DIR}/prices_probabilities.csv')

    def cutoff(self, day: int):
        if day >= 68:
            return 14
        return self.cutoff_prices.loc[0, str(day)]

    @lru_cache(maxsize=1000)
    def p(self, price: float) -> float:
        return self.prices_probabilities.loc[price - 1]["probability"]

    @lru_cache(maxsize=1000)
    def p_sale(self, day: int, price: int) -> float:
        """Implementation of p_sale,j,d from page 12"""
        # Get probability of this price occurring
        p_j: float = self.p(price)
        complicated_sum: float = 0

        for f in range(day + 2, 68):
            inner_product: float = 1
            for h in range(day + 1, f - 1):
                inner_sum: float = 0
                for k in range(1, self.cutoff(h + 1)):
                    inner_sum += self.p(k)
                inner_product *= inner_sum
            complicated_sum += inner_product

        return p_j + p_j * complicated_sum

    @lru_cache(maxsize=1000)
    def expected_value(self, day: int, price: int, amount: int, fit: Parameters) -> float:
        """A straightforward implementation of equation (8)"""
        ev: float = 0

        for j in range(1, price):
            sale_prob: float = self.p_sale(day, j)
            subjective_sale_prob: float = prelec(sale_prob, fit.g)

            inner_bracket: float = amount * (self.cutoff(day) - j)
            ev += subjective_sale_prob * (inner_bracket ** fit.a)

        for j in range(price + 1, 15 + 1):
            sale_prob2: float = self.p_sale(day, j)
            subjective_sale_prob2: float = prelec(sale_prob2, fit.g)

            inner_bracket2: float = amount * (j - self.cutoff(day))
            ev -= subjective_sale_prob2 * fit.l * (inner_bracket2 ** fit.b)
        return ev

    @lru_cache(maxsize=1000)
    def expected_value_2(self, day: int, price: int, amount: int, fit: Parameters) -> float:
        """A straightforward implementation of equation (6)"""
        ev: float = 0

        # First term
        for j in range(self.cutoff(day + 1), price):
            inner_bracket1 = amount * (price - j)
            inner_bracket1 = inner_bracket1 ** fit.a
            ev += inner_bracket1 * prelec(self.p(j), fit.g)

        # Second term
        lower_bound = max(price + 1, self.cutoff(day + 1))
        for j in range(lower_bound, 16):
            inner_bracket2 = amount * (j - price)
            inner_bracket2 = inner_bracket2 ** fit.b
            ev += inner_bracket2 * prelec(self.p(j), fit.g) * fit.l

        # Third Term
        for f in range(day+2, 68):
            # Term 3a
            term_3a = 1
            for h in range(day+1, f-1):
                sum_3a = 0
                for k in range(1, self.cutoff(h)):
                    sum_3a += prelec(self.p(k), fit.g)
                term_3a *= sum_3a
            # Term 3b
            term_3b = 0
            for j in range(self.cutoff(f), price):
                inner_bracket3b = amount * (price - j)
                inner_bracket3b = inner_bracket3b ** fit.a
                term_3b += inner_bracket3b * prelec(self.p(j), fit.g)
            # Term 3c
            term_3c = 0
            lower_bound_3c = max(price + 1, self.cutoff(f))
            for j in range(lower_bound, 16):
                inner_bracket3c = amount * (j - price)
                inner_bracket3c = inner_bracket3c ** fit.b
                term_3c += inner_bracket3c * prelec(self.p(j), fit.g) * fit.l

            ev += term_3a * (term_3b - term_3c)

        return ev

    def expected_value_eut(self, day: int, price: int, amount: int, fit: Optional[Parameters] = None) -> float:
        subject_data: pd.DataFrame = self.get_data_one_subject(0)
        stored: int = int(subject_data.loc[str(day)]["stored"])
        ev = amount * price
        for j in range(1, 16):
            ev += (stored - amount) * self.p_sale(day, j) * j

        return ev


    def predict_one_subject(self, subject: int, fit: Optional[Parameters] = None) -> List[int]:
        """Returns the predicted sale amounts.
            subject: the participant's number in the dataframe.
        """
        # Ensure fit is not None
        if not fit:
            raise ValueError("Prospect Theory needs paramaters!")

        # Get subject data
        subject_data: pd.DataFrame = self.get_data_one_subject(subject)

        # Create predictions list
        predictions: List[int] = []

        # Iterate through each day:
        for day in trange(68, leave=False, desc="Predicting..."):
            # Get stored amount, since sell amount must be less than stored

            stored: int = int(subject_data.loc[str(day)]["stored"])
            price: int = int(subject_data.loc[str(day)]["price"])
            max_expected_value: float = 0
            best_sell_amount: int = 0

            # Find the sell_amount that maximizes expected value
            for sell_amount in range(stored + 1):
                #expected_value = self.expected_value(day, price, sell_amount, fit)
                expected_value = self.expected_value_2(day, price, sell_amount, fit)
                # Save the best value
                if expected_value > max_expected_value:
                    max_expected_value = expected_value
                    best_sell_amount = sell_amount
            predictions.append(best_sell_amount)

        #print(predictions)
        return predictions

if __name__ == '__main__':

    # Initilize model
    model = PTModel()

    # Run stupid fitting
    model.stupid_fit(verbose=True, precision=1)

    # EUT_params = Parameters(a=1, b=1, l=1, g=1)
    # for subject in range(model.num_subjects):
    #     model.best_fits[subject] = EUT_params

    # Finalizes predictions
    # Note: error_type = 'absolute' means that the model will use absolute differences
    #       between prediction and sale amounts to determine error. error_type = 'proportional'
    #       would use the difference in proportions of goods sold instead. The second seems to
    #       be what Glass used in the report, but the numbers in Table 3 seem to suggest
    #       the usage of absolute difference.
    mean_error = model.finalize_and_mean_error(error_type="absolute")

    # Prints
    print(f'mean_error = {mean_error}')
    print(model.data)
