from prediction_model import *
import numpy as np

class EUTModel(PredictionModel):
    """This model implements the prediciton of Expected Utility Theory"""

    def __init__(self):
        super().__init__()
        self.free_params = [] # Empty, since EUT has no free params

        # Get cutoff prices
        self.cutoff_prices = pd.read_csv(f'{DATA_DIR}/prices_cutoff_eut.csv')

    def predict_one_subject(self, subject: int, fit: Optional[Parameters] = None) -> List[int]:
        """Returns the predicted sale amounts."""
        subject_data = self.get_data_one_subject(subject)

        # Return the prediction that the subject will sell all units when the price is above the cutoff price
        predictions: List[int] = np.where(subject_data['price'] >= self.cutoff_prices.loc[subject], subject_data['stored'], 0)

        return predictions

if __name__ == '__main__':

    model = EUTModel()
    mean_error = model.finalize_and_mean_error()
    print(f'mean_error = {mean_error}')
    print(model.data)
