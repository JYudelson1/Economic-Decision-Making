from prediction_model import *

class EUTModel(PredictionModel):
    """This model implements the prediciton of Expected Utility Theory"""

    def __init__(self):
        super().__init__()
        self.free_params = [] # Empty, since EUT has no free params

        # Get cutoff prices
        self.cutoff_prices = pd.read_csv(f'{DATA_DIR}/prices_cutoff_eut.csv')

    def predict_one_subject(self, subject: int, fit: Optional[Parameters] = None) -> List[int]:
        """Returns the predicted sale amounts.
            subject: the participant's number in the dataframe.
        """
        subject_data = self.get_data_one_subject(subject)

        # Return the prediction that the subject will sell all units when the price is above the cutoff price
        predictions: List[int] = np.where(subject_data['price'] >= self.cutoff_prices.loc[subject], subject_data['stored'], 0)

        return predictions

if __name__ == '__main__':

    # Initilize model
    model = EUTModel()

    # Finalizes predictions
    # Note: error_type = 'absolute' means that the model will use absolute differences
    #       between prediction and sale amounts to determine error. error_type = 'proportional'
    #       would use the difference in proportions of goods sold instead. The second seems to
    #       be what Glass used in the report, but the numbers in Table 3 seem to suggest
    #       the usage of absolute difference.
    mean_error = model.finalize_and_mean_error(error_type="absolute")
    print(f'mean_error = {mean_error}')
    print(model.data)
