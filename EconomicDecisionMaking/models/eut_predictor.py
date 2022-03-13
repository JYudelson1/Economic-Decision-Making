## Adding package to PATH
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(dirname(abspath(__file__)))))

## Imports
from EconomicDecisionMaking.models.prediction_model import *

## Classes
class EUTModel(PredictionModel):
    """This model implements the prediciton of Expected Utility Theory"""

    def __init__(self):
        super().__init__()
        self.free_params = [] # Empty, since EUT has no free params

        # Get cutoff prices
        self.cutoff_prices = pd.read_csv(f'{DATA_DIR}/eut_cutoff_simple.csv')

    def predict_one_subject(self, subject: int, fit: Optional[Parameters] = None) -> List[int]:
        """Returns the predicted sale amounts.
            subject: the participant's number in the dataframe.
        """
        subject_data = self.get_data_one_subject(subject)

        # Return the prediction that the subject will sell all units when the price is above the cutoff price
        predictions: List[int] = np.where(subject_data['price'] >= self.cutoff_prices.loc[0], subject_data['stored'], 0)
        
        # Generates error for each participant 
        error_per_participant = {} 
        error_per_participant[subject] = self.mean_error_one_subject_proportion(subject, predictions)
        print(error_per_participant)

        return predictions

if __name__ == '__main__':

    ### Initialize model
    model = EUTModel()

    
    ### Finalizes predictions

    # This generates error for each participant (The output was copied and pasted in an excel)
    mean_error = model.finalize_and_mean_error(error_type="proportional")

    # This generates error overall beteen all participants
    mean_error = model.finalize_and_mean_error()
    print(f'mean_error = {mean_error}')

    # This generates stored, sold, price, and prediction by participant and day. 2nd line outputs to csv
    print(model.data)
    #model.data.to_csv('eut.csv')