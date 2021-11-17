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
        self.cutoff_prices = pd.read_csv(f'{DATA_DIR}/eut_tw_cutoff_simple.csv')

    def predict_one_subject(self, subject: int, fit: Optional[Parameters] = None) -> List[int]:
        """Returns the predicted sale amounts.
            subject: the participant's number in the dataframe.
        """
        subject_data = self.get_data_one_subject(subject)
        #print("self", self.cutoff_prices)
        # Return the prediction that the subject will sell all units when the price is above the cutoff price
        predictions_tw50: List[int] = np.where(subject_data['price'] >= self.cutoff_prices.loc[0], subject_data['stored'], 0)
        predictions_tw25: List[int] = np.where(subject_data['price'] >= self.cutoff_prices.loc[1], subject_data['stored'], 0)
        predictions_tw15: List[int] = np.where(subject_data['price'] >= self.cutoff_prices.loc[2], subject_data['stored'], 0)
        predictions_tw9:  List[int] = np.where(subject_data['price'] >= self.cutoff_prices.loc[3], subject_data['stored'], 0)
        predictions_tw6:  List[int] = np.where(subject_data['price'] >= self.cutoff_prices.loc[4], subject_data['stored'], 0)
        predictions_tw5:  List[int] = np.where(subject_data['price'] >= self.cutoff_prices.loc[5], subject_data['stored'], 0)
        predictions_tw4:  List[int] = np.where(subject_data['price'] >= self.cutoff_prices.loc[6], subject_data['stored'], 0)
        predictions_tw3:  List[int] = np.where(subject_data['price'] >= self.cutoff_prices.loc[7], subject_data['stored'], 0)
        predictions_tw2:  List[int] = np.where(subject_data['price'] >= self.cutoff_prices.loc[8], subject_data['stored'], 0)

        predictions_per_participant = {} #Create dictionary
        predictions_per_participant['tw50'] = predictions_tw50
        predictions_per_participant['tw25'] = predictions_tw25
        predictions_per_participant['tw15'] = predictions_tw15
        predictions_per_participant['tw9']  = predictions_tw9
        predictions_per_participant['tw6']  = predictions_tw6
        predictions_per_participant['tw5']  = predictions_tw5
        predictions_per_participant['tw4']  = predictions_tw4
        predictions_per_participant['tw3']  = predictions_tw3
        predictions_per_participant['tw2']  = predictions_tw2

        error_per_participant = {} #Create dictionary
        error_per_participant['tw50'] = self.mean_error_one_subject_proportion(subject, predictions_tw50)
        error_per_participant['tw25'] = self.mean_error_one_subject_proportion(subject, predictions_tw25)
        error_per_participant['tw15'] = self.mean_error_one_subject_proportion(subject, predictions_tw15)
        error_per_participant['tw9']  = self.mean_error_one_subject_proportion(subject, predictions_tw9)
        error_per_participant['tw6']  = self.mean_error_one_subject_proportion(subject, predictions_tw6)
        error_per_participant['tw5']  = self.mean_error_one_subject_proportion(subject, predictions_tw5)
        error_per_participant['tw4']  = self.mean_error_one_subject_proportion(subject, predictions_tw4)
        error_per_participant['tw3']  = self.mean_error_one_subject_proportion(subject, predictions_tw3)
        error_per_participant['tw2']  = self.mean_error_one_subject_proportion(subject, predictions_tw2)

        min_error = min(error_per_participant.values())
        print(min_error)
        ideal_key = [k for k, v in error_per_participant.items() if v == min_error]

        predictions_all = {}
        for name, value in predictions_per_participant.items():
            if name in ideal_key:
                predictions_all[name] = value

        # This selects the first set of predictions (specifically for those where there are more than one TW that fits), since the predictions should be the same
        predictions = list(predictions_all.values())[0]


        return predictions


if __name__ == '__main__':

    ### Initialize model
    model = EUTModel()

    ### Finalizes predictions
    mean_error = model.finalize_and_mean_error()
    print(f'mean_error = {mean_error}')
    print(model.data)
