## Adding package to PATH
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(dirname(abspath(__file__)))))

## Imports
from EconomicDecisionMaking.models.prediction_model import *

## Classes
class EUT_TW_Model(PredictionModel):
    """This model implements the prediciton of Expected Utility Theory"""

    def __init__(self):
        super().__init__()
        self.free_params = [] # Empty, since EUT has no free params

        # Get cutoff prices
        ## A note on eut_tw_cutoff_simple.csv~
        ### each column is a day, and each row is a time window in the following order [50 days,25,15,9,6,5,4,3,2]
        ### Thee cutoff prices are determined using the cutoff prices for EUT. For example, for the 50-day tw, the cutoff price from EUT for 50 days is used for every day
        ### until there are less than 50 days left. Then the cutoff prices for the remaining days match EUT cutoff prices and their corresponding days 
        self.cutoff_prices = pd.read_csv(f'{DATA_DIR}/eut_tw_cutoff_simple.csv')

  
    def predict_one_subject(self, subject: int, fit: Optional[Parameters] = None) -> List[int]:
        """Returns the predicted sale amounts.
            subject: the participant's number in the dataframe.
        """
        subject_data = self.get_data_one_subject(subject)
   
        # Start of code that finds optimal time window and errors

        # 1. Return the prediction that subject will sell all units when the price is above the cutoff price for each time window
        predictions_tw50: List[int] = np.where(subject_data['price'] >= self.cutoff_prices.loc[0], subject_data['stored'], 0)
        predictions_tw25: List[int] = np.where(subject_data['price'] >= self.cutoff_prices.loc[1], subject_data['stored'], 0)
        predictions_tw15: List[int] = np.where(subject_data['price'] >= self.cutoff_prices.loc[2], subject_data['stored'], 0)
        predictions_tw9: List[int] = np.where(subject_data['price'] >= self.cutoff_prices.loc[3], subject_data['stored'], 0)
        predictions_tw6: List[int] = np.where(subject_data['price'] >= self.cutoff_prices.loc[4], subject_data['stored'], 0)
        predictions_tw5: List[int] = np.where(subject_data['price'] >= self.cutoff_prices.loc[5], subject_data['stored'], 0)
        predictions_tw4: List[int] = np.where(subject_data['price'] >= self.cutoff_prices.loc[6], subject_data['stored'], 0)
        predictions_tw3: List[int] = np.where(subject_data['price'] >= self.cutoff_prices.loc[7], subject_data['stored'], 0)
        predictions_tw2: List[int] = np.where(subject_data['price'] >= self.cutoff_prices.loc[8], subject_data['stored'], 0)

        # 2. Puts predictions in a dictionary with key equalling time window and value containing the list of predictions
        predictions_per_participant = {} 
        predictions_per_participant['tw50'] = predictions_tw50
        predictions_per_participant['tw25'] = predictions_tw25
        predictions_per_participant['tw15'] = predictions_tw15
        predictions_per_participant['tw9'] = predictions_tw9
        predictions_per_participant['tw6'] = predictions_tw6
        predictions_per_participant['tw5'] = predictions_tw5
        predictions_per_participant['tw4'] = predictions_tw4
        predictions_per_participant['tw3'] = predictions_tw3
        predictions_per_participant['tw2'] = predictions_tw2
        
        # 3. Calculates and puts errors in a dictionary with key equalling time window and value containing the list of predictions
        error_per_participant = {} 
        error_per_participant['tw50'] = self.mean_error_one_subject_proportion(subject, predictions_tw50)
        error_per_participant['tw25'] = self.mean_error_one_subject_proportion(subject, predictions_tw25)
        error_per_participant['tw15'] = self.mean_error_one_subject_proportion(subject, predictions_tw15)
        error_per_participant['tw9'] = self.mean_error_one_subject_proportion(subject, predictions_tw9)
        error_per_participant['tw6'] = self.mean_error_one_subject_proportion(subject, predictions_tw6)
        error_per_participant['tw5'] = self.mean_error_one_subject_proportion(subject, predictions_tw5)
        error_per_participant['tw4'] = self.mean_error_one_subject_proportion(subject, predictions_tw4)
        error_per_participant['tw3'] = self.mean_error_one_subject_proportion(subject, predictions_tw3)
        error_per_participant['tw2'] = self.mean_error_one_subject_proportion(subject, predictions_tw2)

        # 4. Gets minimum error which is used in next step
        min_error = min(error_per_participant.values()) 
 
        # 5. Gets optimal time window for each participant, and prints out Subject number, optimal TW and error value
        ## Printed error results have to be copied into a spreadsheet, as the csv generated later does not include this
        ideal_key = [k for k, v in error_per_participant.items() if v == min_error] 
        print(subject,":",ideal_key, min_error)

        # 6. Puts predictions based on optimal time windows in dictionary
        predictions_all = {}
        for name, value in predictions_per_participant.items():
            if name in ideal_key:
                
                predictions_all[name] = value

        # 7. Because some subjects have two optimal time windows, this selects the first set of predictions, which will be outputted into a csv later
        ## IMPORTANT NOTES ON THIS - This code is not ideal in that the csv should output all predictions for when there are two time windows.
        ### For subjects with two opitmal time windows, the predictions are not exactly the same for each day despite producing the same error value.
        
        predictions = list(predictions_all.values())[0]

        ### The following code below prints out the predictions for particiapnts with multiple optimal time windows. 
        ### The printed results were copied and compiled in the generated csv along with the other results. 
        if subject == 5: # for instance, Subject 5 best fitting TW's are 6 and 9 so this prints out predictions for 6 and 9
            print(subject,error_per_participant['tw6'],  predictions_per_participant['tw6'])
            print(subject,error_per_participant['tw9'],  predictions_per_participant['tw9'])
        elif subject == 14:
            print(subject,error_per_participant['tw6'],  predictions_per_participant['tw6'])
            print(subject,error_per_participant['tw9'],  predictions_per_participant['tw9'])
        elif subject == 16:
            print(subject,error_per_participant['tw6'],  predictions_per_participant['tw6'])
            print(subject,error_per_participant['tw9'],  predictions_per_participant['tw9'])
        elif subject == 17:
            print(subject,error_per_participant['tw6'],  predictions_per_participant['tw6'])
            print(subject,error_per_participant['tw9'],  predictions_per_participant['tw9'])
        elif subject == 52:
            print(subject,error_per_participant['tw25'],  predictions_per_participant['tw25'])
            print(subject,error_per_participant['tw9'],  predictions_per_participant['tw9'])
    
        
        # Returns 'predictions' so it can be outputted into a csv later. This excludes error and also predictions where there is a second time window as mentioned above.
        return predictions



if __name__ == '__main__':

    ### Initialize model
   
    model = EUT_TW_Model()
    predictions_csv = {}
    
    #df.to_csv('eut_tw.csv')
   
    # Finalizes predictions
    # Note: error_type = 'absolute' means that the model will use absolute differences
    #       between prediction and sale amounts to determine error. error_type = 'proportional'
    #       would use the difference in proportions of goods sold instead. The second seems to
    #       be what Glass used in the report, but the numbers in Table 3 seem to suggest
    #       the usage of absolute difference.

    mean_error = model.finalize_and_mean_error(error_type="proportional")

    print(f'mean_error = {mean_error}')
    #print(model.data)
    model.data.to_csv('eut_tw.csv')
