## Adding package to PATH
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(dirname(abspath(__file__)))))

## Imports
import EconomicDecisionMaking.models.pt_predictor as pt_predictor
import EconomicDecisionMaking.save_model_as_spreadsheet as save_model_as_spreadsheet
from EconomicDecisionMaking.utils import DATA_DIR

if __name__ == '__main__':
        version = "exhaustive_iter_test"

        pt_predictor.main(version=version)

        save_model_as_spreadsheet.spreadsheet_main(version=f'{DATA_DIR}/pt_{version}.pkl',
                                       filename=f'{DATA_DIR}/pt_predictions_{version}.xlsx',
                                       use_all_fits=False)
