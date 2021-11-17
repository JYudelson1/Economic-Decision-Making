## Adding package to PATH
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(dirname(abspath(__file__)))))

## Imports
from EconomicDecisionMaking.models.ev_based_model import *
from EconomicDecisionMaking.models.hpt_predictor import HPTModel

pd.options.mode.chained_assignment = None  # default='warn'

## Classes
class HPTTWModel(HPTModel):
    """This model implements the predictions of Prospect Theory, with Arnold Glass's Time Window"""

    def __init__(self):
        super().__init__()
        self.free_params: list[str] = ["xg", "xl", "g", "l", "tw"]

def main(version: str) -> None:

    ### Initialize model
    model = HPTTWModel()

    ### Run fitting
    model.exhaustive_fit(precision=0.5, verbose=True)

    mean_error = model.finalize_and_mean_error()
    std_deviation = model.std_dev_of_error()

    ### Prints
    print(f'mean_error = {mean_error}')
    print(f'std_dev = {std_deviation}')
    print(model.data)

    ### Saves data
    with open(f'{DATA_DIR}/hpt_tw_{version}.pkl', "wb") as f:
        pkl.dump(model, f)

if __name__ == '__main__':
    ### Model name (to save to data dir)
    version = "exhaustive_0-5_930_prop"

    main(version=version)
