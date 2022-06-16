## Adding package to PATH
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(dirname(abspath(__file__)))))

## Imports
from EconomicDecisionMaking.models.ev_based_model import *
from EconomicDecisionMaking.models.hpt_predictor  import HPTModel

pd.options.mode.chained_assignment = None  # default='warn'

## Classes
class HPTTWModel(HPTModel):
    """This model implements the predictions of Prospect Theory, with Arnold Glass's Time Window"""

    def __init__(self) -> None:
        super().__init__()
        self.free_params: list[str] = ["xg", "xl", "g", "l", "tw"]

    def load_hpt_fits(self) -> None:
        with open(f'{DATA_DIR}/hpt_final.pkl', 'rb') as f:
            pt_model = pd.read_pickle(f)
            pt_best_fits = pt_model.best_fits

        self.lowest_errors = {}
        for subject in trange(57, desc='loading HPT fits'):
            best_fit = pt_best_fits[subject]
            best_fit.tw = 68
            pt_error = self.error_of_fit(subject=subject, fit=best_fit)
            self.best_fits[subject] = best_fit
            self.lowest_errors[subject] = pt_error

def main(version: str) -> None:

    ### Initialize model
    model = HPTTWModel()
    model.load_hpt_fits()

    ### Run fitting
    # model.smart_tw_fit(
    #     model.recursive_linear_fit_one_subject,
    #     precision = .001,
    #     verbose = True,
    #     start_fit = Parameters(xl=0, xg=0, g=1.0, l=1.0)
    # )

    model.iterative_exhaustive_search(
        precisions=(0.1, 0.01, 0.001),
        verbose=True,
        start=True
    )

    # model.smart_tw_fit(
    #     model.iterative_exhaustive_search_one_subject,
    #     precisions=(0.05, 0.01, 0.001),
    #     verbose=True,
    #     start=True
    # )

    mean_error    = model.finalize_and_mean_error()
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
