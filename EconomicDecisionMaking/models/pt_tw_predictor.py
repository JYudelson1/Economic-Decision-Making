## Adding package to PATH
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(dirname(abspath(__file__)))))

## Imports
from EconomicDecisionMaking.models.ev_based_model import *
from EconomicDecisionMaking.models.pt_predictor   import PTModel

pd.options.mode.chained_assignment = None  # default='warn'

## Classes
class PT_TWModel(PTModel):
    """This model implements the predictions of Prospect Theory,
    with Arnold Glass's Time Window"""

    def __init__(self):
        super().__init__()
        self.free_params: List[str] = ["a", "b", "g", "l", "tw"]

    def load_pt_fits(self) -> None:
        with open(f'{DATA_DIR}/pt_final.pkl', 'rb') as f:
            pt_model = pd.read_pickle(f)
            pt_best_fits = pt_model.best_fits

        self.lowest_errors = {}
        for subject in trange(57, desc='loading PT fits'):
            best_fit = pt_best_fits[subject]
            best_fit.tw = 68
            pt_error = self.error_of_fit(subject=subject, fit=best_fit)
            self.best_fits[subject] = best_fit
            self.lowest_errors[subject] = pt_error

def main(version: str) -> None:

    ### Initialize model
    model = PT_TWModel()
    model.load_pt_fits()
    # model.smart_tw_fit(model.iterative_exhaustive_search_one_subject,
    #                    precisions=(.1, .01, .001),
    #                    start=True,
    #                    verbose=True)
    # model.smart_tw_fit(model.recursive_linear_fit_one_subject,
    #                    precision=.001,
    #                    verbose=True,
    #                    start_fit=Parameters(a=1.0, b=1.0, g=1.0, l=1.0, tw=68))


    ### Run fitting
    #start_fit = Parameters(a=1.0, b=1.0, g=1.0, l=1.0, tw=68)
    #model.minimize_fit(start_fit=start_fit, verbose=True, method="Nelder-Mead")
    #model.bfs_fit(verbose=True, precision=0.05, start_fit=start_fit)
    #model.greedy_fit(verbose=True, precision=0.1, start_fit=start_fit)
    #model.simulated_annealing_fit(start_fit=start_fit, verbose=True)
    #model.exhaustive_fit(precision=0.2, verbose=True)
    # model.iterative_exhaustive_search(
    #     precisions = (.1, .01, .001),
    #     verbose = True,
    #     tw = 1
    # )
    for subject in tqdm([35, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47, 49, 50, 52, 55, 56]):
        model.iterative_exhaustive_search_one_subject(
            subject=subject,
            precisions=(.1, .01, .001),
            verbose=True,
            tw=1
        )

    # precisions = (0.2, 0.05, 0.01, 0.0025, 0.001)
    # model.iterative_exhaustive_search(precisions, verbose=True, start=True)

    ### Print
    model.print_info()

    ### Saves data
    with open(f'{DATA_DIR}/pt_tw_{version}.pkl', "wb") as f:
        pkl.dump(model, f)

if __name__ == '__main__':
    ### model name (to save to data dir)
    # version = "exhaustive_iter_1030"
    # main(version=version)
    ## Extrapolating tw values
    # with open(f'{DATA_DIR}/pt_exhaustive_iter_full_111.pkl', "rb") as f:
    #     pt_model = pd.read_pickle(f)
    # pt_tw_model = PT_TWModel()
    # for subject in trange(pt_model.num_subjects):
    #     best_tw = 67
    #     lowest_error = 100
    #     fit = pt_model.best_fits[subject]
    #     pt_tw_model.best_fits[subject] = Parameters(a=fit.a, b=fit.b, g=fit.g, l=fit.l, tw=67)
    #     for tw in trange(1, 68, leave=False):
    #         # Predict sale amounts based on fit
    #         pt_tw_model.best_fits[subject].tw = tw
    #         predictions: List[int] = pt_tw_model.predict_one_subject(subject, pt_tw_model.best_fits[subject])
    #
    #         # Get error for the given fit
    #         error: float = pt_tw_model.mean_error_one_subject_proportion(subject, predictions)
    #         if error <= lowest_error:
    #             best_tw = tw
    #             lowest_error = error
    #     pt_tw_model.best_fits[subject].tw = best_tw
    # pt_tw_model.print_info()
    # with open(f'{DATA_DIR}/pt_tw_extrap_1119.pkl', "wb") as f:
    #     pkl.dump(pt_tw_model, f)
    main("tw1_test")
