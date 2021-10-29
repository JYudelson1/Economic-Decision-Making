from pt_predictor import *
import pickle as pkl

pd.options.mode.chained_assignment = None  # default='warn'

class PT_TWModel(PTModel):
    """This model implements the predictions of Prospect Theory,
    with Arnold Glass's Time Window"""

    def __init__(self):
        super().__init__()
        self.free_params: List[str] = ["a", "b", "g", "l", "tw"]

def main(version: str) -> None:

    # Error type can be "absolute" or "proportional"
    error_type = "proportional"

    # Initialize model
    model = PT_TWModel()

    # Run fitting
    #start_fit = Parameters(a=1.0, b=1.0, g=1.0, l=1.0)
    #model.minimize_fit(start_fit=start_fit, verbose=True, error_type=error_type, method="Nelder-Mead")
    #model.bfs_fit(verbose=True, precision=0.05, error_type=error_type, start_fit=start_fit)
    #model.greedy_fit(verbose=True, precision=0.1, error_type=error_type, start_fit=start_fit)
    #model.simulated_annealing_fit(start_fit=start_fit, verbose=True, error_type=error_type)
    #model.exhaustive_fit(precision=0.2, verbose=True, error_type=error_type)

    precisions = (0.1, 0.01, 0.001)
    model.iterative_exhaustive_search(precisions, verbose=True, start=True)

    # Print
    model.print_info()

    # Saves data
    with open(f'{DATA_DIR}/pt_tw_{version}.pkl', "wb") as f:
        pkl.dump(model, f)

if __name__ == '__main__':
    # model name (to save to data dir)
    version = "exhaustive_iter_full_1029"
    main(version=version)
