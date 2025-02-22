## Adding package to PATH
# import sys
# from os.path import dirname, abspath
# sys.path.append(dirname(dirname(dirname(abspath(__file__)))))

## Imports
from EconomicDecisionMaking.utils import *

class PredictionModel():
    """A model that uses the experiment data to generate predictions."""

    def __init__(self):
        """Initializes the PredictionModel"""

        # Get experiment data
        self.data: pd.DataFrame = get_full_data()
        self.num_subjects: int  = 1 + self.data.index[-1][0]
        self.num_days:     int  = 1 + int(self.data.index[-1][1])

        # Some models have free parameters. In that case, they vary from subject to subject
        # This is a mapping from subjects to Parameter values
        self.best_fits:     Dict[int, Parameters] = {}
        self.all_best_fits: Dict[int, List[Parameters]] = {subject: [] for subject in range(self.num_subjects)}

        # Set free paramaters based on model type
        # NOTE: This should be reimplemented for each individual model
        self.free_params: List[str]

    def get_data_one_subject(self, subject: int) -> pd.DataFrame:
        """Returns all data corresponding to the given subject.
        Inputs:
            subject: the participant's number in the dataframe."""
        return self.data.loc[subject]

    def predict_one_subject(self, subject: int, fit: Optional[Parameters] = None) -> List[int]:
        """Returns the predicted sale amounts based on the given fit.
        Inputs:
            subject: the participant's number in the dataframe.
            fit: a Parameters object that contains some values to be used in the prediction."""
        # Note: this should be implemented in each individual model.
        raise NotImplementedError

    def mean_error_one_subject_absolute(self, subject: int, predictions: List[int]) -> float:
        """Evaluates the mean error for one subject.
        This error is based on the difference between predicted number of units sold and actual number of units sold.
        Inputs:
            subject: the participant's number in the dataframe.
            predictions: the model's predictions for the number of units to be sold each day by this participant."""
        total_error: float = 0
        subject_data: pd.DataFrame = self.get_data_one_subject(subject)

        # Iterate through the days and sum the error
        for day in range(self.num_days):
            day_error: float = abs(predictions[day] - subject_data['sold'][day])
            total_error += day_error

        # Find the mean
        mean_error: float = total_error / self.num_days

        return mean_error

    def mean_error_one_subject_proportion(self, subject: int, predictions: List[int]) -> float:
        """Evaluates the mean error for one subject.
        This error is based on the difference between predicted proportion of units sold and actual proportion of units sold.
        This is more in line with report.docx than mean_error_one_subject_absolute.
        Inputs:
            subject: the participant's number in the dataframe.
            predictions: the model's predictions for the number of units to be sold each day by this participant."""
        d_0: int = 0 # Number of days for which the participant had no goods stored
        total_error:  float = 0
        subject_data: pd.DataFrame = self.get_data_one_subject(subject)

        # Iterate through the days and sum the error
        for day in range(self.num_days):
            if subject_data['stored'][day] == 0:
                d_0 += 1
                continue
            day_error: float = abs(predictions[day] - subject_data['sold'][day]) / subject_data['stored'][day]
            total_error += day_error

        # Find the mean
        mean_error: float = total_error / (self.num_days - d_0)

        return mean_error

    def get_error_fn(self, error_type: str) -> Callable[[int, List[int]], float]:
        """Returns the right error function based on the string input.
        Inputs:
            error_type: should the error be calculated as the absolute difference
                        between the prediction and the amount, or as
                        the difference in proportion of goods sold. report.docx
                        seems to use proportional."""
        if error_type == "proportional":
            error_fn = self.mean_error_one_subject_proportion
        elif error_type == "absolute":
            error_fn = self.mean_error_one_subject_absolute
        else:
            raise ValueError("Error type must be proportional or absolute!")

        return error_fn

    def mean_error_all_subjects(self, error_type: str = "proportional", verbose: bool = True, save_predictions: bool = False) -> List[float]:
        """Evaluates the average mean error across all subjects.
        Inputs:
            error_type: should the error be calculated as the absolute difference
                        between the prediction and the amount, or as
                        the difference in proportion of goods sold. report.docx
                        seems to use proportional.
            verbose: uses tqdm progess bar if true
            save_predictions: if true, saves each prediction to self.data"""
        errors: List[float] = []

        # Set correct error function
        error_fn = self.get_error_fn(error_type)

        # Get errors for each subject
        for subject in trange(self.num_subjects, disable=(not verbose), desc="All Errors"):
            # Get the already-determined best fit paramters
            best_fit: Optional[Parameters] = self.best_fits.get(subject)

            # Predict sale amounts based on best_fit
            predictions: List[int] = self.predict_one_subject(subject, best_fit)

            # Get error
            error: float = error_fn(subject, predictions)
            errors.append(error)

            # If save_predictions, store predictions in self.data
            if save_predictions:
                self.data.loc[subject, "prediction"] = predictions

        return errors

    def finalize_and_mean_error(self, error_type: str = "proportional", verbose: bool = True) -> float:
        """ Returns the mean of all mean errors.
        Additionally, saves all best fit predictions to self.data
        Inputs:
            error_type: should the error be calculated as the absolute difference
                        between the prediction and the amount, or as
                        the difference in proportion of goods sold. report.docx
                        seems to use proportional.
            verbose: uses tqdm progess bar if true"""

        all_errors: List[float] = self.mean_error_all_subjects(error_type,
                                                               verbose,
                                                               save_predictions=True)

        return np.mean(all_errors)

    def std_dev_of_error(self, error_type: str = "proportional", verbose: bool = True) -> float:
        """ Returns the standard deviation of all mean errors.
        Inputs:
            error_type: should the error be calculated as the absolute difference
                        between the prediction and the amount, or as
                        the difference in proportion of goods sold. report.docx
                        seems to use proportional.
            verbose: uses tqdm progess bar if true"""

        all_errors: List[float] = self.mean_error_all_subjects(error_type,
                                                               verbose)

        return np.std(all_errors)

    def print_info(self, error_type="proportional"):
        """Prints mean error and std deviation of error of the model.
           Additionally, saves prediction data to the model's DataFrame,
           and then prints the dataframe."""

        mean_error = self.finalize_and_mean_error(error_type=error_type)
        std_deviation = self.std_dev_of_error(error_type=error_type)

        print(f'mean_error = {mean_error}')
        print(f'std_dev = {std_deviation}')
        print(self.data)

    def load_cutoffs(self, filename: str) -> pd.DataFrame:
        """Loads i' cutoff values from a .csv file
        Inputs:
            filename: the name of the save file"""
        cutoff_prices = pd.read_csv(filename)
        # for subject in range(self.num_subjects):
        #     for day in range(self.num_days):
        #         self.data.loc[subject, "cutoff"][day] = cutoff_prices.loc[subject][day]
        return cutoff_prices

    def error_of_fit(self, subject: int, fit: Parameters) -> float:
        predictions: List[int] = self.predict_one_subject(subject, fit)
        return self.mean_error_one_subject_proportion(subject, predictions)

    def exhaustive_fit_one_subject(self,
                                subject:    int,
                                precision:  float,
                                verbose:    bool = False,
                                error_type: str = "proportional",
                                tw:         Optional[int] = None) -> None:
        """Performs the exhaustive fit algorithm for one subject and saves the best fit.
        The exhaustive fit algorithm consists of iterating through all possible parameter values (with a given level of precision) and accepting the best.
        Inputs:
            subject: the participant's number in the dataframe.
            precision: the amount to increment each value when iterating through all possible values.
            verbose: set to True to get progress bars for the fitting.
            error_type: should the error be calculated as the absolute difference
                        between the prediction and the amount, or as
                        the difference in proportion of goods sold. report.docx
                        seems to use proportional.
            tw: if provided, tw will stay at this constant level."""

        lowest_error: float  = float('inf')
        best_fit: Parameters = Parameters(1, 1, 1, 1, 68)

        # Set correct error function
        error_fn = self.get_error_fn(error_type)

        # Get lists of all possible values for all free params
        valid_parameter_ranges: Dict[str, List[float]] = get_valid_param_ranges(precision)
        if tw is not None:
            valid_parameter_ranges["tw"] = [tw]

        # Remove data on non-free params:
        ranges: List[List[Any]] = [valid_parameter_ranges[param] if param in self.free_params
                                                                 else [None]
                                                                 for param in PARAM_LIST ]

        # Get all possible values via cartesian product
        all_possible_fits = product(*ranges)

        # Iterate through every possible value
        iterations = 1
        for p_range in ranges:
            iterations *= len(p_range)
        for fit in tqdm(all_possible_fits,
                        disable=(not verbose),
                        leave=False,
                        total=iterations,
                        desc="Attempting all fits..."):
            # Skip fits where a > b
            if fit[0] is not None and fit[1] is not None and fit[0] > fit[1]:
                continue
            # Keep fits where xl-.1 <= xg <= xl
            if (fit[2] is not None and fit[3] is not None) and (fit[2] > fit[3] or fit[2] < fit[3]-0.1):
                continue

            # Predict sale amounts based on fit
            fit_params: Parameters = Parameters(*fit)
            predictions: List[int] = self.predict_one_subject(subject, fit_params)

            # Get error for the given fit
            error: float = error_fn(subject, predictions)

            # Check if it's the best so far:
            if error < lowest_error:
                lowest_error = error
                best_fit     = fit_params
                self.all_best_fits[subject] = [fit_params]
            elif error == lowest_error:
                if fit_params.a is not None and fit_params.a >= best_fit.a:
                    best_fit = fit_params
                elif not fit_params.a:
                    best_fit = fit_params
                self.all_best_fits[subject].append(fit_params)

        self.best_fits[subject] = best_fit

    def exhaustive_fit(self, precision: float = 0.001, verbose: bool = False, error_type: str = "proportional", tw: Optional[int] = None) -> None:
        """Does the exhaustive fit algorithm for all subjects. Modifies in place.
        Inputs:
            precision: the amount to increment each value when iterating through all possible values.
            verbose: set to True to get progress bars for the fitting.
            error_type: should the error be calculated as the absolute difference
                        between the prediction and the amount, or as
                        the difference in proportion of goods sold. report.docx
                        seems to use proportional.
            tw: if provided, tw will stay at this constant level."""
        for subject in trange(self.num_subjects, disable=(not verbose), desc="Exhaustive Fit"):
            self.exhaustive_fit_one_subject(subject, precision, verbose, error_type, tw=tw)

    def exhaustive_fit_with_guess_one_subject(self,
                                subject:        int,
                                precision:      float,
                                prev_precision: float,
                                verbose:        bool = False,
                                error_type:     str = "proportional",
                                tw:             Optional[int] = None) -> None:
        """Performs the exhaustive fit algorithm for one subject and saves the best fit.
        The exhaustive fit algorithm consists of iterating through all possible parameter values (with a given level of precision) and accepting the best.
        Inputs:
            subject: the participant's number in the dataframe.
            precision: the amount to increment each value when iterating through all possible values.
            verbose: set to True to get progress bars for the fitting.
            error_type: should the error be calculated as the absolute difference
                        between the prediction and the amount, or as
                        the difference in proportion of goods sold. report.docx
                        seems to use proportional.
            tw: if provided, tw will stay at this constant level."""

        prev_best_fit: Parameters = self.best_fits[subject]
        initial_pred = self.predict_one_subject(subject=subject, fit=prev_best_fit)
        lowest_error = self.mean_error_one_subject_proportion(subject, initial_pred)
        best_fit: Parameters = prev_best_fit

        # Set correct error function
        error_fn = self.get_error_fn(error_type)

        # Get lists of all possible values for all free params
        low  = lambda fit, old_precision, floor: max(floor, fit - old_precision/2) if fit else 0
        high = lambda fit, old_precision, ceil:  min(ceil,  fit + old_precision/2) if fit else 0
        valid_parameter_ranges: Dict[str, List[float]] = {
            "a": list(np.arange(
                            low(prev_best_fit.a,  prev_precision, precision),
                            high(prev_best_fit.a, prev_precision, 1) + EPS,
                            precision
                )),
            "b": list(np.arange(
                            low(prev_best_fit.b,  prev_precision, precision),
                            high(prev_best_fit.b, prev_precision, 1) + EPS,
                            precision
                )),
            "xg": list(np.arange(
                            low(prev_best_fit.a,  prev_precision, precision),
                            high(prev_best_fit.a, prev_precision, 1) + EPS,
                            precision
                )),
            "xl": list(np.arange(
                            low(prev_best_fit.b,  prev_precision, precision),
                            high(prev_best_fit.b, prev_precision, 1) + EPS,
                            precision
                )),
            "g": list(np.arange(
                            low(prev_best_fit.g,  prev_precision, precision),
                            high(prev_best_fit.g, prev_precision, 1) + EPS,
                            precision
                )),
            "l": list(np.arange(
                            low(prev_best_fit.l,  prev_precision, 1),
                            high(prev_best_fit.l, prev_precision, 2) + EPS,
                            precision
                )),
            "tw": list(np.arange(2, NUM_DAYS, 1))
        }
        if tw is not None:
            valid_parameter_ranges["tw"] = [tw]

        # Remove data on non-free params:
        ranges: List[List[Any]] = [valid_parameter_ranges[param] if param in self.free_params
                                                                 else [None]
                                                                 for param in PARAM_LIST ]

        # Get all possible values via cartesian product
        all_possible_fits = product(*ranges)

        # Iterate through every possible value
        iterations = 1
        for range in ranges:
            iterations *= len(range)
        for fit in tqdm(all_possible_fits,
                        disable=(not verbose),
                        leave=False,
                        total=iterations,
                        desc="Attempting all fits..."):
            # Skip fits where a > b
            if fit[0] is not None and fit[1] is not None and fit[0] > fit[1]:
                continue
            # Keep fits where xl-.1 <= xg <= xl
            if (fit[2] is not None and fit[3] is not None) and (fit[2] > fit[3] or fit[2] < fit[3]-0.1):
                continue

            # Predict sale amounts based on fit
            fit_params: Parameters = Parameters(*fit)
            predictions: List[int] = self.predict_one_subject(subject, fit_params)

            # Get error for the given fit
            error: float = error_fn(subject, predictions)

            # Check if it's the best so far:
            if error < lowest_error:
                lowest_error = error
                best_fit     = fit_params
                self.all_best_fits[subject] = [fit_params]
            elif error == lowest_error:
                if fit_params.a is not None and fit_params.a >= best_fit.a:
                    best_fit = fit_params
                elif not fit_params.a:
                    best_fit = fit_params
                self.all_best_fits[subject].append(fit_params)

        self.best_fits[subject] = best_fit

    def exhaustive_fit_with_guess(self,
                                  precision: float,
                                  prev_precision: float,
                                  verbose: bool = False,
                                  error_type: str = "proportional",
                                  tw: Optional[int] = None) -> None:
        """Does the exhaustive fit algorithm for all subjects. Modifies in place.
        Use case is after exhaustive fit, to hone in on the previous best guess.
        Inputs:
            precision: the amount to increment each value when iterating through all possible values.
            prev_precision: the level of precision the previous iteration of exhaustive search used.
            verbose: set to True to get progress bars for the fitting.
            error_type: should the error be calculated as the absolute difference
                        between the prediction and the amount, or as
                        the difference in proportion of goods sold. report.docx
                        seems to use proportional.
            tw: if provided, tw will stay at this constant level."""
        for subject in trange(self.num_subjects, disable=(not verbose), desc=f'Exhaustive Fit (p={precision})'):
            self.exhaustive_fit_with_guess_one_subject(subject, precision, prev_precision, verbose, error_type, tw=tw)

    def iterative_exhaustive_search_one_subject(self,
                                                subject:    int,
                                                precisions: Tuple[float, ...],
                                                verbose:    bool = False,
                                                error_type: str = "proportional",
                                                start:      bool = True,
                                                tw:         Optional[int] = None) -> None:
        """Does the iterative exhaustive fit algorithm for all subjects. Modifies in place.
        Successively hones in on smaller regions of the search space.
        Inputs:
            subject: the participant's number in the dataframe.
            precisions: A decreasing list of precision values
            verbose: set to True to get progress bars for the fitting.
            error_type: should the error be calculated as the absolute difference
                        between the prediction and the amount, or as
                        the difference in proportion of goods sold. report.docx
                        seems to use proportional.
            start: if True, also run first exhaustive fit with first precision value.
            tw: if provided, tw will stay at this constant level."""
        if start:
            self.exhaustive_fit_one_subject(subject=subject, precision=precisions[0],
                                                             verbose=verbose, error_type=error_type,
                                                             tw=tw)
            # store predictions in self.data
            predictions: List[int] = self.predict_one_subject(subject, self.best_fits[subject])
            self.data.loc[subject, "prediction"] = predictions

        for i in range(len(precisions) - 1):
            self.exhaustive_fit_with_guess_one_subject(subject=subject, precision=precisions[i+1],
                                                                        prev_precision=precisions[i],
                                                                        verbose=verbose, error_type=error_type,
                                                                        tw=tw)
            # store predictions in self.data
            predictions: List[int] = self.predict_one_subject(subject, self.best_fits[subject])
            self.data.loc[subject, "prediction"] = predictions

    def iterative_exhaustive_search(self,
                                    precisions: Tuple[float, ...],
                                    verbose: bool     = False,
                                    error_type: str   = "proportional",
                                    start: bool       = True,
                                    tw: Optional[int] = None) -> None:
        """Does the iterative exhaustive fit algorithm for all subjects. Modifies in place.
        Successively hones in on smaller regions of the search space.
        Inputs:
            precisions: A decreasing list of precision values
            verbose: set to True to get progress bars for the fitting.
            error_type: should the error be calculated as the absolute difference
                        between the prediction and the amount, or as
                        the difference in proportion of goods sold. report.docx
                        seems to use proportional.
            start: if True, also run first exhaustive fit with first precision value.
            tw: if provided, tw will stay at this constant level."""
        if start:
            self.exhaustive_fit(precision=precisions[0], verbose=True, error_type=error_type, tw=tw)
            if verbose:
                self.print_info()
            else:
                self.mean_error_all_subjects(error_type, False, save_predictions=True)

        for i in range(len(precisions) - 1):
            self.exhaustive_fit_with_guess(precision=precisions[i+1],
                                           prev_precision=precisions[i],
                                           verbose=True,
                                           error_type=error_type,
                                           tw=tw)
            if verbose:
                self.print_info()
            else:
                self.mean_error_all_subjects(error_type, False, save_predictions=True)

    def greedy_fit_one_subject(self,
                               subject:    int,
                               precision:  float = 0.001,
                               verbose:    bool = False,
                               error_type: str = "proportional",
                               start_fit:  Optional[Parameters] = None) -> None:
        """Performs the greedy fit algorithm for one subject and saves the best fit.
        The greedy fit algorithm consists of starting at one spot in the search space
            and exclusively tarveling to the neighbor with the lowest error.
        NOTE: Greedy fit doesn't guarantee optimality, but it's fast. Don't use for actual values,
            but useful as a sanity check.
        NOTE: modifies self.best_fits in place
        Inputs:
            subject: the participant's number in the dataframe.
            precision: the amount to increment each value when traversing the search space.
            verbose: set to True to get progress bars for the fitting.
            error_type: should the error be calculated as the absolute difference
                        between the prediction and the amount, or as
                        the difference in proportion of goods sold. report.docx
                        seems to use proportional.
            start_fit: the first parameters to use when traversing the search space.
            """

        # Ensure greedy fit has a starting point
        if not start_fit:
            raise ValueError("Greedy fit needs a starting parameter set!")

        # Set correct error function
        error_fn = self.get_error_fn(error_type)

        # Get starting error
        self.best_fits[subject] = start_fit
        predictions: List[int]  = self.predict_one_subject(subject, start_fit)
        current_error: float    = error_fn(subject, predictions)

        # Frontier contains the points to be visited
        frontier = {start_fit}
        visited = {start_fit: current_error}

        while frontier:

            # Use itertools.product to get a list of all naighbors
            current = frontier.pop()
            new_error = visited[current]
            if new_error < current_error:
                continue

            all_neighbors = list(get_all_neighbors(current, precision))

            # Iterate through each neighbor
            for neighbor in tqdm(all_neighbors,
                                 disable=(not verbose),
                                 leave=False,
                                 desc="Checking neighbors"):
                # Get fit as parameter
                neighbor_fit = Parameters(*neighbor)
                # If fit uses a and b, a must be less than b
                if neighbor_fit.a and neighbor_fit.b and neighbor_fit.a > neighbor_fit.b:
                    continue
                if visited.get(neighbor_fit) is not None:
                    continue
                # Get predictions and errors for each neighbor
                predictions = self.predict_one_subject(subject, neighbor_fit)
                neighbor_error: float = error_fn(subject, predictions)
                visited[neighbor_fit] = neighbor_error
                # Save best neighbor error
                if neighbor_error <= current_error:
                    # Local save and modify in plave
                    if neighbor_error < current_error:
                        current_error = neighbor_error
                        self.best_fits[subject] = neighbor_fit
                        self.all_best_fits[subject] = []
                    self.all_best_fits[subject].append(neighbor_fit)
                    frontier.add(neighbor_fit)
                    if neighbor_error == 0:
                        return


    def greedy_fit(self,
                   precision:  float = 0.001,
                   verbose:    bool  = False,
                   error_type: str   = "proportional",
                   start_fit:  Optional[Parameters] = None) -> None:
        """Does the greedy fit algorithm for all subjects. Modifies in place.
        Inputs:
            precision: the amount to increment each value when traversing the search space.
            verbose: set to True to get progress bars for the fitting.
            error_type: should the error be calculated as the absolute difference
                        between the prediction and the amount, or as
                        the difference in proportion of goods sold. report.docx
                        seems to use proportional.
            start_fit: the first parameters to use when traversing the search space."""
        for subject in trange(self.num_subjects, disable=(not verbose), desc="Greedy Fit"):
            self.greedy_fit_one_subject(subject, precision, verbose, error_type, start_fit)
            print(subject, self.error_of_fit(subject, self.best_fits[subject]), self.best_fits[subject])

    def bfs_fit_one_subject(self,
                               subject:    int,
                               precision:  float = 0.001,
                               verbose:    bool  = False,
                               error_type: str   = "proportional",
                               start_fit:  Optional[Parameters] = None) -> None:
        """Performs the BFS fit algorithm for one subject and saves the best fit.
        The BFS fit algorithm consists of starting at one spot in the search space
            and eaxploring each neighbor with a lower error..
        NOTE: bfs fit doesn't guarantee optimality.
        NOTE: modifies self.best_fits in place
        Inputs:
            subject: the participant's number in the dataframe.
            precision: the amount to increment each value when traversing the search space.
            verbose: set to True to get progress bars for the fitting.
            error_type: should the error be calculated as the absolute difference
                        between the prediction and the amount, or as
                        the difference in proportion of goods sold. report.docx
                        seems to use proportional.
            start_fit: the first parameters to use when traversing the search space.
            """

        # Ensure greedy fit has a starting point
        if not start_fit:
            raise ValueError("BFS fit needs a starting parameter set!")

        # Set correct error function
        error_fn = self.get_error_fn(error_type)

        # Get initial error
        predictions: List[int]  = self.predict_one_subject(subject, start_fit)
        start_error: float      = error_fn(subject, predictions)
        self.best_fits[subject] = start_fit

        # Keep track of every visited node, along with its error
        visited: Dict[Parameters, bool] = {}

        # Keep track of the current best candidate
        lowest_error: float = start_error

        # BFS keeps track of a queue of nodes to be visited, and checks the oldest first
        bfs_queue: List[Tuple[Parameters, float]] = [(start_fit, float('inf'))]
        print("", end="\r")
        while len(bfs_queue) != 0:
            # Get current node
            print(f'{len(bfs_queue)} nodes remaining / {len(visited)} visited', end="\r")
            current, parent_error = bfs_queue.pop(0)

            if visited.get(current):
                continue

            visited[current] = True

            predictions = self.predict_one_subject(subject, current)
            error = error_fn(subject, predictions)

            if error > parent_error:
                continue

            if error < lowest_error:
                lowest_error = error
                self.best_fits[subject] = current
                self.all_best_fits[subject] = [current]
            elif error == lowest_error:
                self.all_best_fits[subject].append(current)

            # Use itertools.product to get a list of all neighbors
            all_neighbors = list(get_all_neighbors(current, precision))

            # Iterate through each neighbor
            for neighbor in all_neighbors:

                # Get fit as parameter
                neighbor_fit = Parameters(*neighbor)
                # If fit uses a and b, a must be less than b
                if neighbor_fit.a is not None and neighbor_fit.b is not None and neighbor_fit.a > neighbor_fit.b:
                    continue
                # Skip visited nodes
                if visited.get(neighbor_fit):
                    continue
                # Get predictions and errors for each neighbor
                bfs_queue.append((neighbor_fit, error))

    def bfs_fit(self,
                   precision:  float = 0.001,
                   verbose:    bool  = False,
                   error_type: str   = "proportional",
                   start_fit: Optional[Parameters] = None) -> None:
        """Does the BFS fit algorithm for all subjects. Modifies in place.
        Inputs:
            precision: the amount to increment each value when traversing the search space.
            verbose: set to True to get progress bars for the fitting.
            error_type: should the error be calculated as the absolute difference
                        between the prediction and the amount, or as
                        the difference in proportion of goods sold. report.docx
                        seems to use proportional.
            start_fit: the first parameters to use when traversing the search space."""
        for subject in trange(self.num_subjects, disable=(not verbose), desc="BFS Fit"):
            self.bfs_fit_one_subject(subject, precision, verbose, error_type, start_fit)

    def error_fn_target(self, subject: int, params: np.array, error_type: str) -> float:
        """A wrapper around the error functions that works with np arrays.
        Allows scipy.optimize to minimize the value of the function directly.
        Inouts:
            subject: the participant's number in the dataframe.
            params: the parameters to be evaluated as a fit.
            error_type: should the error be calculated as the absolute difference
                        between the prediction and the amount, or as
                        the difference in proportion of goods sold. report.docx
                        seems to use proportional.
            """
        error_fn = self.get_error_fn(error_type)
        fit = Parameters(*list(params))

        # Scale TW up
        # Note: TW was scaled down by a factor of TW_FACTOR
        #       for the sake of making minimize_fit and
        #       simulated_annealing_fit work well with a
        #       small step size.
        if fit.tw:
            fit.tw = TW_FACTOR * fit.tw
        predictions  = self.predict_one_subject(subject, fit)
        error: float = error_fn(subject, predictions)
        return error

    def minimize_fit_one_subject(self, subject: int, guess: Parameters, error_type: str, method: str) -> None:
        """Uses scipy.optimize.minimize to minimize the error function.
        Modifies in place.
        Inputs:
            subject: the participant's number in the dataframe.
            guess: starting parameters
            error_type: absolute or proportional
            method: optimization method. Can be ‘Nelder-Mead’, ‘L-BFGS-B’,
                                                ‘TNC’, ‘SLSQP’, or ‘trust-constr’"""

        # Initialize the bounds for each parameter
        guess_copy = guess.deepcopy()

        # Note: TW is scaled down by a factor of TW_FACTOR
        #       for the sake of making minimize_fit and
        #       simulated_annealing_fit work well with a
        #       small step size.
        if guess_copy.tw:
            guess_copy.tw = guess_copy.tw / TW_FACTOR

        # Setting bounds
        lower_bounds = [0.0, 0.0, 0.0, 1.0, 1 / TW_FACTOR]
        upper_bounds = [1.0, 1.0, 1.0, 2, 68 / TW_FACTOR]
        for i, param in enumerate(['a', 'b', 'g', 'l', 'tw']):
            if getattr(guess_copy, param):
                continue
            upper_bounds[i] = lower_bounds[i]
            setattr(guess_copy, param, lower_bounds[i])
        bounds = Bounds(lower_bounds, upper_bounds)

        # Convert the initial parameter set to a numpy array
        np_guess = np.array(guess_copy.tuplify())
        # Create the target fn
        target: Callable[[np.array], float] = lambda g: self.error_fn_target(subject, g, error_type)

        # Run the minimization function!
        # hess = lambda x: np.zeros((5, 5))
        with catch_warnings():
            simplefilter('ignore')
            result = minimize(target,
                              np_guess,
                              method=method,
                              #hess=hess,
                              bounds=bounds)
        result_params = Parameters(*list(result.x))

        # Remove all dummy values from the result, keep only free params
        for param in ['a', 'b', 'g', 'l', 'tw']:
            if param in guess.free_params:
                continue
            setattr(result_params, param, None)
        result_params.free_params = guess.free_params
        if result_params.tw:
            result_params.tw = ceil(result_params.tw * TW_FACTOR)

        # Save result
        self.best_fits[subject] = result_params

    def minimize_fit(self, start_fit: Parameters, verbose: bool = False, error_type: str = "proportional", method='trust-constr') -> None:
        """Does fit optimization algorithm for all subjects. Modifies in place.
        Outsources the minimization to scipy.optimize.minimize.
        Inputs:
            verbose: set to True to get progress bars for the fitting.
            error_type: should the error be calculated as the absolute difference
                        between the prediction and the amount, or as
                        the difference in proportion of goods sold. report.docx
                        seems to use proportional.
            start_fit: the first parameters to use when traversing the search space.
            method: optimization method. Can be ‘Nelder-Mead’, ‘L-BFGS-B’,
                                                ‘TNC’, ‘SLSQP’, or ‘trust-constr’"""
        for subject in trange(self.num_subjects, disable=(not verbose), desc=f'{method} Fit'):
            self.minimize_fit_one_subject(subject, start_fit, error_type, method)

    def simulated_annealing_fit_one_subject(self, subject: int, guess: Parameters, error_type: str) -> None:
        """Uses scipy.optimize.anneal to minimize the error function.
        Modifies in place.
        Inputs:
            subject: the participant's number in the dataframe.
            guess: starting parameters
            error_type: absolute or proportional
            """

        # Initialize the bounds for each parameter
        guess_copy = guess.deepcopy()
        # Note: TW is scaled down by a factor of TW_FACTOR
        #       for the sake of making minimize_fit and
        #       simulated_annealing_fit work well with a
        #       small step size.
        if guess_copy.tw:
            guess_copy.tw = guess_copy.tw / TW_FACTOR

        # Setting bounds
        lower_bounds = [0.0, 0.0, 0.0, 1.0, 1 / TW_FACTOR]
        upper_bounds = [1.0, 1.0, 1.0, 2, 68 / TW_FACTOR]
        for i, param in enumerate(['a', 'b', 'g', 'l', 'tw']):
            if getattr(guess_copy, param):
                continue
            upper_bounds[i] = lower_bounds[i]
            setattr(guess_copy, param, lower_bounds[i])
        bounds = Bounds(lower_bounds, upper_bounds)
        minimizer_kwargs = { "method": "L-BFGS-B",
                            "bounds":bounds,
                            # "options":
                            #     {
                            #         "eps":np.array([.00001, .00001, .00001, .01, 1/TW_FACTOR**2])
                            #     }
                            }

        # Convert the initial parameter set to a numpy array
        np_guess = np.array(guess_copy.tuplify())

        # Create the target fn
        target: Callable[[np.array], float] = lambda g: self.error_fn_target(subject, g, error_type)

        # Run the minimization function!
        result = basinhopping(target,
                          np_guess,
                          minimizer_kwargs=minimizer_kwargs,
                          niter=40,
                          T=0.1,
                          stepsize=0.1
                          )
        result_params = Parameters(*list(result.x))

        # Remove all dummy values from the result, keep only free params
        for param in ['a', 'b', 'g', 'l', 'tw']:
            if param in guess.free_params:
                continue
            setattr(result_params, param, None)
        result_params.free_params = guess.free_params
        if result_params.tw:
            result_params.tw = ceil(result_params.tw * TW_FACTOR)

        # Save result
        self.best_fits[subject] = result_params
        self.all_best_fits[subject].append(result_params)

    def simulated_annealing_fit(self, start_fit: Parameters, verbose: bool = False, error_type: str = "proportional") -> None:
        """Does simulated annealing algorithm for all subjects. Modifies in place.
        Outsources the minimization to scipy.optimize.anneal.
        Inputs:
            verbose: set to True to get progress bars for the fitting.
            error_type: should the error be calculated as the absolute difference
                        between the prediction and the amount, or as
                        the difference in proportion of goods sold. report.docx
                        seems to use proportional.
            start_fit: the first parameters to use when traversing the search space."""
        for subject in trange(self.num_subjects, disable=(not verbose), desc=f'Basinhopping Fit'):
            self.simulated_annealing_fit_one_subject(subject, start_fit, error_type)

    def smart_tw_fit(self, fit_one_fn: Callable, **kwargs: Dict[Any, Any]) -> None:
        skip_this_subject = {i: False for i in range(57)}

        for tw in trange(67, 0, -1, desc='descending through tws', leave=True):
            # Make sure the right tw will be used
            if kwargs.get("start_fit"):
                kwargs['start_fit'].tw = tw
            else:
                kwargs['tw'] = tw

            # Iterate through each subject
            for subject in trange(57, desc=f'Fitting at tw={tw}', leave=False):
                # Skip the properly fitted ones
                if self.lowest_errors[subject] == 0:
                    continue
                if skip_this_subject[subject]:
                    continue

                # Save older values for comparison
                prev_lowest_error = self.lowest_errors[subject]
                prev_best_fit = self.best_fits[subject]

                # Do the fitting!
                fit_one_fn(subject=subject, **kwargs)

                # Compare to old values
                new_error = self.error_of_fit(subject, self.best_fits[subject])
                if prev_lowest_error < new_error: #If error goes up
                    self.best_fits[subject] = prev_best_fit
                    skip_this_subject[subject] = True
                else: #If tw=tw fits aren't of the same or lesser error as tw=tw+1
                    no_new_fits = True
                    for fit in self.all_best_fits[subject]:
                        if fit.tw == tw:
                            no_new_fits = False
                            break
                    if self.best_fits[subject].tw == tw: no_new_fits = False
                    if no_new_fits:
                        skip_this_subject[subject] = True
                if skip_this_subject[subject]:
                    print(f'subject {subject} has stopped at tw={tw}')
