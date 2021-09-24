from utils import *

class PredictionModel():
    """A model that uses the experiment data to generate predictions."""

    def __init__(self):
        """Initilizes the PredictionModel"""

        # Some models have free parameters. In that case, they vary from subject to subject
        # This is a mapping from subjects to Parameter values
        self.best_fits: Dict[int, Optional[Parameters]] = {}

        # Get experiment data
        self.data: pd.DataFrame = get_full_data()
        self.num_subjects: int = 1 + self.data.index[-1][0]
        self.num_days: int = 1 + int(self.data.index[-1][1])

        # Set free paramaters based on model type
        # NOTE: This should be reimplemented for each individual model
        self.free_params: Optional[List[str]] = None

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
        total_error: float = 0
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

    def load_cutoffs(self, filename: str) -> pd.DataFrame:
        """Loads i' cutoff values from a .csv file
        Inputs:
            filename: the name of the save file"""
        cutoff_prices = pd.read_csv(filename)
        # for subject in range(self.num_subjects):
        #     for day in range(self.num_days):
        #         self.data.loc[subject, "cutoff"][day] = cutoff_prices.loc[subject][day]
        return cutoff_prices

    def exhaustive_fit_one_subject(self,
                                subject: int,
                                precision: float,
                                verbose: bool = False,
                                error_type: str = "proportional") -> None:
        """Performs the exhaustive fit algorithm for one subject and saves the best fit.
        The exhaustive fit algorithm consists of iterating through all possible parameter values (with a given level of precision) and accepting the best.
        Inputs:
            subject: the participant's number in the dataframe.
            precision: the amount to increment each value when iterating through all possible values.
            verbose: set to True to get progress bars for the fitting.
            error_type: should the error be calculated as the absolute difference
                        between the prediction and the amount, or as
                        the difference in proportion of goods sold. report.docx
                        seems to use proportional."""

        lowest_error: float = float('inf')
        best_fit: Optional[Parameters] = None

        # Set correct error function
        error_fn = self.get_error_fn(error_type)

        # Get lists of all possible values for all free params
        valid_parameter_ranges: Dict[str, List[float]] = get_valid_param_ranges(precision)

        # Remove data on non-free params:
        ranges: List[List[Any]] = [valid_parameter_ranges[param] if param in self.free_params else [None] for param in ("a", "b", "g", "l", "tw") ]

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
            if fit[0] and fit[1] and fit[0] > fit[1]:
                continue

            # Predict sale amounts based on fit
            fit_params: Parameters = Parameters(*fit)
            predictions: List[int] = self.predict_one_subject(subject, fit_params)

            # Get error for the given fit
            error: float = error_fn(subject, predictions)

            # Check if it's the best so far:
            if error < lowest_error:
                lowest_error = error
                best_fit = fit_params

        self.best_fits[subject] = best_fit

    def exhaustive_fit(self, precision: float = 0.001, verbose: bool = False, error_type: str = "proportional") -> None:
        """Does the exhaustive fit algorithm for all subjects. Modifies in place.
        Inputs:
            precision: the amount to increment each value when iterating through all possible values.
            verbose: set to True to get progress bars for the fitting.
            error_type: should the error be calculated as the absolute difference
                        between the prediction and the amount, or as
                        the difference in proportion of goods sold. report.docx
                        seems to use proportional."""
        for subject in trange(self.num_subjects, disable=(not verbose), desc="Stupid Fit"):
            self.exhaustive_fit_one_subject(subject, precision, verbose, error_type)

    def greedy_fit_one_subject(self,
                               subject: int,
                               precision: float = 0.001,
                               verbose: bool = False,
                               error_type: str = "proportional",
                               start_fit: Optional[Parameters] = None) -> None:
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
        predictions: List[int] = self.predict_one_subject(subject, start_fit)
        current_error: float = error_fn(subject, predictions)

        # Changed flag will tell us whether the best fit changes over the course
        # of one iteration of the algorithm. If it doesn't change, then we are
        # in a local minimum and can stop.
        changed: bool = True

        while changed:
            # Reset changed flag
            changed = False

            # Use itertools.product to get a list of all naighbors
            all_neighbors = list(get_all_neighbors(self.best_fits[subject], precision))

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
                # Get predictions and errors for each neighbor
                predictions = self.predict_one_subject(subject, neighbor_fit)
                neighbor_error: float = error_fn(subject, predictions)
                # Save best neighbor error
                if neighbor_error < current_error:
                    # Local save and modify in plave
                    current_error = neighbor_error
                    self.best_fits[subject] = neighbor_fit
                    # Set changed flag
                    changed = True

    def greedy_fit(self,
                   precision: float = 0.001,
                   verbose: bool = False,
                   error_type: str = "proportional",
                   start_fit: Optional[Parameters] = None) -> None:
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

    def bfs_fit_one_subject(self,
                               subject: int,
                               precision: float = 0.001,
                               verbose: bool = False,
                               error_type: str = "proportional",
                               start_fit: Optional[Parameters] = None) -> None:
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
        predictions: List[int] = self.predict_one_subject(subject, start_fit)
        start_error: float = error_fn(subject, predictions)
        self.best_fits[subject] = start_fit

        # Keep track of every visited node, along with its error
        visited: Dict[Parameters, float] = {start_fit: start_error}

        # Keep track of the current best candidate
        lowest_error: float = start_error

        # BFS keeps track of a queue of nodes to be visited, and checks the oldest first
        bfs_queue: List[Parameters] = [start_fit]

        while len(bfs_queue) != 0:
            # Get current node
            current = bfs_queue.pop(0)
            current_error = visited[current]

            # Use itertools.product to get a list of all neighbors
            all_neighbors = list(get_all_neighbors(current, precision))

            # Iterate through each neighbor
            for neighbor in tqdm(all_neighbors,
                                 disable=(not verbose),
                                 leave=False,
                                 desc=f'({len(bfs_queue)} neighbors left / {len(visited)} visited)'):

                # Get fit as parameter
                neighbor_fit = Parameters(*neighbor)
                # If fit uses a and b, a must be less than b
                if neighbor_fit.a and neighbor_fit.b and neighbor_fit.a > neighbor_fit.b:
                    continue
                # Skip visited nodes
                if visited.get(neighbor_fit):
                    continue
                # Get predictions and errors for each neighbor
                predictions = self.predict_one_subject(subject, neighbor_fit)
                neighbor_error: float = error_fn(subject, predictions)
                # Mark down neighbor error
                visited[neighbor_fit] = neighbor_error
                # Add promising neighbors to queue
                if neighbor_error < current_error:
                    bfs_queue.append(neighbor_fit)
                # Save best fit
                if neighbor_error < lowest_error:
                    lowest_error = neighbor_error
                    self.best_fits[subject] = neighbor_fit

    def bfs_fit(self,
                   precision: float = 0.001,
                   verbose: bool = False,
                   error_type: str = "proportional",
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
        predictions = self.predict_one_subject(subject, fit)
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
        lower_bounds = [0.0, 0.0, 0.0, 1.0, 1]
        upper_bounds = [1.0, 1.0, 1.0, 3.5, 68]
        for i, param in enumerate(['a', 'b', 'g', 'l', 'tw']):
            if getattr(guess_copy, param):
                continue
            upper_bounds[i] = lower_bounds[i]
            setattr(guess_copy, param, lower_bounds[i])
        bounds = Bounds(lower_bounds, upper_bounds)
        # Convert the initial parameter set to a numpy array
        np_guess = np.array(guess_copy.tuplify())
        # Create the target fn
        target: Callable[np.array, float] = lambda g: self.error_fn_target(subject, g, error_type)

        # Run the minimization function!
        #hess = lambda x: np.zeros((5, 5))
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
        lower_bounds = [0.0, 0.0, 0.0, 1.0, 1]
        upper_bounds = [1.0, 1.0, 1.0, 3.5, 68]
        for i, param in enumerate(['a', 'b', 'g', 'l', 'tw']):
            if getattr(guess_copy, param):
                continue
            upper_bounds[i] = lower_bounds[i]
            setattr(guess_copy, param, lower_bounds[i])
        bounds = Bounds(lower_bounds, upper_bounds)
        minimizer_kwargs = { "method": "L-BFGS-B","bounds":bounds }
        # Convert the initial parameter set to a numpy array
        np_guess = np.array(guess_copy.tuplify())
        # Create the target fn
        target: Callable[np.array, float] = lambda g: self.error_fn_target(subject, g, error_type)

        # Run the minimization function!
        result = basinhopping(target,
                          np_guess,
                          minimizer_kwargs=minimizer_kwargs,
                          niter=10)
        result_params = Parameters(*list(result.x))
        # Remove all dummy values from the result, keep only free params
        for param in ['a', 'b', 'g', 'l', 'tw']:
            if param in guess.free_params:
                continue
            setattr(result_params, param, None)
        result_params.free_params = guess.free_params

        # Save result
        self.best_fits[subject] = result_params

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
