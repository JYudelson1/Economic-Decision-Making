from HPTTW_utils import *


class PredictionModel():
    """A model that uses the experiment data to generate predictions."""

    def __init__(self):
        """Initializes the PredictionModel"""

        # Get experiment data
        self.data: pd.DataFrame = get_full_data()
        self.num_subjects: int = 1 + self.data.index[-1][0]
        self.num_days: int = 1 + int(self.data.index[-1][1])

        # Some models have free parameters. In that case, they vary from subject to subject
        # This is a mapping from subjects to Parameter values
        self.best_fits: Dict[int, Optional[Parameters]] = {}
        self.all_best_fits: Dict[int, List[Parameters]] = {subject: [] for subject in range(self.num_subjects)}

        # Set free paramaters based on model type
        # NOTE: This should be reimplemented for each individual model
        self.free_params: Optional[List[str]] = None

    def get_data_one_subject(self, subject: int) -> pd.DataFrame:
        """Returns all data corresponding to the given subject.
        Inputs:
            subject: the participant's number in the dataframe."""
        return self.data.loc[subject]

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
        d_0: int = 0  # Number of days for which the participant had no goods stored
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

    def mean_error_all_subjects(self, error_type: str = "proportional", verbose: bool = True,
                                save_predictions: bool = False) -> List[float]:
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

    def generate_cutoffs(self, fit: Parameters) -> List[int]:
        """Generates the i' cutoffs for a given fit.
        # It turns out that cutoff prices depend ONLY on the fit.
        # That is to say, the same fit with the same model should always generate the same cutoff prices.
        Inputs:
            fit: the Parameters used to calculate the cutoff values. """
        cutoffs: List[int] = [1]
        iter = range(1, self.num_days)
        for day in tqdm(iter, desc=f'Cutoffs', leave=False, disable=True):
            price: int = cutoffs[day - 1]  # price has to be at least the last cutoff value

            # Separate EV function for day 2 (with index 1)
            if day == 1:
                ev = self.expected_value_day_2(price, 1, fit, tuple(cutoffs))
            else:
                ev = self.expected_value(day, price, 1, fit, tuple(cutoffs))

            while ev <= 0 and price < 15:
                # Continually increment price until it is worthwhile to sell
                price += 1
                ev = self.expected_value(day, price, 1, fit, tuple(cutoffs))
            cutoffs.append(price)
        return cutoffs

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
        best_fit: Optional[Parameters] = Parameters(1, 1, 1, 1, 68)

        # Set correct error function
        error_fn = self.get_error_fn(error_type)

        # Get lists of all possible values for all free params
        valid_parameter_ranges: Dict[str, List[float]] = get_valid_param_ranges(precision)

        # Remove data on non-free params:
        ranges: List[List[Any]] = [valid_parameter_ranges[param] if param in self.free_params else [None] for param in ("xg", "xl", "g", "l", "tw")]

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
            # Keep fits where xl-.1 <= xg <= xl
            if (fit[0] is not None and fit[1] is not None) and (fit[0] > fit[1] or fit[0] < fit[1]-0.1):
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
            elif error == lowest_error and fit_params.a > best_fit.a:
                self.best_fits[subject] = best_fit

        self.best_fits[subject] = best_fit
        #self.all_best_fits[subject].append(best_fit)

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

    def predict_one_subject(self, subject: int, fit: Optional[Parameters] = None) -> List[int]:
        """Returns the predicted sale amounts for each day.
        Inputs:
            subject: the participant's number in the dataframe.
            fit: the parameters to be used to predict the subject's behavior.
                NOTE: The free_params used in the fit will depend on the exact model.
        """
        # Ensure fit is not None
        if not fit:
            raise ValueError("Expected Value based models need parameters!")

        # Get subject data
        subject_data: pd.DataFrame = self.get_data_one_subject(subject)

        # Create predictions list
        predictions: List[int] = []

        # Generate cutoff values for this fit
        cutoffs = self.generate_cutoffs(fit)

        # Turn cutoffs into a tuple
        # This is to make it hashable and cacheable, and therefore save compute time
        cutoffs_tuple: Tuple[int, ...] = tuple(cutoffs)

        # Iterate through each day, backwards:
        for day in trange(self.num_days, leave=False, desc="Predicting...", disable=True):
            # Get stored amount, since sell amount must be less than stored

            stored: int = int(subject_data.loc[str(day)]["stored"])
            price: int = int(subject_data.loc[str(day)]["price"])
            max_expected_value: float = 0
            best_sell_amount: int = 0

            # If the price < cutoff price, no chance of selling:
            cutoff_price = cutoffs[day]
            if price < cutoff_price:
                predictions.append(0)
                continue

            # Sell everything the last day
            if day == 0:
                predictions.append(stored)
                continue

            # Find the sell_amount that maximizes expected value
            for sell_amount in range(stored + 1):
                expected_value: float = 0
                if day == 1:  # second to last day
                    expected_value = self.expected_value_day_2(price, sell_amount, fit, cutoffs_tuple)
                elif day >= 2:  # all other days
                    expected_value = self.expected_value(day, price, sell_amount, fit, cutoffs_tuple)
                # Save the best value
                if expected_value > max_expected_value:
                    max_expected_value = expected_value
                    best_sell_amount = sell_amount
            predictions.append(best_sell_amount)

        return predictions
