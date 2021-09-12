import pandas as pd
from typing import Optional, Dict, List, Union, Any
from utils import *
from tqdm import tqdm, trange
from itertools import product

class Parameters():
    """A data class to hold values for free parameters"""

    def __init__(self, a = None,
                        b = None,
                        g = None,
                        l = None,
                        tw = None):
        self.a: float = a
        self.b: float = b
        self.g: float = g
        self.l: float = l
        self.tw: int = tw

        # Store a list of the free parameters
        self.free_params: List[str] = [param for param in ("a", "b", "g", "l", "tw") if getattr(self, param) != None]

class PredictionModel():
    """A model that uses the experiment data to generate predictions."""

    def __init__(self):
        """Initilizes the PredictionModel"""

        # Some models have free parameters. In that case, they vary from subject to subject
        self.best_fits: Dict[int, Optional[Parameters]] = {}

        # Get experiment data
        self.data: pd.DataFrame = get_full_data()
        self.num_subjects: int = 1 + self.data.index[-1][0]
        self.num_days: int = 1 + int(self.data.index[-1][1])

        # Set free paramaters based on model type
        # NOTE: This should be reimplemented for each individual model
        self.free_params: Optional[List[str]] = None

    def get_valid_param_ranges(self, precision: float = 0.001) -> Dict[str, List[float]]:
        """Returns a list of all the valid values for each parameter, given the precision.
        Note that all params ranges are returned, even if the parameter is not free.
        Inputs:
            precision: the amount to increment each value when iterating through all possible values."""
        valid_parameter_ranges: Dict[str, List[float]] = {
            "a": list(np.arange(0, 1 + precision, precision)),
            "b": list(np.arange(0, 1 + precision, precision)),
            "g": list(np.arange(0, 1 + precision, precision)),
            "l": list(np.arange(1, 3.5 + precision, precision)),
            "tw": list(np.arange(0, self.num_days + 1, 1))
        }
        return valid_parameter_ranges

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

    def finalize_and_mean_error(self, error_type: str = "proportional") -> float:
        """Evaluates the average mean error across all subjects,
        after setting the predictions in self.data to be the predictions given by the best fit for each subject.
        Inputs:
            error_type: should the error be calculated as the absolute difference
                        between the prediction and the amount, or as
                        the difference in proportion of goods sold. report.docx
                        seems to use proportional."""
        total_error: float = 0

        # Set correct error function
        if error_type == "proportional":
            error_fn = self.mean_error_one_subject_proportion
        elif error_type == "absolute":
            error_fn = self.mean_error_one_subject_absolute
        else:
            raise ValueError("Error type must be proportional or absolute!")

        for subject in range(self.num_subjects):
            # Get the already-determined best fit paramters
            best_fit: Optional[Parameters] = self.best_fits.get(subject)

            # Predict sale amounts based on best_fit
            predictions: List[int] = self.predict_one_subject(subject, best_fit)

            # Store predictions in self.data
            self.data.loc[subject, "prediction"] = predictions

            # Get error
            total_error += error_fn(subject, predictions)

        return (total_error / self.num_subjects)

    def stupid_fit_one_subject(self,
                                subject: int,
                                precision: float,
                                verbose: bool = False,
                                error_type: str = "proportional") -> None:
        """Performs the stupid fit algorithm for one subject and saves the best fit.
        The stupid fit algorithm consists of iterating through all possible parameter values (with a given level of precision) and accepting the best.
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
        if error_type == "proportional":
            error_fn = self.mean_error_one_subject_proportion
        elif error_type == "absolute":
            error_fn = self.mean_error_one_subject_absolute
        else:
            raise ValueError("Error type must be proportional or absolute!")

        # Get lists of all possible values for all free params
        valid_parameter_ranges: Dict[str, List[float]] = self.get_valid_param_ranges(precision)

        # Remove data on non-free params:
        ranges: List[List[Any]] = [valid_parameter_ranges[param] if param in self.free_params else [None] for param in ("a", "b", "g", "l", "tw") ]

        # Get all possible values via cartesian product
        all_possible_fits = product(*ranges)

        # Iterate through every possible value
        for fit in tqdm(all_possible_fits, disable=(not verbose), leave=False):
            # Predict sale amounts based on fit
            fit_params: Parameters = Parameters(*fit)
            predictions: List[int] = self.predict_one_subject(subject, fit_params)

            # Get error for the given fit
            error: float = error_fn(subject, predictions)

            # Check if it's the best so far:
            if error < lowest_error:
                lowest_error = error
                best_fit = fit

        self.best_fits[subject] = best_fit

    def stupid_fit(self, precision: float = 0.001, verbose: bool = False) -> None:
        """Does the stupid fit algorithm for all subjects. Modifies in place.
        Inputs:
            precision: the amount to increment each value when iterating through all possible values.
            verbose: set to True to get progress bars for the fitting."""
        for subject in trange(self.num_subjects, disable=(not verbose)):
            self.stupid_fit_one_subject(subject, precision, verbose)
