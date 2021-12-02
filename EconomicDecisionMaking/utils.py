## Adding package to PATH
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

## Imports
import pandas as pd #type: ignore
import numpy  as np #type: ignore
import pickle as pkl
from scipy.optimize import minimize, Bounds, basinhopping #type: ignore
from math           import exp, ceil
from typing         import Optional, Dict, List, Union, Any, Tuple, Generator, Callable, Iterable, Set
from tqdm           import tqdm, trange #type: ignore
from itertools      import product
from functools      import lru_cache
from warnings       import catch_warnings, simplefilter

## Constants
DATA_DIR   = os.path.dirname(os.path.abspath(__file__)) + "/data"
CACHE_SIZE = 10 ** 6
NUM_DAYS   = 68
TW_FACTOR  = NUM_DAYS
L_FACTOR   = 1
EPS        = np.finfo(float).eps
PARAM_LIST = ("a", "b", "xg", "xl", "g", "l", "tw")

## Utility Classes
class Parameters():
    """A data class to hold values for free parameters"""

    __slots__ = ['a', 'b', 'xg', 'xl', 'g', 'l', 'tw', 'free_params']

    def __init__(self, a   = None,
                        b  = None,
                        xg = None,
                        xl = None,
                        g  = None,
                        l  = None,
                        tw = None):
        self.a:  float = a
        self.b:  float = b
        self.xl: float = xl
        self.xg: float = xg
        self.g:  float = g
        self.l:  float = l
        self.tw: int   = tw

        # Store a list of the free parameters
        self.free_params: List[str] = [param for param in PARAM_LIST
                                                    if getattr(self, param) != None]

    def __eq__(self, other):
        """Check if two Parameter objects store equivalent info"""
        if not isinstance(other, Parameters):
            return False
        for param in PARAM_LIST:
            if getattr(self, param) != getattr(other, param):
                return False
        return True

    def __repr__(self):
        """String representation of Parameter object"""
        return f'Parameters(a={self.a},b={self.b},xg={self.xg},xl={self.xl},g={self.g},l={self.l},tw={self.tw})'

    def __hash__(self):
        """Make Parameters hashable, and allow identical params to hash to same locations"""
        return (self.a, self.b, self.xg, self.xl, self.g, self.l, self.tw).__hash__()

    def tuplify(self):
        """Return all info as a tuple"""
        return (self.a, self.b, self.xg, self.xl, self.g, self.l, self.tw)

    def deepcopy(self):
        """Returns a deepcopied Parameters object with the same values as self"""
        return Parameters(*self.tuplify())

## Utility functions
def get_full_data() -> pd.DataFrame:
    """
    Gets full experiment data as DataFrame. Sorted first by subject #, then by day.
    Data includes: stored, sold, price
    """

    # Read .csv files
    daily_prices_raw = pd.read_csv(f'{DATA_DIR}/prices.csv')
    stored_raw       = pd.read_csv(f'{DATA_DIR}/stored.csv')
    sold_raw         = pd.read_csv(f'{DATA_DIR}/sold.csv')

    # These 3 dataframes are structured the same when imported. This code changes the dataframe to a multilevel index dataframe, so that the first-level index is the participant, and the second-level index is the day

    stored = stored_raw.stack()
    stored = pd.DataFrame(stored)
    stored.rename(columns={0:'stored'}, inplace=True)
    #print(stored)

    sold = sold_raw.stack()
    sold = pd.DataFrame(sold)
    sold.rename(columns={0:'sold'}, inplace=True)
    #print(sold)

    daily_prices = daily_prices_raw.stack()
    daily_prices = pd.DataFrame(daily_prices)
    daily_prices.rename(columns={0:'price'}, inplace=True)
    #print(prices)

    ##########

    # This joins the dataframes restructured above into a dataframe called 'participants'
    participants = stored.join(sold, how="outer")
    participants = participants.join(daily_prices, how="outer")

    participants = participants.rename_axis(index=('participant','day')) #renames index levels

    return participants

def get_valid_param_ranges(precision: float = 0.001) -> Dict[str, List[float]]:
    """Returns a list of all the valid values for each parameter, given the precision.
    Note that all params ranges are returned, even if the parameter is not free.
    Inputs:
        precision: the amount to increment each value when iterating through all possible values."""
    valid_parameter_ranges: Dict[str, List[float]] = {
        "a":  list(np.arange(precision, 1 + EPS, precision)),
        "b":  list(np.arange(precision, 1 + EPS, precision)),
        "xg": list(np.arange(precision, 1 + EPS, precision)),
        "xl": list(np.arange(precision, 1 + EPS, precision)),
        "g":  list(np.arange(precision, 1 + EPS, precision)),
        "l":  list(np.arange(1, 2 + EPS, precision)),
        "tw": list(np.arange(2, NUM_DAYS, 1))
    }
    return valid_parameter_ranges

def get_all_neighbors(current: Optional[Parameters], precision: float) -> Any:
    """Given a Parameters object, returns all the neighbors within precision of it.
    NOTE: precision has no effect on TW, for which the neighbor will always be one above or below.
    NOTE: this returns a generator object from itertools.product, NOT a list!
    NOTE: this always includes the current node as well.
    Inputs:
        current: the Parameters object whose neighbors should be returned.
        precision: the amount to increment each value when traversing the search space"""

    # Ensure the starting point is real
    if current is None:
        return None

    ranges : List[List[Union[float, int]]] = []
    params: List[str] = current.free_params

    for param in ["a", "b", "xg", "xl", "g"]:
        if param in params:
            current_val = getattr(current, param)
            range: List[float] = [current_val]
            if current_val - precision > 0.0:
                range.append(current_val - precision)
            if current_val + precision <= 1.0:
                range.append(current_val + precision)
            ranges.append(range)
        else:
            ranges.append([None])

    if "l" in params:
        current_val = current.l
        range = [current_val]
        if current_val - precision >= 1:
            range.append(current_val - precision)
        if current_val + precision <= 2:
            range.append(current_val + precision)
        ranges.append(range)
    else:
        ranges.append([None])

    if "tw" in params:
        current_val = current.tw
        range = [current_val]
        if current_val - 1 >= 0:
            range.append(current_val - 1)
        if current_val + 1 <= NUM_DAYS - 1:
            range.append(current_val + 1)
        ranges.append(range)
    else:
        ranges.append([None])

    return product(*ranges)

# Get probabilities of each price
prices_probabilities: pd.DataFrame = pd.read_csv(f'{DATA_DIR}/prices_probabilities.csv')

@lru_cache(maxsize=CACHE_SIZE)
def p(price: int) -> float:
    """Gets the probability of a given price occuring.
    NOTE: This uses the normalized integer price, i.e. $1.5 -> 15
    Inputs:
        price: price as an int"""
    return prices_probabilities.loc[price - 1]["probability"]

@lru_cache(maxsize=CACHE_SIZE)
def prelec(p: float, g: float) -> float:
    """The subjective probability function.
    Inputs:
        p: the objective probability
        g: gamma, the overweighting constant. g >= 1"""
    # Note: seperated into two lines to avoid numpy warnings
    neg_log: float = -np.log(p + np.finfo(float).eps)
    return exp(-(neg_log ** g))

if __name__ == '__main__':

    # Check that the functions work
    print(get_full_data().index[-1][0])
    print(len(get_full_data()))
