# Economic-Decision-Making

## Getting Started

To install requirements, go to the main directory and run
```
pip install -r requirements.txt
```

## File Directory:

Below are the files in the project, along with a few functions each contains. This may be a bit behind the latest updates, but it's a good idea to at least generally have an idea of what happens in each file.

- utils.py:
  - Contains various utility functions
    - get_full_data()
    - get_valid_param_ranges()
    - p()
  - Contains a couple of constants
- prediction_model.py:
  - Contains implementation of the Parameters class
    - Which just bundles together parameter values for ease of writing code, mainly
  - Implements PredictionModel class  
    - This is the most general model, and the base class for all others
- ev_based_model.py:
  - Implements EVModel class
    - This is a model which determines cutoff prices based on an explicit expected value function. I think this is the case for most of the models he mentions.
    - Implements a dummy expected_value() function
    - Implements a predict_one_subject() function
- eut_predictor.py
  - Implements EUTModel class
- pt_predictor.py
  - Implements PTModel class
  - Explicitly implements expected value function for prospect theory
  - NOTE: This is now a subclass of EVModel, not PredictionModel.


## Cutoff Information:

The following took me a while to realize, so I'm putting it here just in case. The cutoffs have to be recalculated for each fit. In particular, they depend on the expected value, as they are the smallest amount of goods to be sold on a given day so that the expected value is greater than zero.

However, note that the expected value also depends on the cutoffs. This seems circular, but isn't: it's iterative. There's a back and forth cycle wherein each function calls the other to extend the list of known cutoff prices one day more. The implementation of this is found in EVModel.generate_cutoffs().

## Remaining Questions:

1. Should error be absolute or proportional?
  - That is, should error be calcucated as the absolute difference between the predicted sale and the actual sale, or as a proportion of the amount they could have chosen to sell?
2. Implementation of time windows.
  - Most likely, this is a simple change to generate_cutoffs(). If I had to guess, just by mucking around with the ranges.

## Sample main function:

The following is the annotated main function for pt_predictor.py. I think it's a good base for the other models:

```
if __name__ == '__main__':

    # Error type can be "absolute" or "proportional"
    error_type = "absolute"

    # Initialize model
    model = PTModel()

    # Run stupid fitting
    # Note: for real fitting algorithm, precision should be .01
    # Run bfs fitting
    start_fit = Parameters(a=1.0, b=1.0, g=1.0, l=1.0)
    model.bfs_fit(verbose=True, precision=0.1, error_type=error_type, start_fit=start_fit)

    # Finalizes predictions

    mean_error = model.finalize_and_mean_error(error_type=error_type)
    std_deviation = model.std_dev_of_error(error_type=error_type)

    # Prints
    print(f'mean_error = {mean_error}')
    print(f'std_dev = {std_deviation}')
    print(model.data)

    # Saves best cutoff data
    model.save_cutoffs(f'{DATA_DIR}/prices_cutoff_pt.csv')
    with open(f'{DATA_DIR}/pt_all_data.pkl', "wb") as f:
        pkl.dump(model.data, f)

```
