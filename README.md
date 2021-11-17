# Economic Decision Making

## Getting Started

To install requirements, go to the main directory and run
```
pip install -r requirements.txt
```

The easiest way to run a model is to run one of the ```full_procedure``` python files, from the ```procedures``` directory.
Just change the version string in the procedure file, and that will automatically run the fitting and save the output in ```EconomicDecisionMaking/data```

## File Directory:

Below are the files in ```EconomicDecisionMaking```, along with a few functions each contains. This may be a bit behind the latest updates, but it's a good idea to at least generally have an idea of what happens in each file.

- ```utils.py```:
  - Contains various utility functions
    - ```get_full_data()```
    - ```get_valid_param_ranges()```
    - ```p()```
    - ```get_all_neighbors()```
    - ```prelec()```
  - Contains a couple of constants
  - Contains implementation of the Parameters class
    - Which bundles together parameter values for models
- ```models/prediction_model.py```:
  - Implements ```PredictionModel``` class  
    - This is the most general model, and the base class for all others
    - Includes many search functions for generating best fits, including:
      - Exhaustive search (all options)
      - **Iterative exhaustive search (narrowing in on the correct values)**
        - Currently in use.
      - Basinhopping (simulated annealing search)
      - BFS fit (Starting at a given fit, travels to all better neighbors)
      - Etc.
    - Includes functions for determining errors
- ```models/ev_based_model.py```:
  - Implements ```EVModel``` class
    - This is a model which determines cutoff prices based on an explicit expected value function. I think this is the case for most of the models mentioned in the report.
    - Implements a dummy ```expected_value()``` function
    - Implements a ```predict_one_subject()``` function
    - Implements a ```generate_cutoffs()``` function
    - NOTE: Any subclass of this should only need to implement an ```expected_value()``` and a ```main``` function.
- ```models/eut_predictor.py```
  - Implements ```EUTModel``` class
- ```models/pt_predictor.py```
  - Implements ```PTModel``` class
  - Explicitly implements expected value function for prospect theory
  - NOTE: This is now a subclass of ```EVModel```, not ```PredictionModel```.
- ```models/hpt_predictor.py```
    - Implements ```HPTTWModel``` class
    - Explicitly implements expected value function for hyperbolic prospect theory

## Cutoff Information:

The following took me a while to realize, so I'm putting it here just in case. The cutoffs have to be recalculated for each fit. In particular, they depend on the expected value, as they are the smallest amount of goods to be sold on a given day so that the expected value is greater than zero.

However, note that the expected value also depends on the cutoffs. This seems circular, but isn't: it's iterative. There's a back and forth cycle wherein each function calls the other to extend the list of known cutoff prices one day more. The implementation of this is found in EVModel.generate_cutoffs().

## Remaining Questions (**Solved**):

1. Should error be absolute or proportional?
  - That is, should error be calcucated as the absolute difference between the predicted sale and the actual sale, or as a proportion of the amount they could have chosen to sell?
  - **Answer:** Proportional.
2. Implementation of time windows.
  - Most likely, this is a simple change to generate_cutoffs().
  - **Answer:** After ```tw``` days, the cutoff stays the same throughout the time period.

## Sample main function:

The following is the annotated main function for ```pt_predictor.py```. I think it's a good base for the other models:

```
if __name__ == '__main__':

    # model name (to save to data dir)
    version = "exhaustive_iter_full_1029"

    # Initialize model
    model = PTModel()

    # Run fitting
    precisions = (0.1, 0.01, 0.001)
    model.iterative_exhaustive_search(precisions, verbose=True, start=True)

    # Finalizes predictions and prints
    mode.print_info()

    # Saves data
    with open(f'{DATA_DIR}/pt_{version}.pkl', "wb") as f:
        pkl.dump(model, f)

```

Note: ```iterative_exhaustive_search``` narrows in on the correct solution, first finding the best values with an increment of 0.1, then narrowing in on the correct value for the hundredths place, and finally the thousandths place. It's currently the best search algorithm we use.
