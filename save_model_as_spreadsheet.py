import pickle as pkl
from openpyxl import Workbook
from openpyxl.styles import Font
from pt_predictor import *

def main(version: str, filename: str) -> None:

    # Load model
    with open(version, "rb") as f:
        model = pkl.load(f)

    workbook = Workbook()
    paramsheet = workbook.active
    paramsheet.title = "Parameters"

    predsheet = workbook.create_sheet("Predictions")

    # Add headers to first sheet
    paramsheet["A1"] = "Subject"
    for i, cell in enumerate(paramsheet.iter_cols(min_col = 2,
                                                  max_col = 6,
                                                  min_row = 1,
                                                  max_row = 1)):
        if i >= len(model.free_params):
            break
        cell[0].value = model.free_params[i]

    # Add headers to second sheet
    predsheet["A1"] = "Subject"
    for i, cell in enumerate(predsheet.iter_cols(min_col = 2,
                                                  max_col = 69,
                                                  min_row = 1,
                                                  max_row = 1)):
        cell[0].value = f'Day {i}'

    # Bold the first row of everything
    bold = Font(bold=True)
    for cell in paramsheet["1:1"]:
        cell.font = bold

    for cell in predsheet["1:1"]:
        cell.font = bold

    # Iterate through each patient, getting each best fit
    current_row = 2
    for subject in range(57):
        # Iterate through each best-fit
        for fit in model.all_best_fits[subject]:
            paramsheet[f'A{current_row}'] = subject
            predsheet[f'A{current_row}'] = subject
            # save only the free params
            for i, param in enumerate(fit.free_params):
                # Rounding just to remove floating point error
                paramsheet[current_row][i+1].value = str(round(getattr(fit,param), 4))
            # Get predictions for that fit
            predictions: List[int] = model.predict_one_subject(subject, fit)
            # Save predictions
            for i, cell in enumerate(predsheet.iter_cols(min_col = 2,
                                                         max_col = 69,
                                                         min_row = current_row,
                                                         max_row = current_row)):
                cell[0].value = predictions[i]
        current_row += 1

    workbook.save(filename=filename)

if __name__ == '__main__':

    filename = "data/pt_predictions.xlsx"
    version = "data/pt_exhaustive_0-05_930_prop.pkl"
    main(version=version, filename=filename)
