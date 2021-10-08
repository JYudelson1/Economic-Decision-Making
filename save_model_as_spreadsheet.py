import pickle as pkl
from openpyxl import Workbook
from openpyxl.styles import Font
from pt_predictor import *

def main(version: str, filename: str) -> None:

    # Load model
    with open(version, "rb") as f:
        model = pkl.load(f)

    workbook = Workbook()

    # Make sheets
    paramsheet = workbook.active
    paramsheet.title = "Parameters"
    predsheet = workbook.create_sheet("Predictions")
    soldsheet = workbook.create_sheet("Sold")
    storedsheet = workbook.create_sheet("Stored")
    errorsheet = workbook.create_sheet("Errors")

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

    # Add headers to third sheet
    soldsheet["A1"] = "Subject"
    for i, cell in enumerate(soldsheet.iter_cols(min_col = 2,
                                                  max_col = 69,
                                                  min_row = 1,
                                                  max_row = 1)):
        cell[0].value = f'Day {i}'

    # Add headers to fourth sheet
    storedsheet["A1"] = "Subject"
    for i, cell in enumerate(storedsheet.iter_cols(min_col = 2,
                                                  max_col = 69,
                                                  min_row = 1,
                                                  max_row = 1)):
        cell[0].value = f'Day {i}'

    # Add headers to fifth sheet
    errorsheet["A1"] = "Subject"
    errorsheet["B1"] = "Proportional Error"
    errorsheet["C1"] = "Percent Error"

    # Bold the first row of everything
    bold = Font(bold=True)
    for sheet in workbook.worksheets:
        for cell in sheet["1:1"]:
            cell.font = bold


    # Iterate through each patient, getting each best fit
    current_row = 2
    for subject in trange(57, desc="Saving..."):
        ## Save prediction data & best fit data
        # Iterate through each best-fit
        for fit in model.all_best_fits[subject]:
            paramsheet[f'A{current_row}'] = subject
            predsheet[f'A{current_row}'] = subject
            errorsheet[f'A{current_row}'] = subject
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
            # Save errors
            error: float = model.mean_error_one_subject_proportion(subject, predictions)
            errorsheet[f'B{current_row}'] = error
            errorsheet[f'C{current_row}'] = str(100 * error) + "%"

        # Save sold data
        soldsheet[f'A{current_row}'] = subject
        for i, cell in enumerate(soldsheet.iter_cols(min_col = 2,
                                                     max_col = 69,
                                                     min_row = current_row,
                                                     max_row = current_row)):
            cell[0].value = int(model.data.loc[subject]['sold'][i])

        # Save stored data
        storedsheet[f'A{current_row}'] = subject
        for i, cell in enumerate(storedsheet.iter_cols(min_col = 2,
                                                     max_col = 69,
                                                     min_row = current_row,
                                                     max_row = current_row)):
            cell[0].value = int(model.data.loc[subject]['stored'][i])

        current_row += 1

    # Save final mean error & std dev of error
    all_errors: List[float] = model.mean_error_all_subjects(verbose=False)
    errorsheet[f'A{current_row + 1}'] = "Mean Error:"
    errorsheet[f'B{current_row + 1}'] = np.mean(all_errors)
    errorsheet[f'A{current_row + 2}'] = "Mean Percent Error:"
    errorsheet[f'B{current_row + 2}'] = str(100 * np.mean(all_errors)) + "%"
    errorsheet[f'A{current_row + 3}'] = "Standard Deviation:"
    errorsheet[f'B{current_row + 3}'] = np.std(all_errors)

    # Adjust size of errorsheet to fit:
    cell_to_text = lambda x: str(x) if x else ""
    for col in errorsheet.columns:
        length = max(len(cell_to_text(cell)) for cell in col)
        errorsheet.column_dimensions[col[0].column_letter].width = length

    workbook.save(filename=filename)

if __name__ == '__main__':

    filename = "data/pt_predictions.xlsx"
    version = "data/pt_exhaustive_0-05_930_prop.pkl"
    main(version=version, filename=filename)
