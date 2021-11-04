import pt_predictor, save_model_as_spreadsheet
from utils import DATA_DIR

if __name__ == '__main__':
        version = "exhaustive_iter_full_111"

        pt_predictor.main(version=version)

        save_model_as_spreadsheet.spreadsheet_main(version=f'{DATA_DIR}/pt_{version}.pkl',
                                       filename=f'{DATA_DIR}/pt_predictions_{version}.xlsx',
                                       use_all_fits=False)
