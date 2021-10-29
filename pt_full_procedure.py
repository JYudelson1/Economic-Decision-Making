import pt_predictor, save_model_as_spreadsheet

if __name__ == '__main__':
        version = "v2_exhaustive_iter_full_1029"

        pt_predictor.main(version=version)

        save_model_as_spreadsheet.main(version=f'{DATA_DIR}/pt_{version}.pkl',
                                       filename=f'{DATA_DIR}/pt_predictions_{version}.xlsx')
