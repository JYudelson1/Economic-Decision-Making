import pandas as pd

DATA_DIR = "data"

def get_full_data() -> pd.DataFrame:
    """
    Gets full experiment data as DataFrame. Sorted first by subject #, then by day.
    Data includes: stored, sold, price
    """

    # Read .csv files
    daily_prices_raw = pd.read_csv(f'{DATA_DIR}/prices.csv')
    stored_raw = pd.read_csv(f'{DATA_DIR}/stored.csv')
    sold_raw = pd.read_csv(f'{DATA_DIR}/sold.csv')

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

def std_dev(error: float, n: int) -> float:
    # TODO: Implement this
    raise NotImplementedError

if __name__ == '__main__':

    # Check that the function works
    print(get_full_data().index[-1][0])
    print(len(get_full_data()))
