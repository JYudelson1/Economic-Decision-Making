import pandas as pd
from math import ceil
pd.options.mode.chained_assignment = None  # default='warn'

prices_probabilities = pd.read_csv("data/prices_probabilities.csv")

##########

# Multiply the price times probabiltiy

prices_probabilities['price_times_prob'] = prices_probabilities['price_to_use'] * prices_probabilities['probability']


##########


# Get util_of_sale (which is sum of prob*price of given price and above)

prices_probabilities['util_of_sale'] = "" #creates column for util_of_sale

for row in prices_probabilities.index: #This loops through each row and sums up price_times_prob from the price of that row and above - the dataframe is already sorted to be ascending by price to ensure this
    prices_probabilities['util_of_sale'][row] = prices_probabilities['price_times_prob'].iloc[row:].sum(axis=0)


# Get prob_avail_sale (which is sum of probabilities below given price). The code is similar to the previous part.

prices_probabilities['prob_avail_sale'] = ""

for row in prices_probabilities.index:
    prices_probabilities['prob_avail_sale'][row] = prices_probabilities['probability'].iloc[0:row].sum(axis=0)


##########

daily_prices = pd.read_csv("data/prices_simple.csv")

# Create columns
daily_prices['cutoff'] = ""
daily_prices['util_of_hold'] = ""
daily_prices['util_of_sale'] = ""
daily_prices['prob_avail_sale'] = ""


# Get the cutoff price

for row in daily_prices.index:  #Loops through rows in daily_prices. Note that the dataframe is sorted in ascending order by day.

    if row == 0: #This established information for day 1, which is needed to calculate day 2's cutoff
        daily_prices['cutoff'][row] = 1
        daily_prices['util_of_sale'][row] = prices_probabilities['util_of_sale'][row]
        daily_prices['prob_avail_sale'][row] = prices_probabilities['prob_avail_sale'][row]
        daily_prices['util_of_hold'][row] = daily_prices['util_of_sale'][row]

    else:  #This loops through subsequent rows/days

        daily_prices['cutoff'][row] = ceil(daily_prices['util_of_hold'][row-1]) #This sets the cutoff price of that row/day, based on util_of_hold of previous day

        # Get numbers needed for later calculations
        cutoff_index =  daily_prices['cutoff'][row] - 1 #This variable serves as an index to lookup data in the prices_probabilties dataframe. The reason 1 is subtracted, is because the index for each row of the prices_probabilties dataframe is 1 less than the price of that row.
        daily_prices['util_of_sale'][row] = prices_probabilities['util_of_sale'][cutoff_index] # gets util_of_sale of the price of the cutoff
        daily_prices['prob_avail_sale'][row] = prices_probabilities['prob_avail_sale'][cutoff_index] # gets prob_avail_sale of the price of the cutoff

        # Create variables for next loop that determines util_of_hold
        util_of_hold = daily_prices['util_of_sale'][row] #A variable is created to store calculations that use util_of_hold
        prob = daily_prices['prob_avail_sale'][row] #A variable is created to store calculations that use prob_avail_sale
        i = row-1

        # Calculate util_of_hold
        while i>=0: #This loops through each row going backwards. So for example, if we're calculating utility of holding on day 5, this is using calculations from day 4,3,2 and 1 to get the value
            util_of_hold = util_of_hold + (daily_prices['util_of_sale'][i] * prob) #The variable util_of_hold gets updated each loop until it produces the final result
            prob = prob * daily_prices['prob_avail_sale'][i] #The variable prob also gets updated each loop
            i = i-1 #
        daily_prices['util_of_hold'][row] = util_of_hold #store determined util_of_hold for the day

print(daily_prices)
