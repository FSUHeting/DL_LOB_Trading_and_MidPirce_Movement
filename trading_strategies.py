# This is the trading strategy. 
# we can trade after each predict > probability threshold.
# For each prediction, assume data is e1, e2, ..., e100, e101, e102, ..., e200.
# we use e1, e2, ..., e100 to make a prediction. The trading start at e101.

import numpy as np


def one_share_trade(current_price, future_price, trading_type, pred_prob, prob_threshold, profit_threshold):
    # trading_type : long or short
    m, forecast_size = future_price.shape  # future price is a matrix of future price for each e101, ..., e200.
    trading_index = 0 # first prediction
    profit = np.empty(m + forecast_size + 1) # profit history
    profit[0] = 0 # start at 0
    while trading_index < m:
        if pred_prob[trading_index] >= prob_threshold: # if probability is high enough
            position_open = True # open position
            holding_time = 0 # stock holding time <= 100
            for i in range(forecast_size): # from e101 to e200
                if ((trading_type == 'long' and future_price[trading_index, i] >= current_price[trading_index] + profit_threshold) or 
                (trading_type == 'short' and future_price[trading_index, i] + profit_threshold <= current_price[trading_index])):
                # if we can make a profit, close position.
                    position_open = False
                    holding_time = i + 1
                    break
            
            # if we cannot make a profit during e101 to e200, close position anyway.
            if position_open:
                position_open = False
                holding_time = forecast_size
            
            # update profit history 
            profit[trading_index + 1: trading_index + holding_time] = profit[trading_index]
            if trading_type == 'long':
                profit[trading_index + holding_time] = profit[trading_index] + future_price[trading_index, holding_time - 1] - current_price[trading_index]
            else:
                profit[trading_index + holding_time] = profit[trading_index] - future_price[trading_index, holding_time - 1] + current_price[trading_index]
            trading_index = trading_index + holding_time
        else: # probability is low so do nothing
            profit[trading_index + 1] = profit[trading_index]
            trading_index = trading_index + 1

    profit[trading_index + 1:] = profit[trading_index]
    return profit    
