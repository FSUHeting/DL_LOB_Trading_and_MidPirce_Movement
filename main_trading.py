# This is the trading experiment. We will preprocess the data first. Then we will train two models. 
# This is a time based paper.
# One for long stock only and one for short stock only.
# The data is on https://lobsterdata.com/info/DataSamples.php. 


# choose not to use GPU. If you want to use GPU, you can remove the following three lines.
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import matplotlib.pyplot as plt
import data_preprocess
import deep_learning_models
import trading_strategies


# load the order and message book. We use the level 5 limit order book data.
file_path = 'data/'
data_name = 'AAPL_2012-06-21_34200000_57600000_'
data_level = 5
data_order = np.loadtxt(file_path + data_name + 'orderbook_' + str(data_level) + '.csv', delimiter=',')
data_message = np.loadtxt(file_path + data_name + 'message_' + str(data_level) + '.csv', delimiter=',')

# set time window, forecast size, and look back range.
time_window = 0.25
forecast_size = 100
look_back = 100

# turn data into evenly spaced, then split the one day dataset into two half day datasets.
evenly_spaced_data = data_preprocess.rescale_data(data_order, data_message, data_level, time_window)
data_x, data_y = data_preprocess.generate_data(evenly_spaced_data, forecast_size, look_back)
del evenly_spaced_data

# model for long only 
# set profit threshold and generate the data for training and testing.
profit_threshold_for_model = 0.03
train_x, train_y, test_x, test_y = data_preprocess.train_test_split(data_x, data_y, forecast_size)
train_price, train_volume, train_prob = data_preprocess.data_for_trading_model(train_x, train_y, 'long', profit_threshold_for_model)
test_price, test_volume, test_prob = data_preprocess.data_for_trading_model(test_x, test_y, 'long', profit_threshold_for_model)
print('positive same ratio in train: ', np.mean(train_prob))
print('positive same ratio in test: ', np.mean(test_prob))

# set batch size and learning rate. 
# train the model and predict the probability of making a profit by longing one share of stock
batch_size = 256
learning_rate = 0.001
cnn_model_for_long_1 = deep_learning_models.cnn_classification_trading_model(look_back, data_level, learning_rate)
cnn_model_for_long_1.fit([train_price, train_volume], train_prob, epochs=1, batch_size=batch_size,
                       validation_data=[[test_price, test_volume], test_prob])
pred_long_prob_1 = cnn_model_for_long_1.predict([test_price, test_volume]).flatten()
print("ps rate: ", np.mean(pred_long_prob_1))

# set the probability threshold and profit target.
# use the predicted probability to trade.
prob_threshold = 0.7
profit_threshold = 0.01
cnn_long_1_profit = trading_strategies.one_share_trade(test_x[:, -1, 0], test_y[:, :, 2], 'long', pred_long_prob_1, prob_threshold, profit_threshold)
del train_price, train_volume, train_prob, test_price, test_volume, test_prob

# model for short only 
# set profit threshold and generate the data for training and testing.
profit_threshold_for_model = 0.03
train_price, train_volume, train_prob = data_preprocess.data_for_trading_model(train_x, train_y, 'short', profit_threshold_for_model)
test_price, test_volume, test_prob = data_preprocess.data_for_trading_model(test_x, test_y, 'short', profit_threshold_for_model)
print('positive same ratio in train: ', np.mean(train_prob))
print('positive same ratio in test: ', np.mean(test_prob))

# set batch size and learning rate. 
# train the model and predict the probability of making a profit by shorting one share of stock
batch_size = 256
learning_rate = 0.001
cnn_model_for_short_1 = deep_learning_models.cnn_classification_trading_model(look_back, data_level, learning_rate)
cnn_model_for_short_1.fit([train_price, train_volume], train_prob, epochs=1, batch_size=batch_size,
                       validation_data=[[test_price, test_volume], test_prob])
pred_short_prob_1 = cnn_model_for_short_1.predict([test_price, test_volume]).flatten()
print("ps rate: ", np.mean(pred_short_prob_1))

# set the probability threshold and profit target.
# use the predicted probability to trade.
prob_threshold = 0.7
profit_threshold = 0.01
cnn_short_1_profit = trading_strategies.one_share_trade(test_x[:, -1, 2], test_y[:, :, 0], 'short', pred_short_prob_1, prob_threshold, profit_threshold)
del train_x, train_y, test_x, test_y, train_price, train_volume, train_prob, test_price, test_volume, test_prob

# plot trading profit 
f, axarr = plt.subplots(1,3)
axarr[0].plot(cnn_long_1_profit)
axarr[1].plot(cnn_short_1_profit)
axarr[2].plot(cnn_long_1_profit + cnn_short_1_profit)
plt.show()

# print the final profit
print('long profit: ', cnn_long_1_profit[-1])
print('short profit: ', cnn_short_1_profit[-1])
print('total profit: ', cnn_long_1_profit[-1] + cnn_short_1_profit[-1])
