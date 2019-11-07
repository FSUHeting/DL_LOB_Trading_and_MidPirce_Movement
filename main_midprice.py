# We modify our trading model to be a 3-classes classification model.
# Then we apply the model on the benchmark dataset to predict the mid-price movement.
# The data is time based.
# The data is on https://lobsterdata.com/info/DataSamples.php.


# choose not to use GPU. If you want to use GPU, you can remove the following three lines.
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
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

# turn data into evenly spaced. 
evenly_spaced_data = data_preprocess.rescale_data(data_order, data_message, data_level, time_window)
data_x, data_y = data_preprocess.generate_data(evenly_spaced_data, forecast_size, look_back)
del evenly_spaced_data

# split the one day dataset into two half day datasets.
train_x, train_y, test_x, test_y = data_preprocess.train_test_split(data_x, data_y, forecast_size)

batch_size = 256
learning_rate = 0.001

# generate data for training
train_price, train_volume, train_prob = data_preprocess.data_for_mid_price_model(train_x, train_y)
print('positive same ratio in train: ', np.mean(train_prob[:, 0]))
print('negative same ratio in train: ', np.mean(train_prob[:, 1]))
print('neutral same ratio in train: ', np.mean(train_prob[:, 2]))

# generate data for testing
test_price, test_volume, test_prob = data_preprocess.data_for_mid_price_model(test_x, test_y)   
print('positive same ratio in test: ', np.mean(test_prob[:, 0]))
print('negative same ratio in test: ', np.mean(test_prob[:, 1]))
print('neutral same ratio in test: ', np.mean(test_prob[:, 2]))

# train the deep learning model and predict 
cnn_model = deep_learning_models.cnn_classification_regular_mid_price_model(look_back, data_level, learning_rate)
cnn_model.fit([train_price, train_volume], train_prob, epochs=1, batch_size=batch_size, verbose=0)
pred_prob = cnn_model.predict([test_price, test_volume])

# calculate the accuracy, precision, recall, and f1 score.
y_true = np.argmax(test_prob, axis=1)
y_pred = np.argmax(pred_prob, axis=1)
acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')
print('f1: ', f1)
print('acc: ', acc)
print('precision: ', precision)
print('recall: ',recall)
