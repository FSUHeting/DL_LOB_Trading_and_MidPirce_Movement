# We modify our trading model to be a 3-classes classification model.
# Then we apply the model on the benchmark dataset to predict the mid-price movement.
# The data is event based.
# The data is on https://etsin.avointiede.fi/dataset/urn-nbn-fi-csc-kata20170601153214969115. 
# Click the "Access this dataset freely." to download the dataset.


# choose not to use GPU. If you want to use GPU, you can remove the following three lines.
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import data_preprocess
import deep_learning_models

# choose the Decimal preprocessed data from the benchmark dataset. 
# We only use level 5 limit order book data.
# prepar the data for training.
file_path = 'data/'
data_name = '_Dst_Auction_DecPre_CF_1.txt'
data_level = 5
forecast_size = 10
look_back = 100
data_train = data_preprocess.read_benchmark_data(file_path + 'Train' + data_name, data_level)
data_x, data_y = data_preprocess.generate_data(data_train, forecast_size, look_back)
del data_train

# get inout and target data for training our deep learning model.
# check the distribution
train_price, train_volume, train_prob = data_preprocess.benchmark_data_for_model(data_x, data_y)
print('positive sample ratio in train: ', np.mean(train_prob[:, 0]))
print('negative sample ratio in train: ', np.mean(train_prob[:, 1]))
print('neutral sample ratio in train: ', np.mean(train_prob[:, 2]))
del data_x, data_y

# set batch size and learning rate. Train the deep learning model.
batch_size = 256
learning_rate = 0.001
cnn_model = deep_learning_models.cnn_classification_benchmark_mid_price_model(look_back, data_level, learning_rate)
cnn_model.fit([train_price, train_volume], train_prob, epochs=1, batch_size=batch_size)
del train_price, train_volume, train_prob

# get inout and target data for testing our deep learning model.
data_test = data_preprocess.read_benchmark_data(file_path + 'Test' + data_name, data_level)
data_x, data_y = data_preprocess.generate_data(data_test, forecast_size, look_back)
del data_test

# check the distribution
test_price, test_volume, test_prob = data_preprocess.benchmark_data_for_model(data_x, data_y)
print('positive sample ratio in train: ', np.mean(test_prob[:, 0]))
print('negative sample ratio in train: ', np.mean(test_prob[:, 1]))
print('neutral sample ratio in train: ', np.mean(test_prob[:, 2]))
del data_x, data_y

# predict the test data.
pred_prob = cnn_model.predict([test_price, test_volume])

# calculate the accuracy, precision, recall, and f1 score.
y_true = np.argmax(test_prob, axis=1)
y_pred = np.argmax(pred_prob, axis=1)
print('accuracy: ', accuracy_score(y_true, y_pred))
print('precision: ', precision_score(y_true, y_pred, average='macro'))
print('recall: ', recall_score(y_true, y_pred, average='macro'))
print('f1: ', f1_score(y_true, y_pred, average='macro'))
