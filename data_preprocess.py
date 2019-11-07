# functions for data preprocessing

import numpy as np
from sklearn.preprocessing import StandardScaler

# turn the unevenly spaced time series into evenly spaced time series
def rescale_data(data_order, data_message, data_level, time_window=0.1):
    start_time = 34201
    end_time = 57599
    data = np.concatenate((data_message[:-1, 0].reshape([-1, 1]), data_order[1:, :]), axis=1)
    data_new = np.empty([int((end_time - start_time) / time_window) + 1, int(4 * data_level)])
    time_scale = np.arange(start_time, end_time, time_window)
    k = 0
    for i in range(time_scale.shape[0]):
        while k < data.shape[0] and data[k, 0] < time_scale[i]:
            k = k + 1
        
        data_new[i] = data[k - 1, 1:]

    n, m = data_new.shape
    for i in range(int(m / 2)):
        data_new[:, 2 * i] = data_new[:, 2 * i] / 10000

    return data_new

# generate (x, y) pair for cnn model. x: k by look back range by n. y: k by forecast size by n.
# n is 4 times level. k is the num_rows in the code.
def generate_data(data, forecast_size, look_back):
    m, n = data.shape
    num_rows = m - look_back - forecast_size
    data_x = np.empty([num_rows, look_back, n])
    data_y = np.empty([num_rows, forecast_size, n])

    for i in range(num_rows):
        data_x[i, :, :] = data[i: i + look_back, :]
        data_y[i, :, :] = data[i + look_back: i + look_back + forecast_size, :]

    return data_x, data_y
    

# split the data 1:1 for train and test
def train_test_split(data_x, data_y, forecast_size):
    n = data_x.shape[0]
    train_num = int(n * 0.5)
    train_x = data_x[:train_num]
    train_y = data_y[:train_num]
    test_x = data_x[train_num + forecast_size:]
    test_y = data_y[train_num + forecast_size:]
    return train_x, train_y, test_x, test_y
    
# generate matrix price, volumes and target probability
def data_for_trading_model(data_x, data_y, trading_type, profit_threshold): 
    # trading_type : long or short
    n = data_x.shape[0]
    price = np.empty_like(data_x[:,:,0::2])
    price_ask = data_x[:,:,0::4]
    price_bid = data_x[:,:,2::4]
    np.copyto(price, np.concatenate((price_bid[:,:,::-1], price_ask), axis=-1))
    volume = np.empty_like(data_x[:,:,1::2])
    volume_ask = data_x[:,:,1::4]
    volume_bid = data_x[:,:,3::4]
    np.copyto(volume, np.concatenate((volume_bid[:,:,::-1], volume_ask), axis=-1))
    volume = np.log(volume)
    if trading_type == 'long':
        profit = np.max(data_y[:, :, 2], axis=1) - data_x[:, -1, 0]
    else:
        profit = data_x[:, -1, 2] - np.min(data_y[:, :, 0], axis=1)
        
    prob = profit >= profit_threshold
    for i in range(n):
        price_mean = np.mean(price[i, :, :], axis=0)
        price[i, :, :] = price[i, :, :] - price_mean
        volume_mean = np.mean(volume[i, :, :], axis=0)
        volume[i, :, :] = volume[i, :, :] - volume_mean
    
    return price, volume, prob    

# generate matrix price, volumes and target probability
def data_for_mid_price_model(data_x, data_y):
    n = data_x.shape[0]
    price = np.empty_like(data_x[:,:,0::2])
    price_ask = data_x[:,:,0::4]
    price_bid = data_x[:,:,2::4]
    np.copyto(price, np.concatenate((price_bid[:,:,::-1], price_ask), axis=-1))
    volume = np.empty_like(data_x[:,:,1::2])
    volume_ask = data_x[:,:,1::4]
    volume_bid = data_x[:,:,3::4]
    np.copyto(volume, np.concatenate((volume_bid[:,:,::-1], volume_ask), axis=-1))
    volume = np.log(volume)
    
    midPrice = (price_ask[:,-1,0] + price_bid[:,-1,0])/2
    midAverage = np.average(data_y[:,:,0] + data_y[:,:,2], axis=1)/2
    alpha = 0.00002
    prob = np.zeros([n, 3])
    for i in range(n):
        if midPrice[i]*(1+alpha) < midAverage[i]:
            prob[i][0] = 1
        elif midPrice[i]*(1-alpha) > midAverage[i]:
            prob[i][1] = 1
        else:
            prob[i][2] = 1

    for i in range(n):
        price_mean = np.mean(price[i, :, :], axis=0)
        price[i, :, :] = price[i, :, :] - price_mean
        volume_mean = np.mean(volume[i, :, :], axis=0)
        volume[i, :, :] = volume[i, :, :] - volume_mean
    
    return price, volume, prob    

# load benchmark dataset and convert it back to raw limit order book dataset.
def read_benchmark_data(data_path, data_level):
    with open(data_path, 'r') as f:
        data = np.genfromtxt(f, max_rows=int(4*data_level))
    
    data = np.transpose(data)
    n = data.shape[1]
    for i in range(n):
        if i % 2 == 0:
            data[:, i] = data[:, i] * 100
        else:
            data[:, i] = data[:, i] * 1000000
        
    return data

# generate matrix price, volumes and target probability
def benchmark_data_for_model(data_x, data_y):    
    n = data_x.shape[0]
    price = np.empty_like(data_x[:,:,0::2])
    price_ask = data_x[:,:,0::4]
    price_bid = data_x[:,:,2::4]
    np.copyto(price, np.concatenate((price_bid[:,:,::-1], price_ask), axis=-1))
    volume = np.empty_like(data_x[:,:,1::2])
    volume_ask = data_x[:,:,1::4]
    volume_bid = data_x[:,:,3::4]
    np.copyto(volume, np.concatenate((volume_bid[:,:,::-1], volume_ask), axis=-1))
    volume = np.log(volume)
    midPrice = (price_ask[:,-1,0] + price_bid[:,-1,0])/2
    midAverage = np.average(data_y[:,:,0] + data_y[:,:,2], axis=1)/2
    alpha = 0.00002
    prob = np.zeros([n, 3])
    for i in range(n):
        if midPrice[i]*(1+alpha) < midAverage[i]:
            prob[i][0] = 1
        elif midPrice[i]*(1-alpha) > midAverage[i]:
            prob[i][1] = 1
        else:
            prob[i][2] = 1

    for i in range(n):
        price_mean = np.mean(price[i, :, :], axis=0)
        price[i, :, :] = price[i, :, :] - price_mean
        volume_mean = np.mean(volume[i, :, :], axis=0)
        volume[i, :, :] = volume[i, :, :] - volume_mean
    
    return price, volume, prob 
    