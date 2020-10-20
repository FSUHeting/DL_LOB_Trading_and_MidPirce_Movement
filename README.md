# FSU_DL_LOB_Trading

Deep learning for limit order book trading and mid-price movement. Some data files are too big to upload. If you want to run the code, you need to download the dataset first.

## There are two datasets in this project.

### 1, LobsterData

The data is on https://lobsterdata.com/info/DataSamples.php. Download the level 5 data of Amazon, Apple, Google, Intel, Microsoft.

### 2, "Benchmark" dataset

The "Bechmark" dataset is from this paper: Benchmark dataset for mid-price forecasting of limit orderbook data with machine learning methods. Adamantios Ntakaris, Martin Magris, Juho Kanniainen, Moncef Gabbouj, Alexandros Iosifidis.(2018)

https://onlinelibrary.wiley.com/doi/epdf/10.1002/for.2543
 
The data is on https://etsin.avointiede.fi/dataset/urn-nbn-fi-csc-kata20170601153214969115. Click the "Access this dataset freely." to download the dataset. (sorry)

Unzip the the data file if needed. Put the files in the "data" folder without subfolders.


## There are three experiments in this project.

More details about the models can be found on the paper: Deep learning for Limit Order Book Trading and Mid-price Movement prediction.

### 1, limit order book trading. 

We use two CNN models to predict bid ask spread cross. One for long and one for short. 

run main_trading.py

### 2, mid-pirce moving prediction on LobsterData.

We use a CNN model to predict the mid-price movement. 

run main_midprice.py

### 3, mid-pirce moving prediction on "Benchmark" dataset.

We use a deep CNN model to predict the mid-price movement. 

run main_benchmark.py
