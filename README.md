# FSU_DL_LOB_Trading

Deep learning for limit order book trading and mid-price movement. Some data files are too big to upload. If you want to run the code, you need to download the dataset first.

## There are two datasets in this project.

### 1, LobsterData

The data is on https://lobsterdata.com/info/DataSamples.php. Download the level 5 data of Amazon, Apple, Google, Intel, Microsoft.

### 2, "Benchmark" dataset

The data is on https://etsin.avointiede.fi/dataset/urn-nbn-fi-csc-kata20170601153214969115. Click the "Access this dataset freely." to download the dataset.

Unzip the the data file if needed. Put the files in the "data" folder without subfolders.


## There are three experiments in this project.

### 1, limit order book trading. 

We use two CNN models to predict bid ask spread cross. 

run main_trading.py

### 2, mid-pirce moving prediction on LobsterData.

run main_midprice.py

### 3, mid-pirce moving prediction on "Benchmark" dataset.

run main_benchmark.py
