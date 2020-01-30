# demandforecasting
### Forecast taxi demand for given areas in New York City 

Predictions are made based on a preprocessed dataset of the NYC Taxi and Limousine Commission (TLC) data (https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) containing hourly taxi demand from 01/2009 up to 06/2018 for all 263 taxi zones. 


Different forecasting methods are implemented:

- Baseline models (Random Walk & Canary Model)

- ARIMA model (statsmodels)

- Feedforward-NN (Keras)

- LSTM-NN (Keras)

- Autoencoder inspired by the work of "Deep and Confident Prediction for Time Series at Uber" by Zhu and Laptev (2017)
https://arxiv.org/pdf/1709.01907.pdf



## Details of models
Besides versions of the feedforward-NN and LSTM model which process data of a single area to predict future demand, "multivariate" models are provided. The "multivariate" models are implemented such that a window of past demand is processed to predict future demand for multiple areas at the same time. The multivariate models are tested with data of the 20 busiest taxi districts.

