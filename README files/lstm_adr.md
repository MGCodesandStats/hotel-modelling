## Forecasting ADR Trends For Hotels Using LSTM

Average Daily Rate (ADR) is recognised as one of the most important metrics for hotels.

It is calculated as follows:

```
ADR = Revenue ÷ sold rooms
```

Essentially, ADR is measuring the average price of a hotel room over a given period.

## Background

The dataset under study is that of a Portuguese hotel which consists of the ADR for each individual booking as one of the included variables. Using time series analysis, let us assume that the hotel wishes to 1) calculate the average ADR value across all customers in a given week and 2) use this data to forecast future ADR trends.

A long-short term memory network (LSTM) is used to do this.

## Data Manipulation
