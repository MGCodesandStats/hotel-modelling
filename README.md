[Home](https://mgcodesandstats.github.io/) |
[GitHub](https://github.com/mgcodesandstats) |
[Speaking Engagements](https://mgcodesandstats.github.io/speaking-engagements/) |
[Terms](https://mgcodesandstats.github.io/terms/) |
[E-mail](mailto:contact@michael-grogan.com)

# Predicting Hotel Cancellations with Machine Learning

The purpose of this project is to predict hotel cancellations for two separate hotels in Portugal (H1 and H2), both on a classification and time series basis. Included in the GitHub repository are the datasets and notebooks for all models run. The Python version used is 3.6.5.

The original datasets and research by Antonio et al. can be found here: [Using Data Science to Predict Hotel Booking Cancellations](https://www.researchgate.net/publication/309379684_Using_Data_Science_to_Predict_Hotel_Booking_Cancellations)

The classification models were built using data from the H1 dataset, with predictions then compared to the H2 dataset.

Time series forecasting was conducted on H1 and H2 independently.

## Findings

* Xgboost showed the best recall performance at 94%. In other words, the model correctly identified 94% of all customers who cancelled their hotel booking across the H2 dataset.

* LSTM showed strong results in predicting the weekly ADR value. The mean weekly ADR for H1 was 160.49 with an RMSE of 33.77. The mean weekly ADR for H2 was 131.42 with an RMSE of 38.15.

* ARIMA showed stronger performance in predicting weekly cancellations across H1, while LSTM showed stronger performance across the H2 dataset.

**H1 Results**

| Reading      | ARIMA | LSTM |
| ----------- | ----------- | ----------- |
| MDA      | 0.86       | 0.8       |
| RMSE   | 57.95        | 60.92       |
| MFE   | -12.72        | -51.62        |


**H2 Results**

| Reading      | ARIMA | LSTM |
| ----------- | ----------- | ----------- |
| MDA      | 0.86       | 0.8       |
| RMSE   | 274.08        | 109.53        |
| MFE   | 156.33        | 54.07        |

Each individual article with relevant findings can be accessed as below:

### Feature Selection and Classification

* [Classification of Hotel Cancellations Using KNN and SMOTE](https://www.michael-grogan.com/hotel-modelling/articles/knn)

* [Feature Selection Methods](https://www.michael-grogan.com/hotel-modelling/articles/feature_selection)

* [Imbalanced Classes: Predicting Hotel Cancellations with Support Vector Machines](https://www.michael-grogan.com/hotel-modelling/articles/unbalanced_svm)

* [XGBoost and Classification](https://www.michael-grogan.com/hotel-modelling/articles/boosting)

### Time Series Forecasting

* [Forecasting Average Daily Rate Trends For Hotels Using LSTM](https://www.michael-grogan.com/hotel-modelling/articles/lstm_adr)

* [Predicting Weekly Hotel Cancellations with ARIMA](https://www.michael-grogan.com/hotel-modelling/articles/arima)

* [Predicting Weekly Hotel Cancellations with an LSTM Network](https://www.michael-grogan.com/hotel-modelling/articles/lstm_weeklycancellations)
