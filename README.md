[Home](https://mgcodesandstats.github.io/) |
[Medium](https://medium.com/@firstclassanalyticsmg) |
[LinkedIn](https://www.linkedin.com/in/michaeljgrogan/) |
[GitHub](https://github.com/mgcodesandstats) |
[Speaking Engagements](https://mgcodesandstats.github.io/speaking-engagements/) |
[Terms](https://mgcodesandstats.github.io/terms/) |
[E-mail](mailto:contact@michael-grogan.com)

# Predicting Hotel Cancellations with Machine Learning

The purpose of this project is to predict hotel cancellations for two separate hotels in Portugal, both on a classification and time series basis. Included in the GitHub repository are the datasets and notebooks for all models run. The Python version used is 3.6.5.

## Summary of Results

- SVM (Support Vector Machines) showed the highest overall F1 acccuracy score of **70%** in classifying hotel booking cancellations (both customers that cancel versus those that follow through with the booking). Recall (for booking cancellations) came in at **69%**.

- However, XGBoost demonstrated a recall of **94%** for booking cancellations (i.e. of all customers who cancelled their hotel booking, the model correctly identified 94% of these), while overall accuracy came in at **55%**.

- ARIMA showed a superior performance in forecasting weekly hotel cancellations for dataset H1, while LSTM showed superior performance for dataset H2.

- An LSTM model was used to forecast ADR (average daily rate) trends, while a regression-based neural network was used to predict lead time across customers. Findings are available in more detail below.

## Findings

Each individual article with relevant findings can be accessed as below:

### Feature Selection, Classification and Regression

- [Classification of Hotel Cancellations Using KNN and SMOTE](https://www.michael-grogan.com/hotel-modelling/articles/knn)

- [Feature Selection Methods](https://www.michael-grogan.com/hotel-modelling/articles/feature_selection)

- [Imbalanced Classes: Predicting Hotel Cancellations with Support Vector Machines](https://www.michael-grogan.com/hotel-modelling/articles/unbalanced_svm)

- [Regression-based neural networks with TensorFlow v2.0: Predicting Hotel Lead Time](https://www.michael-grogan.com/hotel-modelling/articles/regression_neural_network)

- [XGBoost and Classification](https://www.michael-grogan.com/hotel-modelling/articles/boosting)

### Time Series Forecasting

- [Forecasting Average Daily Rate Trends For Hotels Using LSTM](https://www.michael-grogan.com/hotel-modelling/articles/lstm_adr)

- [Predicting Weekly Hotel Cancellations with ARIMA](https://www.michael-grogan.com/hotel-modelling/articles/arima)

- [Predicting Weekly Hotel Cancellations with an LSTM Network](https://www.michael-grogan.com/hotel-modelling/articles/lstm_weeklycancellations)
