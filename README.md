[Home](https://mgcodesandstats.github.io/) |
[GitHub](https://github.com/mgcodesandstats) |
[Speaking Engagements](https://mgcodesandstats.github.io/speaking-engagements/) |
[Terms](https://mgcodesandstats.github.io/terms/) |
[E-mail](mailto:contact@michael-grogan.com)

# Predicting Hotel Cancellations and ADR with Machine Learning

The purpose of this project is to predict hotel cancellations and ADR (average daily rate) values for two separate Portuguese hotels (H1 and H2). Included in the GitHub repository are the datasets and notebooks for all models run.

The original datasets and research by Antonio et al. can be found here: [Hotel Booking Demand Datasets (2019)](https://www.sciencedirect.com/science/article/pii/S2352340918315191). All other relevant references have been cited in the below articles.

## Project Stages

### Stage 1: Data Manipulation and Feature Selection

- Used pandas to collate over 115,000 individual cancellation and ADR entries into a weekly time series format.

- Identified **lead time**, **country of origin**, **market segment**, **deposit type**, **customer type**, **required car parking spaces**, and **week of arrival** as the most important features in explaining the variation in hotel cancellations.

### Stage 2: Classification

- Trained [classification models](https://github.com/MGCodesandStats/hotel-modelling/blob/master/classification.ipynb) on the H1 dataset and tested against the H2 dataset. Used boto3 and botocore to import data from AWS S3 bucket to SageMaker.

- Used the **Explainable Boosting Classifier** by InterpretML, **KNN**, **Naive Bayes**, **Support Vector Machines**, and **XGBoost** to predict cancellations across the test set.

- SVM demonstrated the best performance overall with an f1-score accuracy of **71%**, and **66%** recall across the cancellation class.

- An ANN model was also trained in conjunction with dice_ml to identify [Diverse Counterfactual Explanations](https://github.com/MGCodesandStats/hotel-modelling/blob/master/interpretml-dice-ml.ipynb) for hotel bookings, i.e. changes in feature parameters that would cause a non-canceling customer to cancel, and vice versa.

### Stage 3: Regression

- Used regression modelling to predict **ADR (average daily rate)** across each customer.

- Trained regression models on the H1 dataset and tested against the H2 dataset.

- [Regression-based neural network with elu activation function](https://github.com/MGCodesandStats/hotel-modelling/blob/master/regression-nn-elu.ipynb) showed the best performance, with a mean absolute error of **28** compared to the mean ADR of **105** across the test set.

### Stage 4: Time Series

- Used ARIMA and LSTM models to forecast weekly ADR trends. 

- ARIMA demonstrated best results in forecasting ADR for [H1](https://github.com/MGCodesandStats/hotel-modelling/blob/master/timeseries-arima-adr-h1.ipynb) (RMSE of **10** relative to mean ADR of **160**) and [H2](https://github.com/MGCodesandStats/hotel-modelling/blob/master/timeseries-arima-adr-h2.ipynb) (RMSE of **8** relative to mean ADR of **131**).