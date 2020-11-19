[Home](https://mgcodesandstats.github.io/) |
[GitHub](https://github.com/mgcodesandstats) |
[Speaking Engagements](https://mgcodesandstats.github.io/speaking-engagements/) |
[Terms](https://mgcodesandstats.github.io/terms/) |
[E-mail](mailto:contact@michael-grogan.com)

# Predicting Hotel Cancellations with Machine Learning

The purpose of this project is to predict hotel cancellations and ADR (average daily rate) values for two separate Portuguese hotels (H1 and H2). Included in the GitHub repository are the datasets and notebooks for all models run.

The original datasets and research by Antonio et al. can be found here: [Hotel Booking Demand Datasets](https://www.sciencedirect.com/science/article/pii/S2352340918315191). All other relevant references have been cited in the below articles.

## Project Stages

**Stage 1: Data Manipulation and Feature Selection**

- Used pandas to collate individual cancellation entries into a weekly time series.

- Applied transformations where appropriate to allow for analysis of categorical features.

- Used the ExtraTreesClassifier and Forward and Backward Feature Selection to identify *lead time*, *country of origin*, *market segment*, *deposit type*, *customer type*, *required car parking spaces*, and *week of arrival* as the most important features in explaining the variation in hotel cancellations.

**Stage 2: Classification**

- Trained classification models on the H1 dataset and tested against the H2 dataset.

- Used the *Explainable Boosting Classifier* by InterpretML, *KNN*, *Naive Bayes*, *Support Vector Machines*, and *XGBoost* to predict cancellations across the test set.

- SVM demonstrated the best performance overall with an f1-score accuracy of 71%, and 66% recall across the cancellation class.

- An ANN model was also trained in conjunction with dice_ml to identify Diverse Counterfactual Explanations for hotel bookings, i.e. changes in features that would cause a non-cancelling customer to cancel, and vice versa.

**Stage 3: Regression**

- Used regression modelling to predict *ADR (average daily rate)* across each customer.

- Trained regression models on the H1 dataset and tested against the H2 dataset.

- Used *SVM models* and *regression-based neural networks* to predict ADR across the test set.

- The neural network with *elu* activation function across the input and hidden layers showed the best performance, with a mean absolute error of *28* compared to the mean ADR of *105* across the test set.

**Stage 4: Time Series**

Used an LSTM neural network to forecast weekly ADR and cancellation trends. Model demonstrated best results in forecasting ADR for H1 (RMSE of 31 relative to mean ADR of 160) and H2 (RMSE of 36 relative to mean ADR of 131).

## Notebooks and Articles

### Classification

#### Jupyter Notebooks

- [classification.ipynb](https://github.com/MGCodesandStats/hotel-cancellations/blob/master/classification.ipynb)

- [classification-xgboost-hotels.R](https://github.com/MGCodesandStats/hotel-cancellations/blob/master/classification-xgboost-hotels.R)

- [interpretml-classification.ipynb](https://github.com/MGCodesandStats/hotel-cancellations/blob/master/interpretml-classification.ipynb)

- [interpretml-dice-ml.ipynb](https://github.com/MGCodesandStats/hotel-cancellations/blob/master/interpretml-dice-ml.ipynb)

#### Articles

- [Building a Naive Bayes Classifier: Predicting Hotel Cancellations](https://towardsdatascience.com/building-a-naive-bayes-classifier-predicting-hotel-cancellations-31e3b8766614)

- [Classification of Hotel Cancellations Using KNN and SMOTE](https://towardsdatascience.com/classification-of-hotel-cancellations-using-knn-and-smote-3290cc87e74d)

- [DiCE: Diverse Counterfactual Explanations for Hotel Cancellations](https://towardsdatascience.com/dice-diverse-counterfactual-explanations-for-hotel-cancellations-762c311b2c64)

- [Feature Selection Techniques in Python: Predicting Hotel Cancellations](https://towardsdatascience.com/feature-selection-techniques-in-python-predicting-hotel-cancellations-48a77521ee4f)

- [Imbalanced Classes: Predicting Hotel Cancellations with Support Vector Machines](https://towardsdatascience.com/svms-random-forests-and-unbalanced-datasets-predicting-hotel-cancellations-2b983c2c5731)

- [Predicting Hotel Cancellations Using InterpretML](https://towardsdatascience.com/predicting-hotel-cancellations-using-interpretml-e4e64fefc7a8)

### Regression

#### Jupyter Notebooks

- [interpretml-regression-svm.ipynb](https://github.com/MGCodesandStats/hotel-cancellations/blob/master/interpretml-regression-svm.ipynb)

- [regression-nn-elu.ipynb](https://github.com/MGCodesandStats/hotel-cancellations/blob/master/regression-nn-elu.ipynb)

- [regression-nn-relu.ipynb](https://github.com/MGCodesandStats/hotel-cancellations/blob/master/regression-nn-relu.ipynb)

- [regression-svm.ipynb](https://github.com/MGCodesandStats/hotel-cancellations/blob/master/regression-svm.ipynb)

#### Articles

- [Regression-based neural networks: Predicting Average Daily Rates for Hotels](https://towardsdatascience.com/regression-based-neural-networks-with-tensorflow-v2-0-predicting-average-daily-rates-e20fffa7ac9a)

- [Support Vector Machines and Regression Analysis](https://towardsdatascience.com/support-vector-machines-and-regression-analysis-ad5d94ac857f)

### Spark

#### Jupyter Notebooks

- [spark-h1.ipynb](https://github.com/MGCodesandStats/hotel-cancellations/blob/master/spark-h1.ipynb)

- [spark-h2.ipynb](https://github.com/MGCodesandStats/hotel-cancellations/blob/master/spark-h2.ipynb)

#### Articles

- [Productionizing ML Projects with Google BigQuery and PySpark: Predicting Hotel Cancellations](https://towardsdatascience.com/productionising-ml-projects-with-google-bigquery-and-pyspark-predicting-hotel-cancellations-8bf94fdc4af)

### Time Series

#### Jupyter Notebooks

- [timeseries-cnn-cancellations-daily-h1.ipynb](https://github.com/MGCodesandStats/hotel-cancellations/blob/master/timeseries-cnn-cancellations-daily-h1.ipynb)

- [timeseries-interpretml-xgbregressor.ipynb](https://github.com/MGCodesandStats/hotel-cancellations/blob/master/timeseries-interpretml-xgbregressor.ipynb)

- [timeseries-lstm-adr-h1.ipynb](https://github.com/MGCodesandStats/hotel-cancellations/blob/master/timeseries-lstm-adr-h1.ipynb)

- [timeseries-lstm-adr-h2.ipynb](https://github.com/MGCodesandStats/hotel-cancellations/blob/master/timeseries-lstm-adr-h2.ipynb)

- [timeseries-lstm-cancellations-h1.ipynb](https://github.com/MGCodesandStats/hotel-cancellations/blob/master/timeseries-lstm-cancellations-h1.ipynb)

- [timeseries-lstm-cancellations-h2.ipynb](https://github.com/MGCodesandStats/hotel-cancellations/blob/master/timeseries-lstm-cancellations-h2.ipynb)

- [timeseries-xgbregressor-h1.ipynb](https://github.com/MGCodesandStats/hotel-cancellations/blob/master/timeseries-xgbregressor-h1.ipynb)

#### Articles

- [CNN-LSTM: Predicting Daily Hotel Cancellations](https://towardsdatascience.com/cnn-lstm-predicting-daily-hotel-cancellations-e1c75697f124)

- [One-Step Predictions with LSTM: Forecasting Hotel Revenues](https://towardsdatascience.com/one-step-predictions-with-lstm-forecasting-hotel-revenues-c9ef0d3ef2df)

- [Predicting Weekly Hotel Cancellations with XGBRegressor](https://towardsdatascience.com/predicting-weekly-hotel-cancellations-with-xgbregressor-d73eb74a8624)