[Home](https://mgcodesandstats.github.io/) |
[Medium](https://medium.com/@firstclassanalyticsmg) |
[LinkedIn](https://www.linkedin.com/in/michaeljgrogan/) |
[GitHub](https://github.com/mgcodesandstats) |
[Speaking Engagements](https://mgcodesandstats.github.io/speaking-engagements/) |
[Terms](https://mgcodesandstats.github.io/terms/) |
[E-mail](mailto:contact@michael-grogan.com)

# Forecasting Average Daily Rate Trends For Hotels Using LSTM

Here is how an LSTM model can be used to forecast the ADR (average daily rate) for hotels - a cornerstone metric within the industry.

Average Daily Rate (ADR) is recognised as one of the most important [metrics for hotels](https://roompricegenie.com/average-rate-adr/).

It is calculated as follows:

```
ADR = Revenue ÷ sold rooms
```

Essentially, ADR is measuring the average price of a hotel room over a given period.

## Background

The [dataset](https://www.researchgate.net/publication/309379684_Using_Data_Science_to_Predict_Hotel_Booking_Cancellations) under study consists of cancellation bookings for a Portuguese hotel which includes the ADR for each individual booking as one of the included variables.

Using time series analysis, let us assume that the hotel wishes to 1) calculate the average ADR value across all bookings for a given week and 2) use this data to forecast future weekly ADR trends - by weekly ADR we mean the average ADR across all bookings in any one week - hereafter referred to as "weekly ADR".

One will note in the dataset that there are numerous cancellation incidences with a positive ADR value — it is assumed in this case that even though the customer cancelled, they were still ultimately charged for the booking (e.g. cancelling past the cancellation deadline, etc).

A long-short term memory network (LSTM) is used to do this. LSTMs are sequential neural networks that assume dependence between the observations in a particular series. As such, they have increasingly come to be used for time series forecasting purposes.

For reference, ADR per customer is included - given that some customers are also companies as well as individuals, which results in more than one room per booking in many cases.

## Data Manipulation

Using pandas, the full date (year and week number) is joined with the corresponding ADR Value for each booking.

![1_adr](1_adr.png)

These data points were then grouped together to obtain the average ADR per week across all bookings as follows:

```
df4 = df3.groupby('FullDate').agg("mean")
df4
df4.sort_values(['FullDate'], ascending=True)
```

Here is what the new dataframe looks like:

![2_adr](2_adr.png)

As a side note, the full notebook and datasets are available at the link for the GitHub repository provided below, where the data manipulation procedures are illustrated in more detail.

A plot of the time series is generated:

```
import matplotlib.pyplot as plt
plt.plot(tseries)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.ylabel('ADR')
plt.title("Weekly ADR")
plt.show()
```

![3_adr](3_adr.png)

## LSTM Model Configuration

Let’s begin the analysis for the H1 dataset. The first 100 observations from the created time series is called. Then, a dataset matrix is created and the data is scaled.

```
df = df[:100]

# Form dataset matrix
def create_dataset(df, previous=1):
    dataX, dataY = [], []
    for i in range(len(df)-previous-1):
        a = df[i:(i+previous), 0]
        dataX.append(a)
        dataY.append(df[i + previous, 0])
    return np.array(dataX), np.array(dataY)
```

The data is then normalized with MinMaxScaler in order to allow the neural network to interpret it properly:

```
# normalize dataset with MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df = scaler.fit_transform(df)
df
```

Here is a sample of the output:

```
array([[0.35915778],
       [0.42256282],
       [0.53159902],
...
       [0.27125524],
       [0.26293747],
       [0.25547682]])
```

The data is partitioned into training and test sets, with the *previous* parameter set to 5:

```
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

# Training and Validation data partition
train_size = int(len(df) * 0.8)
val_size = len(df) - train_size
train, val = df[0:train_size,:], df[train_size:len(df),:]

# Number of previous
previous = 5
X_train, Y_train = create_dataset(train, previous)
X_val, Y_val = create_dataset(val, previous)
```

When the *previous* parameter is set to this, this essentially means that the value at time *t* (Y_train for the training data), is being predicted using the values *t-1*, *t-2*, *t-3*, *t-4*, and *t-5* (all under X_train).

Here is a sample of the *Y_train* array:

```
array([0.70858066, 0.75574219, 0.7348692 , 0.63555916, 0.34629856,
       0.32723163, 0.18514608, 0.21056117, 0.13243974, 0.1321469 ,
       0.06636683, 0.09516089, 0.02223529, 0.02497857, 0.06036494,
...
       0.12222412, 0.07324677, 0.05206859, 0.05937164, 0.04205497,
       0.0867528 , 0.10976084, 0.0236608 , 0.11987636])
```

Here is a sample of the *X_train* array:

```
array([[0.35915778, 0.42256282, 0.53159902, 0.6084246 , 0.63902841],
       [0.42256282, 0.53159902, 0.6084246 , 0.63902841, 0.70858066],
       [0.53159902, 0.6084246 , 0.63902841, 0.70858066, 0.75574219],
...
       [0.07324677, 0.05206859, 0.05937164, 0.04205497, 0.0867528 ],
       [0.05206859, 0.05937164, 0.04205497, 0.0867528 , 0.10976084],
       [0.05937164, 0.04205497, 0.0867528 , 0.10976084, 0.0236608 ]])
```       

100 epochs are run:

```
# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))

# Generate LSTM network
model = tf.keras.Sequential()
model.add(LSTM(4, input_shape=(1, previous)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history=model.fit(X_train, Y_train, validation_split=0.2, epochs=100, batch_size=1, verbose=2)
```

Here are some sample results:

```
Train on 59 samples, validate on 15 samples
Epoch 1/100
59/59 - 1s - loss: 0.0689 - val_loss: 0.0027
Epoch 2/100
59/59 - 0s - loss: 0.0431 - val_loss: 0.0118
...
Epoch 99/100
59/59 - 0s - loss: 0.0070 - val_loss: 0.0031
Epoch 100/100
59/59 - 0s - loss: 0.0071 - val_loss: 0.0034
dict_keys(['loss', 'val_loss'])
```

This is a visual representation of the training and validation loss:

![4_adr](4_adr.png)

## Training and Validation Predictions

Now, let’s generate some predictions.

```
# Generate predictions
trainpred = model.predict(X_train)
valpred = model.predict(X_val)
```

Here is a sample of training and test predictions:

**Training Predictions**

```
>>> trainpred
array([[0.6923234 ],
       [0.73979336],
       [0.75128263],
...
       [0.09547461],
       [0.11602292],
       [0.050261  ]], dtype=float32)
```

**Test Predictions**

```
>>> valpred

array([[0.06604623],
       [0.0982968 ],
       [0.10709635],
...
       [0.3344252 ],
       [0.2922875 ]], dtype=float32)
```

The predictions are converted back to normal values using ```scaler.inverse_transform```, and the training and validation scores are calculated.

```
import math
from sklearn.metrics import mean_squared_error

# calculate RMSE
trainScore = math.sqrt(mean_squared_error(Y_train[0], trainpred[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
valScore = math.sqrt(mean_squared_error(Y_val[0], valpred[:,0]))
print('Validation Score: %.2f RMSE' % (valScore))
```

**Training and Validation Scores**

```
Train Score: 12.71 RMSE
Validation Score: 8.83 RMSE
```

Here is a plot of the predictions:

![5_adr](5_adr.png)

The test and prediction arrays are reshaped accordingly, and the function for *mean directional accuracy* is defined:

```
import numpy as np

def mda(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Directional Accuracy """
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])).astype(int))
```

### Model Results

The mean directional accuracy is now calculated:

```
>>> mda(Y_val, predictions)
0.8571428571428571
```

An MDA of **86%** is obtained, meaning that the model correctly predicts the direction of the actual weekly ADR trends 86% of the time.

As seen above, a validation score of **8.83** RMSE was also obtained. RMSE is a measure of the deviation in weekly ADR from the actual values, and assumes the same numerical format as the same. The mean weekly ADR across the validation data was **69.99**.

The mean forecast error on the validation data came in at **-1.419**:

```
>>> forecast_error = (predictions-Y_val)
>>> forecast_error
>>> mean_forecast_error = np.mean(forecast_error)
>>> mean_forecast_error
-1.419167548625413
```

## Testing on unseen (test) data

Now that the model has been trained, the next step is to test the predictions of the model on unseen (or test data).

As previously explained, the value at time *t* is being predicted by LSTM using the values *t-1*, *t-2*, *t-3*, *t-4*, and *t-5*.

The last 15 weekly ADR values in the series are predicted in this case.

```
actual = tseries.iloc[100:115]
actual = np.array(actual)
actual
```

The previously built model is now used to predict each value using the previous five values in the time series:

```
# Test (unseen) predictions
# (t) and (t-5)
>>> XNew

array([[ 82.1267268 ,  90.48381679,  85.81940503,  84.46819121,
         83.25621451],
       [ 90.48381679,  85.81940503,  84.46819121,  83.25621451,
         84.12304147],
...
       [189.16831978, 198.22268542, 208.71251185, 211.52835052,
        211.16204036],
       [198.22268542, 208.71251185, 211.52835052, 211.16204036,
        210.28488251]])
```

The variables are scaled appropriately, and ```model.predict``` is invoked:

```
Xnew = scaler.transform(Xnew)
Xnew
Xnewformat = np.reshape(Xnew, (Xnew.shape[0], 1, Xnew.shape[1]))
ynew=model.predict(Xnewformat)
```

Here is an array of the generated predictions:

```
array([0.02153895, 0.0157201 , 0.12966183, 0.22085814, 0.26296526,
       0.33762595, 0.35830092, 0.54184073, 0.73585206, 0.8718423 ,
       0.92918825, 0.9334069 , 0.8861607 , 0.81483454, 0.76510745],
      dtype=float32)
```

The array is converted back to the original value format:

```
>>> ynew = ynew * np.abs(maxt-mint) + np.min(tseries)
>>> ynewpd=pd.Series(ynew)
>>> ynewpd

0      45.410988
1      44.423096
2      63.767456
3      79.250229
4      86.398926
5      99.074379
6     102.584457
7     133.744766
8     166.682877
9     189.770493
10    199.506348
11    200.222565
12    192.201385
13    180.092041
14    171.649673
dtype: float32
```

Here is the calculated **MDA**, **RMSE**, and **MFE (mean forecast error)**.

**MDA = 0.86**

```
>>> mda(actualpd, ynewpd)

0.8666666666666667
```

**RMSE = 33.77**

```
>>> mse = mean_squared_error(actualpd, ynewpd)
>>> rmse = sqrt(mse)
>>> print('RMSE: %f' % rmse)

RMSE: 33.775573
```

**MFE = -30.17**

```
>>> forecast_error = (ynewpd-actualpd)
>>> mean_forecast_error = np.mean(forecast_error)
>>> mean_forecast_error

-30.173496939933216
```

With the mean weekly ADR for the test set coming in at **160.49**, the RMSE and MFE performance do look reasonably strong (the lower the error, the better).

## H2 results

The same procedure was carried out on the H2 dataset (ADR data for a separate hotel in Portugal). Here are the results when comparing the predictions to the test set:

**MDA = 0.86**

```
>>> mda(actualpd, ynewpd)

0.8666666666666667
```

**RMSE = 38.15**

```
>>> mse = mean_squared_error(actualpd, ynewpd)
>>> rmse = sqrt(mse)
>>> print('RMSE: %f' % rmse)

RMSE: 38.155347
```

**MFE = -34.43**

```
>>> forecast_error = (ynewpd-actualpd)
>>> mean_forecast_error = np.mean(forecast_error)
>>> mean_forecast_error

-34.437111023457376
```

For the H2 dataset, the mean weekly ADR on the test set came in at **131.42**, with RMSE and MFE errors low by comparison.

## Conclusion

In this example, you have seen how ADR can be forecasted using an LSTM model. Specifically, the above examples have illustrated:

- How to construct an LSTM model
- Methods to gauge error and accuracy for LSTM model predictions
- Comparison of LSTM model performance vs ARIMA

The datasets and notebooks for this example are available at the [MGCodesandStats GitHub repository](https://github.com/MGCodesandStats/hotel-modelling), along with further research on this topic.
