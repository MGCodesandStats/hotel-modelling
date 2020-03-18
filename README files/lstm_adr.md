# Forecasting ADR Trends For Hotels Using LSTM

Average Daily Rate (ADR) is recognised as one of the most important [metrics for hotels](https://roompricegenie.com/average-rate-adr/).

It is calculated as follows:

```
ADR = Revenue ÷ sold rooms
```

Essentially, ADR is measuring the average price of a hotel room over a given period.

## Background

The dataset under study is that of a Portuguese hotel which consists of the ADR for each individual booking as one of the included variables. Using time series analysis, let us assume that the hotel wishes to 1) calculate the average ADR value across all customers in a given week and 2) use this data to forecast future ADR trends.

A long-short term memory network (LSTM) is used to do this. LSTMs are sequential neural networks that assume dependence between the observations in a particular series. As such, they have increasingly come to be used for time series forecasting purposes.

For reference, ADR per customer is included - given that some customers are also companies as well as individuals, which results in more than one room per booking in many cases.

## Data Manipulation

Using pandas, the full date (year and week number) is joined with the corresponding ADR Value for each booking.

1_adr

These data points were then grouped together to obtain the average ADR per week across all bookings as follows:

```
df4 = df3.groupby('FullDate').agg("mean")
df4
df4.sort_values(['FullDate'], ascending=True)
```

Here is what the new dataframe looks like:

2_adr

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

3_adr

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
...
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
array([0.25961538, 0.39423077, 0.26442308, 0.35576923, 0.64423077,
       0.29807692, 0.82692308, 0.52403846, 0.37019231, 0.88461538,
       0.00961538, 0.38461538, 0.14423077, 0.14903846, 0.19230769,
...
       0.28846154, 0.20673077, 0.10576923, 0.33653846])
```

Here is a sample of the *X_train* array:

```
array([0.70858066, 0.75574219, 0.7348692 , 0.63555916, 0.34629856,
       0.32723163, 0.18514608, 0.21056117, 0.13243974, 0.1321469 ,
...
       0.12222412, 0.07324677, 0.05206859, 0.05937164, 0.04205497,
       0.0867528 , 0.10976084, 0.0236608 , 0.11987636])
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
59/59 - 1s - loss: 0.0996 - val_loss: 0.0030
Epoch 2/100
59/59 - 0s - loss: 0.0711 - val_loss: 0.0086
...
Epoch 99/100
59/59 - 0s - loss: 0.0071 - val_loss: 0.0033
Epoch 100/100
59/59 - 0s - loss: 0.0071 - val_loss: 0.0036
dict_keys(['loss', 'val_loss'])
```

This is a visual representation of the training and validation loss:

4_adr
