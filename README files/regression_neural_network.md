# Regression-based neural networks: Predicting Hotel Lead Time

Keras is an API used for running high-level neural networks - the API is now included as the default one under TensorFlow 2.0, which was developed by Google.

The main competitor to Keras at this point in time is PyTorch, developed by Facebook. While PyTorch has a somewhat higher level of community support, it is a particularly verbose language and I personally prefer Keras for greater simplicity and ease of use in building and deploying models.

In this particular example, a neural network will be built in Keras to solve a regression problem, i.e. one where our dependent variable (y) is in interval format and we are trying to predict the quantity of y with as much accuracy as possible.

## What Is A Neural Network?

A neural network is a computational system that creates predictions based on existing data. Let us train and test a neural network using the neuralnet library in R.

A neural network consists of:

- Input layers: Layers that take inputs based on existing data
- Hidden layers: Layers that use backpropagation to optimise the weights of the input variables in order to improve the predictive power of the model
- Output layers: Output of predictions based on the data from the input and hidden layers

![neuralplot.png](https://github.com/MGCodesandStats/hotel-modelling/blob/master/images/neuralplot.png)

## Our Example

For this example, we use a linear activation function within the keras library to create a regression-based neural network. The purpose of this neural network is to predict a **lead time** value for each customer, which is the number of days between when the customer makes their booking and when they are meant to stay at the hotel. The lead time value applies irrespective of whether the customer ultimately cancels their hotel booking. The chosen features that form the input for this neural network are as follows:

- IsCanceled
- Country of origin
- Market segment
- Deposit type
- Customer type
- Required car parking spaces
- Arrival Date: Year
- Arrival Date: Month
- Arrival Date: Week Number
- Arrival Date: Day of Month

Firstly, the relevant libraries are imported. Note that you will need TensorFlow installed on your system to be able to execute the below code.

*Libraries*

```
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
```

As a neural network is being implemented, the variables need to be normalized for the network to interpret them properly. Therefore, the variables are transformed using the MaxMinScaler():

```
#Variables
y1=np.reshape(y1, (-1,1))
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
print(scaler_x.fit(x1))
xscale=scaler_x.transform(x1)
print(scaler_y.fit(y1))
yscale=scaler_y.transform(y1)
```

The data is then split into training and test data:

```
X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)
```

## Keras Model Configuration

Now, we train the neural network. We are using the five input variables (age, gender, miles, debt, and income), along with two hidden layers of 12 and 8 neurons respectively, and finally using the linear activation function to process the output.

```
model = Sequential()
model.add(Dense(12, input_dim=5, kernel_initializer='normal', activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()
```

The mean_squared_error (mse) and mean_absolute_error (mae) are our loss functions – i.e. an estimate of how accurate the neural network is in predicting the test data. We can see that with the validation_split set to 0.2, 80% of the training data is used to test the model, while the remaining 20% is used for testing purposes.

```
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
```

From the output, we can see that the more epochs are run, the lower our MSE and MAE become, indicating improvement in accuracy across each iteration of our model.
Neural Network Output

Let’s now fit our model.

```
>>> history=model.fit(X_train, y_train, epochs=30, batch_size=150,  verbose=1, validation_split=0.2)

Train on 24036 samples, validate on 6009 samples
Epoch 1/30
24036/24036 [==============================] - 0s 18us/sample - loss: 0.0145 - mse: 0.0145 - mae: 0.0949 - val_loss: 0.0126 - val_mse: 0.0126 - val_mae: 0.0865
Epoch 2/30
24036/24036 [==============================] - 0s 11us/sample - loss: 0.0116 - mse: 0.0116 - mae: 0.0835 - val_loss: 0.0118 - val_mse: 0.0118 - val_mae: 0.0873
Epoch 3/30
24036/24036 [==============================] - 0s 11us/sample - loss: 0.0109 - mse: 0.0109 - mae: 0.0801 - val_loss: 0.0110 - val_mse: 0.0110 - val_mae: 0.0815
Epoch 4/30
24036/24036 [==============================] - 0s 11us/sample - loss: 0.0106 - mse: 0.0106 - mae: 0.0788 - val_loss: 0.0109 - val_mse: 0.0109 - val_mae: 0.0823
Epoch 5/30
24036/24036 [==============================] - 0s 11us/sample - loss: 0.0106 - mse: 0.0106 - mae: 0.0785 - val_loss: 0.0106 - val_mse: 0.0106 - val_mae: 0.0791
...
Epoch 26/30
24036/24036 [==============================] - 0s 11us/sample - loss: 0.0095 - mse: 0.0095 - mae: 0.0722 - val_loss: 0.0100 - val_mse: 0.0100 - val_mae: 0.0742
Epoch 27/30
24036/24036 [==============================] - 0s 11us/sample - loss: 0.0095 - mse: 0.0095 - mae: 0.0721 - val_loss: 0.0100 - val_mse: 0.0100 - val_mae: 0.0760
Epoch 28/30
24036/24036 [==============================] - 0s 11us/sample - loss: 0.0094 - mse: 0.0094 - mae: 0.0719 - val_loss: 0.0097 - val_mse: 0.0097 - val_mae: 0.0733
Epoch 29/30
24036/24036 [==============================] - 0s 11us/sample - loss: 0.0094 - mse: 0.0094 - mae: 0.0718 - val_loss: 0.0097 - val_mse: 0.0097 - val_mae: 0.0724
Epoch 30/30
24036/24036 [==============================] - 0s 11us/sample - loss: 0.0093 - mse: 0.0093 - mae: 0.0717 - val_loss: 0.0100 - val_mse: 0.0100 - val_mae: 0.0730
```

Here, we can see that keras is calculating both the training loss and validation loss, i.e. the deviation between the predicted y and actual y as measured by the mean squared error.

As you can see, we have specified 30 epochs for our model. This means that we are essentially training our model over 30 forward and backward passes, with the expectation that our loss will decrease with each epoch, meaning that our model is predicting the value of y more accurately as we continue to train the model.

Let’s see what this looks like when we plot our respective losses:

```
print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
```

Both the training and validation loss decrease in an exponential fashion as the number of epochs is increased, suggesting that the model gains a high degree of accuracy as our epochs (or number of forward and backward passes) is increased.
Predictions

So, we’ve seen how we can train a neural network model, and then validate our training data against our test data in order to determine the accuracy of our model.

However, what if we now wish to use the model to estimate unseen data?


## Conclusion

In this tutorial, you have learned how to:

- Construct neural networks with Keras
- Scale data appropriately with MinMaxScaler
- Calculate training and test losses
- Make predictions using the neural network model

Many thanks for your time.
