[Home](https://mgcodesandstats.github.io/) |
[Portfolio](https://mgcodesandstats.github.io/portfolio/) |
[Speaking Engagements](https://mgcodesandstats.github.io/speaking-engagements/) |
[Terms and Conditions](https://mgcodesandstats.github.io/terms/) |
[E-mail me](mailto:contact@michaeljgrogan.com) |
[LinkedIn](https://www.linkedin.com/in/michaeljgrogan/)

# Predicting Hotel Cancellations with Support Vector Machines and ARIMA

Hotel cancellations can cause issues for many businesses in the industry. Not only is there the lost revenue as a result of the customer cancelling, but this can also cause difficulty in coordinating bookings and adjusting revenue management practices.

Data analytics can help to solve this issue, in terms of identifying the customers who are most likely to cancel – allowing a hotel chain to adjust its marketing strategy accordingly.

To investigate how machine learning can aid in this task, the **ExtraTreesClassifer**, **logistic regression**, and **support vector machine** models were employed in Python to determine whether cancellations can be accurately predicted with this model. For this example, both hotels are based in Portugal. The Algarve Hotel dataset available from [Science Direct](https://www.sciencedirect.com/science/article/pii/S2352340918315191) was used to train and validate the model, and then the logistic regression was used to generate predictions on a second dataset for a hotel in Lisbon.

## Data Processing

At the outset, there is the consideration of **overfitting** when building the model with the data.

For example, in the original H1 file, there were **11,122** cancellations while **28,938** bookings did not cancel. Therefore, non-cancellations could likely end up being overrepresented in the model. For this reason, the H1 dataset was filtered to include **10,000** cancellations and **10,000** non-cancellations.

For the test dataset (H2.csv), **12,000** observations were selected at random, irrespective of whether the booking was cancelled or not.

The relevant libraries were imported and the relevant data type for each variable was classified:

```
import os
import csv
import random
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler
```

As we can see, there are many variables that can potentially influence whether a customer is going to cancel or not, and not all of these variables will necessarily be relevant in determining this.

The data is imported:

```
train_df = pd.read_csv('H1.csv', dtype=dtypes)
a=train_df.head()
b=train_df
b.sort_values(['ArrivalDateYear','ArrivalDateWeekNumber'], ascending=True)
b
```

![table](https://github.com/MGCodesandStats/hotel-modelling/blob/master/images/table.png)

The variables are then stacked together under the numpy format:

```
# Dependent variable
IsCanceled = train_df['IsCanceled']
y = IsCanceled

# Numerical variables
leadtime = train_df['LeadTime'] #1
staysweekendnights = train_df['StaysInWeekendNights'] #2
staysweeknights = train_df['StaysInWeekNights'] #3
adults = train_df['Adults'] #4
children = train_df['Children'] #5
babies = train_df['Babies'] #6
isrepeatedguest = train_df['IsRepeatedGuest'] #11
previouscancellations = train_df['PreviousCancellations'] #12
previousbookingsnotcanceled = train_df['PreviousBookingsNotCanceled'] #13
bookingchanges = train_df['BookingChanges'] #16
agent = train_df['Agent'] #18
company = train_df['Company'] #19
dayswaitinglist = train_df['DaysInWaitingList'] #20
adr = train_df['ADR'] #22
rcps = train_df['RequiredCarParkingSpaces'] #23
totalsqr = train_df['TotalOfSpecialRequests'] #24

# Categorical variables
mealcat=train_df.Meal.astype("category").cat.codes
mealcat=pd.Series(mealcat)
countrycat=train_df.Country.astype("category").cat.codes
countrycat=pd.Series(countrycat)
marketsegmentcat=train_df.MarketSegment.astype("category").cat.codes
marketsegmentcat=pd.Series(marketsegmentcat)
distributionchannelcat=train_df.DistributionChannel.astype("category").cat.codes
distributionchannelcat=pd.Series(distributionchannelcat)
reservedroomtypecat=train_df.ReservedRoomType.astype("category").cat.codes
reservedroomtypecat=pd.Series(reservedroomtypecat)
assignedroomtypecat=train_df.AssignedRoomType.astype("category").cat.codes
assignedroomtypecat=pd.Series(assignedroomtypecat)
deposittypecat=train_df.DepositType.astype("category").cat.codes
deposittypecat=pd.Series(deposittypecat)
customertypecat=train_df.CustomerType.astype("category").cat.codes
customertypecat=pd.Series(customertypecat)
reservationstatuscat=train_df.ReservationStatus.astype("category").cat.codes
reservationstatuscat=pd.Series(reservationstatuscat)

# Numpy column stack
x = np.column_stack((leadtime,staysweekendnights,staysweeknights,adults,children,babies,mealcat,countrycat,marketsegmentcat,distributionchannelcat,isrepeatedguest,previouscancellations,previousbookingsnotcanceled,reservedroomtypecat,assignedroomtypecat,bookingchanges,deposittypecat,dayswaitinglist,customertypecat,adr,rcps,totalsqr,reservationstatuscat))
x = sm.add_constant(x, prepend=True)
```

## Extra Trees Classifier

As mentioned, there are many variables included in the analysis – not all of them will be relevant in determining whether a customer is likely to cancel or not.

To solve this issue, the Extra Trees Classifier is used to print the importance of each variable in numerical format, and the variables with the highest ranked importance are then chosen accordingly.

```
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x, y)
print(model.feature_importances_)
```

Here are the generated readings:

```
[0.         0.04070268 0.0052648  0.00701335 0.00396219 0.00806383
 0.00075091 0.00614394 0.05941394 0.03322725 0.01097485 0.01110851
 0.00733542 0.00147088 0.0076557  0.01338097 0.00640656 0.03391769
 0.0010779  0.018724   0.01788529 0.06105368 0.0082012  0.63626446]
```

Here is a breakdown of the feature ranking by order with the top features:

![feature-score](https://github.com/MGCodesandStats/hotel-modelling/blob/master/images/feature-score.png)

From the above, the top six identified features of importance are **reservation status**, **country**, **required car parking spaces**, **deposit type**, **customer type**, and **lead time**.

However, a couple of caveats worth mentioning:

- **Reservation status** cannot be used to predict hotel cancellations as the two are highly correlated, i.e. if the reservation status is cancelled, then this variable will already reflect this.

- When an initial logistic regression was run, **customer type** and **required car parking spaces** were shown as insignificant. Therefore, the regression was run again as below with these variables having been dropped from the model.

## Logistic Regression

The data was split into training and test data, and the logistic regression was generated:

```
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=0)

logreg = LogisticRegression().fit(x1_train,y1_train)
logreg

print("Training set score: {:.3f}".format(logreg.score(x1_train,y1_train)))
print("Test set score: {:.3f}".format(logreg.score(x1_test,y1_test)))
```
The following training and test set scores were generated:
```
Training set score: 0.699
Test set score: 0.699
```
Then, the coefficients for the logistic regression itself were generated:

```
import statsmodels.api as sm
logit_model=sm.Logit(y1,x1)
result=logit_model.fit()
print(result.summary())
```

Here are the updated results:

```
Optimization terminated successfully.
         Current function value: 0.596755
         Iterations 7
                           Logit Regression Results                           
==============================================================================
Dep. Variable:             IsCanceled   No. Observations:                20000
Model:                          Logit   Df Residuals:                    19996
Method:                           MLE   Df Model:                            3
Date:                Sat, 17 Aug 2019   Pseudo R-squ.:                  0.1391
Time:                        23:58:55   Log-Likelihood:                -11935.
converged:                       True   LL-Null:                       -13863.
                                        LLR p-value:                     0.000
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const         -2.1417      0.050    -43.216      0.000      -2.239      -2.045
x1             0.0055      0.000     32.013      0.000       0.005       0.006
x2             0.0236      0.001     36.465      0.000       0.022       0.025
x3             2.1137      0.104     20.400      0.000       1.911       2.317
==============================================================================
```

Now, the logistic regression is used to predict cancellations for the test data, and a confusion matrix is generated to determine the incidence of true/false positives and negatives:

```
pr = logreg.predict(x1_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y1_test,pr))
print(classification_report(y1_test,pr))
```

The confusion matrix is generated:

```
[[1898  633]
 [ 874 1595]]
              precision    recall  f1-score   support

           0       0.68      0.75      0.72      2531
           1       0.72      0.65      0.68      2469

    accuracy                           0.70      5000
   macro avg       0.70      0.70      0.70      5000
weighted avg       0.70      0.70      0.70      5000
```

Here is an ROC curve illustrating the true vs false positive rate.

```
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
falsepos,truepos,thresholds=roc_curve(y1_test,logreg.decision_function(x1_test))
plt.plot(falsepos,truepos,label="ROC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
cutoff=np.argmin(np.abs(thresholds))
plt.plot(falsepos[cutoff],truepos[cutoff],'o',markersize=10,label="cutoff",fillstyle="none")
plt.show()
```

![roccurve1](https://github.com/MGCodesandStats/hotel-modelling/blob/master/images/roccurve1.png)

## Support Vector Machine (SVM) generation

The above model has shown a **70%** classification accuracy in determining whether a customer will cancel. The prediction for non-cancellations was **68%** based on precision while it was **72%** for cancellations (also based on precision).

Therefore, an SVM was generated using the training and validation data to determine whether this model would yield higher classification accuracy.

```
from sklearn import svm
clf = svm.SVC(gamma='scale')
clf.fit(x1, y1)  
prclf = clf.predict(x1_test)
prclf
```

A new prediction array is generated:

```
array([1, 0, 0, ..., 1, 1, 0])
```

Here is the new ROC curve generated:

![roccurve2](https://github.com/MGCodesandStats/hotel-modelling/blob/master/images/roccurve2.png)

This is the updated confusion matrix:

```
[[2085  446]
 [ 963 1506]]
              precision    recall  f1-score   support

           0       0.68      0.82      0.75      2531
           1       0.77      0.61      0.68      2469

    accuracy                           0.72      5000
   macro avg       0.73      0.72      0.71      5000
weighted avg       0.73      0.72      0.71      5000
```

The overall accuracy has increased to **73%**, but note that the predictive accuracy for cancellations specifically has improved quite significantly to **77%**, while it remains at **68%** for non-cancellations.

## Testing against unseen data

Now that the SVM has shown improved accuracy against the validation dataset, another dataset H2.csv (also available from Science direct) is used for comparison purposes, i.e. the SVM generated using the last dataset is now used to predict classifications across this dataset (for a different hotel located in Lisbon, Portugal).

The second dataset is loaded using pandas:

```
h2data = pd.read_csv('H2.csv', dtype=dtypes)
a=h2data.head()
b=h2data
b
```

The new variables are sorted into a numpy column stack, and an SVM is run:

```
# Numerical variables
t_leadtime = h2data['LeadTime'] #1
t_staysweekendnights = h2data['StaysInWeekendNights'] #2
t_staysweeknights = h2data['StaysInWeekNights'] #3
t_adults = h2data['Adults'] #4
t_children = h2data['Children'] #5
t_babies = h2data['Babies'] #6
t_isrepeatedguest = h2data['IsRepeatedGuest'] #11
t_previouscancellations = h2data['PreviousCancellations'] #12
t_previousbookingsnotcanceled = h2data['PreviousBookingsNotCanceled'] #13
t_bookingchanges = h2data['BookingChanges'] #16
t_agent = h2data['Agent'] #18
t_company = h2data['Company'] #19
t_dayswaitinglist = h2data['DaysInWaitingList'] #20
t_adr = h2data['ADR'] #22
t_rcps = h2data['RequiredCarParkingSpaces'] #23
t_totalsqr = h2data['TotalOfSpecialRequests'] #24

# Categorical variables
t_mealcat=h2data.Meal.astype("category").cat.codes
t_mealcat=pd.Series(t_mealcat)
t_countrycat=h2data.Country.astype("category").cat.codes
t_countrycat=pd.Series(t_countrycat)
t_marketsegmentcat=h2data.MarketSegment.astype("category").cat.codes
t_marketsegmentcat=pd.Series(t_marketsegmentcat)
t_distributionchannelcat=h2data.DistributionChannel.astype("category").cat.codes
t_distributionchannelcat=pd.Series(t_distributionchannelcat)
t_reservedroomtypecat=h2data.ReservedRoomType.astype("category").cat.codes
t_reservedroomtypecat=pd.Series(t_reservedroomtypecat)
t_assignedroomtypecat=h2data.AssignedRoomType.astype("category").cat.codes
t_assignedroomtypecat=pd.Series(t_assignedroomtypecat)
t_deposittypecat=h2data.DepositType.astype("category").cat.codes
t_deposittypecat=pd.Series(t_deposittypecat)
t_customertypecat=h2data.CustomerType.astype("category").cat.codes
t_customertypecat=pd.Series(t_customertypecat)
t_reservationstatuscat=h2data.ReservationStatus.astype("category").cat.codes
t_reservationstatuscat=pd.Series(t_reservationstatuscat)

a = np.column_stack((t_leadtime,t_countrycat,t_deposittypecat))
a = sm.add_constant(a, prepend=True)
IsCanceled = h2data['IsCanceled']
b = IsCanceled
b=b.values

prh2 = clf.predict(a)
prh2
```

The array of predictions is generated once again:

```
array([0, 0, 1, ..., 0, 1, 0])
```

A classification matrix is generated:

```
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(b,prh2))
print(classification_report(b,prh2))
```

**Classification Output**

```
[[5654 1350]
 [2038 2958]]
              precision    recall  f1-score   support

           0       0.74      0.81      0.77      7004
           1       0.69      0.59      0.64      4996

    accuracy                           0.72     12000
   macro avg       0.71      0.70      0.70     12000
weighted avg       0.71      0.72      0.71     12000
```

The ROC curve is generated:

![roccurve3](https://github.com/MGCodesandStats/hotel-modelling/blob/master/images/roccurve3.png)

Across the test set, the overall prediction accuracy increased to **72%**, while the accuracy for cancellation incidences fell slightly to **69%**.

```
metrics.auc(falsepos, truepos)
```
The computed AUC (area under the curve) is **0.74**.

```
0.7437825188763232
```

# ARIMA Modelling of Weekly Hotel Cancellations

Having investigated the main drivers of hotel cancellations, it is useful to determine whether hotel cancellations can also be predicted in advance. This will be done using the Algarve Hotel dataset in the first instance (H1full.csv). Since we are now seeking to predict the time series trend, all observations are now included in this dataset (cancellations and non-cancellations, irrespective of whether the dataset as a whole is uneven).

To do this, cancellations are analysed on a weekly basis (i.e. the number of cancellations for a given week are summed up).

Firstly, data manipulation procedures were carried out using pandas to sum up the number of cancellations per week and order them correctly.

In configuring the ARIMA model, the first 80 observations are used as **training data**, with the following 20 then used as **validation data**.

Once the model has been configured, the last 15 observations are then used as **test data** to gauge the model accuracy on unseen data.

Here is a snippet of the output:

![cancellationweeks](https://github.com/MGCodesandStats/hotel-modelling/blob/master/images/cancellationweeks.png)

The time series is visualised, and the autocorrelation and partial autocorrelation plots are generated:

**Time Series**

![time-series](https://github.com/MGCodesandStats/hotel-modelling/blob/master/images/time-series.png)

**Autocorrelation**

![autocorrelation](https://github.com/MGCodesandStats/hotel-modelling/blob/master/images/autocorrelation.png)

**Partial Autocorrelation**

![partial-autocorrelation](https://github.com/MGCodesandStats/hotel-modelling/blob/master/images/partial-autocorrelation.png)

```
#Dickey-Fuller Test
result = ts.adfuller(train)
result
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
```

When a Dickey-Fuller test is run, a p-value of less than 0.05 is generated, indicating that the null hypothesis of non-stationarity is rejected (i.e. the data is stationary).

```
ADF Statistic: -2.677149
p-value: 0.078077
Critical Values:
	1%: -3.519
	5%: -2.900
	10%: -2.587
```

An ARIMA model is then run using auto_arima from the **pyramid** library. This is used to select the optimal (p,d,q) coordinates for the ARIMA model.

```
from pyramid.arima import auto_arima
Arima_model=auto_arima(train, start_p=0, start_q=0, max_p=10, max_q=10, start_P=0, start_Q=0, max_P=10, max_Q=10, m=52, seasonal=True, trace=True, d=1, D=1, error_action='warn', suppress_warnings=True, random_state = 20, n_fits=30)
```

The following output is generated:

```
Fit ARIMA: order=(0, 1, 0) seasonal_order=(0, 1, 0, 52); AIC=305.146, BIC=307.662, Fit time=0.139 seconds
Fit ARIMA: order=(1, 1, 0) seasonal_order=(1, 1, 0, 52); AIC=nan, BIC=nan, Fit time=nan seconds
Fit ARIMA: order=(0, 1, 1) seasonal_order=(0, 1, 1, 52); AIC=nan, BIC=nan, Fit time=nan seconds
Fit ARIMA: order=(0, 1, 0) seasonal_order=(1, 1, 0, 52); AIC=nan, BIC=nan, Fit time=nan seconds
Fit ARIMA: order=(0, 1, 0) seasonal_order=(0, 1, 1, 52); AIC=nan, BIC=nan, Fit time=nan seconds
Fit ARIMA: order=(0, 1, 0) seasonal_order=(1, 1, 1, 52); AIC=nan, BIC=nan, Fit time=nan seconds
Fit ARIMA: order=(1, 1, 0) seasonal_order=(0, 1, 0, 52); AIC=292.219, BIC=295.993, Fit time=0.590 seconds
Fit ARIMA: order=(1, 1, 1) seasonal_order=(0, 1, 0, 52); AIC=293.486, BIC=298.518, Fit time=0.587 seconds
Fit ARIMA: order=(2, 1, 1) seasonal_order=(0, 1, 0, 52); AIC=294.780, BIC=301.070, Fit time=1.319 seconds
Fit ARIMA: order=(1, 1, 0) seasonal_order=(0, 1, 1, 52); AIC=nan, BIC=nan, Fit time=nan seconds
Fit ARIMA: order=(1, 1, 0) seasonal_order=(1, 1, 1, 52); AIC=nan, BIC=nan, Fit time=nan seconds
Fit ARIMA: order=(2, 1, 0) seasonal_order=(0, 1, 0, 52); AIC=293.144, BIC=298.176, Fit time=0.896 seconds
Total fit time: 3.549 seconds
```

Based on the lowest AIC, the **SARIMAX(1, 1, 0)x(0, 1, 0, 52)** configuration is identified as the most optimal for modelling the time series.

Here is the output of the model:

![arima-model](https://github.com/MGCodesandStats/hotel-modelling/blob/master/images/arima-model.PNG)

With **90%** of the series used as the training data to build the ARIMA model, the remaining **10%** is now used to test the predictions of the model. Here are the predictions vs the actual data:

![validation-vs-predicted](https://github.com/MGCodesandStats/hotel-modelling/blob/master/images/validation-vs-predicted.png)

We can see that while the prediction values were lower than the actual test values, the direction of the two series seem to be following each other.

From a business standpoint, a hotel is likely more interested in predicting whether the degree of cancellations will increase/decrease in a particular week - as opposed to the precise number of cancellations - which will no doubt be more subject to error and influenced by extraneous factors.

In this regard, the [mean directional accuracy](https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9) is used to determine the degree to which the model accurately forecasts the directional changes in cancellation frequency from week to week.

```
def mda(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Directional Accuracy """
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])).astype(int))
```

An MDA of 89% is yielded:

```
>>> mda(val, predictions)

0.8947368421052632
```

In this regard, the ARIMA model has shown a reasonably high degree of accuracy in predicting directional changes for hotel cancellations across the test set.

The RMSE (root mean square error) is also predicted:

```
>>> import math
>>> from sklearn.metrics import mean_squared_error

>>> mse = mean_squared_error(val, predictions)
>>> rmse = math.sqrt(mse)
>>> print('RMSE: %f' % rmse)

RMSE: 77.047252
```

The RMSE stands at 77 in this case. Note that the units of RMSE are the same as the response variable, in this case - hotel cancellations. With an average cancellation of 94 for all weeks across the validation data, the RMSE of 77 is technically the standard deviation of the unexplained variance. All else being equal, the lower this value, the better.

## Testing against unseen data

Even though the ARIMA model has been trained and the accuracy validated across the validation data, it is still unclear how the model would perform against unseen data (or test data).

In this regard, the ARIMA model is used to generate predictions for n=15 using the test.index to specify the unseen data.

```
>>> test = np.array([[130,202,117,152,131,161,131,139,150,157,173,140,182,143,100]])
```

Firstly, the array is reshaped accordingly:

```
>>> test=test.reshape(-1)
>>> test

array([130, 202, 117, 152, 131, 161, 131, 139, 150, 157, 173, 140, 182,
       143, 100])
```

Now, the predictions are made, and the RMSE (root mean squared error), MDA (mean directional accuracy) and mean forecast errors are calculated:

```
>>> predictionnew=pd.DataFrame(Arima_model.predict(n_periods=15), index=test.index)
>>> predictionnew.columns = ['Unseen_Predicted_Cancellations']
>>> predictionsnew=predictionnew['Unseen_Predicted_Cancellations']

>>> mse_new = mean_squared_error(test, predictionsnew)
>>> rmse_new = math.sqrt(mse_new)
>>> print('RMSE: %f' % rmse_new)

RMSE: 57.955865

>>> mda(test, predictionsnew)

0.8666666666666667

>>> forecast_error_new = (predictionsnew-test)
>>> forecast_error_new

0     -39.903941
1    -128.986739
2     -47.325146
3     -76.683169
4     -14.237713
5      77.591519
6     -34.782635
7      59.277972
8       4.404317
9     -40.860982
10    -38.522419
11     49.074094
12    -44.497360
13     11.040560
14     73.507259
dtype: float64

>>> mean_forecast_error_new = np.mean(forecast_error_new)
>>> mean_forecast_error_new

-12.726958780163237
```

The RMSE has improved slightly (dropped to 57), while the MDA has dropped to 86% and the mean forecast error stands at -12, meaning that the model has a tendency to slightly underestimate the cancellations and therefore the forecast bias is negative.

Here is a plot of the predicted vs actual cancellations:

![predicted-vs-actual](https://github.com/MGCodesandStats/hotel-modelling/blob/master/images/predicted-vs-actual.png)

## ARIMA Modelling on H2 Data

The same procedures were applied - this time using the second dataset.

The following is the ARIMA configuration obtained using pyramid-arima:

![arima-model-2](https://github.com/MGCodesandStats/hotel-modelling/blob/master/images/arima-model-2.PNG)

**Predicted vs. Validation**

![predicted-vs-val-2](https://github.com/MGCodesandStats/hotel-modelling/blob/master/images/predicted-vs-val-2.png)

**Predicted vs. Actual**

![predicted-vs-actual-2](https://github.com/MGCodesandStats/hotel-modelling/blob/master/images/predicted-vs-actual-2.png)

- **RMSE on test data:** 274
- **Mean Directional Accuracy:** 0.8666
- **Mean Forecast Error:** 156.329

# Conclusion

This has been an illustration of how logistic regression and SVM models can be used to predict hotel cancellations. We have also seen how the Extra Trees Classifier can be used as a feature selection tool to identify the most reliable predictors of customer cancellations. The ARIMA model has also been used to predict the degree of hotel cancellations on a week-by-week basis. The MDA demonstrated 86% accuracy in doing so across the test set with an RMSE of 57 on the H1 dataset, and an 86% MDA was yielded once again for the H2 dataset with an RMSE of 274 (with the mean cancellations across the 15 weeks in the test set coming in at 327).

Of course, a limitation of these findings is that both hotels under study are based in Portugal. Testing the model across hotels in other countries would help to validate the accuracy of this model further.
