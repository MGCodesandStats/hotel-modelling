# XGBoost and Classification: Predicting Hotel Cancellations

Boosting is quite a popular technique in machine learning, which aims to improve prediction accuracy by combining many weak models into one strong model.

For this reason, boosting is referred to as an **ensemble method**.

In this example, boosting techniques are used to determine whether a customer will cancel their hotel booking or not.

## Data Overview and Feature Selection

Hotel cancellations represent the response (or dependent) variable, where 1 = cancel, 0 = follow through with booking.

The features for analysis are as follows.

### Interval

```
leadtime = train_df['LeadTime']
arrivaldateyear = train_df['ArrivalDateYear']
arrivaldateweekno = train_df['ArrivalDateWeekNumber']
arrivaldatedayofmonth = train_df['ArrivalDateDayOfMonth']
staysweekendnights = train_df['StaysInWeekendNights']
staysweeknights = train_df['StaysInWeekNights']
adults = train_df['Adults']
children = train_df['Children']
babies = train_df['Babies']
isrepeatedguest = train_df['IsRepeatedGuest'] 
previouscancellations = train_df['PreviousCancellations']
previousbookingsnotcanceled = train_df['PreviousBookingsNotCanceled']
bookingchanges = train_df['BookingChanges']
agent = train_df['Agent']
company = train_df['Company']
dayswaitinglist = train_df['DaysInWaitingList']
adr = train_df['ADR']
rcps = train_df['RequiredCarParkingSpaces']
totalsqr = train_df['TotalOfSpecialRequests']
```

### Categorical

```
arrivaldatemonth = train_df.ArrivalDateMonth.astype("category").cat.codes
arrivaldatemonthcat=pd.Series(arrivaldatemonth)
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
```

The identified features to be included in the analysis using both the **ExtraTreesClassifier** and **forward and backward feature selection** methods are as follows:

- Lead time
- Country of origin
- Market segment
- Deposit type
- Customer type
- Required car parking spaces
- Arrival Date: Year
- Arrival Date: Month
- Arrival Date: Week Number
- Arrival Date: Day of Month

## Boosting Techniques

XGBoost is a boosting technique that has become renowned for its execution speed and model performance, and is increasingly being relied upon as a default boosting method - this method implements the gradient boosting decision tree algorithm which works in a similar manner to adaptive boosting, but instance weights are no longer tweaked at every iteration as in the case of AdaBoost. Instead, an attempt is made to fit the new predictor to the residual errors that the previous predictor made.

## Precision vs. Recall and f1-score

When comparing the accuracy scores, we see that numerous readings are provided in each confusion matrix.

However, a particularly important distinction exists between **precision** and **recall**. 

```
Precison = ((True Positive)/(True Positive + False Positive))

Recall = ((True Positive)/(True Positive + False Negative))
```

The two readings are often at odds with each other, i.e. it is often not possible to increase precision without reducing recall, and vice versa.

An assessment as to the ideal metric to use depends in large part on the specific data under analysis. For example, cancer detection screenings that have false negatives (i.e. indicating patients do not have cancer when in fact they do), is a big no-no. Under this scenario, recall is the ideal metric.

However, for emails - one might prefer to avoid false positives, i.e. sending an important email to the spam folder when in fact it is legitimate.

The f1-score takes both precision and recall into account when devising a more general score.

Which would be more important for predicting hotel cancellations?

Well, from the point of view of a hotel - they would likely wish to identify customers who are ultimately going to cancel their booking with greater accuracy - this allows the hotel to better allocate rooms and resources. Identifying customers who are not going to cancel their bookings may not necessarily add value to the hotel's analysis, as the hotel knows that a significant proportion of customers will ultimately follow through with their bookings in any case.

## Analysis

The data is firstly split into training and validation data for the H1 dataset, with the H2 dataset being used as the test set for comparing the XGBoost predictions with actual cancellation incidences.

Here is an implementation of the XGBoost algorithm:

```
import xgboost as xgb
xgb_model = xgb.XGBClassifier(learning_rate=0.001,
                            max_depth = 1, 
                            n_estimators = 100,
                              scale_pos_weight=5)
xgb_model.fit(x_train, y_train)
```

Note that the *scale_pos_weight* parameter in this instance is set to *5*. The reason for this is to impose greater penalties for errors on the minor class, in this case any incidences of *1* in the response variable, i.e. hotel cancellations. The higher the weight, the greater penalty is imposed on errors on the minor class.

### Performance on Validation Set

Here is the accuracy on the training and validation set:

```
>>> print("Accuracy on training set: {:.3f}".format(xgb_model.score(x_train, y_train)))
>>> print("Accuracy on validation set: {:.3f}".format(xgb_model.score(x_val, y_val)))

Accuracy on training set: 0.415
Accuracy on validation set: 0.414
```

The predictions are generated:

```
>>> xgb_predict=xgb_model.predict(x_val)
>>> xgb_predict

array([1, 1, 1, ..., 1, 1, 1])
```

Here is a confusion matrix comparing the predicted vs. actual cancellations on the validation set:

```
>>> from sklearn.metrics import classification_report,confusion_matrix
>>> print(confusion_matrix(y_val,xgb_predict))
>>> print(classification_report(y_val,xgb_predict))

[[1393 5873]
 [   0 2749]]
              precision    recall  f1-score   support

           0       1.00      0.19      0.32      7266
           1       0.32      1.00      0.48      2749

    accuracy                           0.41     10015
   macro avg       0.66      0.60      0.40     10015
weighted avg       0.81      0.41      0.37     10015
```
Note that while the accuracy in terms of the f1-score (41%) is quite low - the recall score for class 1 (cancellations) is 100%. This means that the model is generating many false positives which reduces the overall accuracy - but this has had the effect of increasing recall to 100%, i.e. the model is 100% successful at identifying all the customers who will cancel their booking, even if this results in some false positives.

### Performance on Test Set

Here is the subsequent classification performance of the XGBoost model on H2, which is the test set in this instance.

```
[[ 1926 44302]
 [    0 33102]]
              precision    recall  f1-score   support

           0       1.00      0.04      0.08     46228
           1       0.43      1.00      0.60     33102

    accuracy                           0.44     79330
   macro avg       0.71      0.52      0.34     79330
weighted avg       0.76      0.44      0.30     79330
```

The accuracy as indicated by the f1-score is slightly higher at 44%, but the recall accuracy for class 1 is at 100% once again.

## Conclusion

In this example, you have seen the use of various boosting methods to predict hotel cancellations. As mentioned, the boosting method in this instance was set to impose greater penalties on the minor class, which had the result of lowering the overall accuracy as measure by the f1-score since there were more false positives present. However, the recall score increased vastly as a result - if it is assumed that false positives are more tolerable than false negatives in this situation - then one could argue that the model has performed quite well on this basis. For reference, an SVM model run on the same dataset demonstrated an overall accuracy of 63%, while recall on class 1 decreased to 75%.

## Useful References

- [Classification: Precision and Recall](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall)

- Hands-On Machine Learning with Scikit-Learn & TensorFlow by Aurélien Geron

- [Machine Learning Mastery: A Gentle Introduction to XGBoost for Applied Machine Learning](https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/)

- [What is LightGBM, How to implement it? How to fine tune the parameters?](https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc)
