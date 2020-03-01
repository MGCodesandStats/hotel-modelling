# SVMs, Random Forests and Unbalanced Datasets: Predicting Hotel Cancellations

When attempting to build a classification algorithm, one must often contend with the issue of an unbalanced dataset.

An **unbalanced dataset** is one where there is an unequal sample size between classes, which induces significant bias into the predictions of the classifier in question.

In this particular example (available from the references section below), a support vector machine (SVM) classification model is used to classify hotel booking customers in terms of cancellation risk, i.e. **1** if the model predicts that the customer will cancel their booking, **0** if the customer will follow through with the booking.

The H1 dataset is used to train and validate the model, while the predictions from the resulting model are then tested using the H2 data.

In this particular dataset, the sample size for the non-cancellation class (0) is significantly greater than the cancellation class (1). In a previous example, this was dealt with by removing numerous **0** entries in order to have an equal sample size between the two classes. However, this is not necessarily the best approach, as many data points are discarded during this process.

Instead, the SVM model can be modified to penalise wrong predictions on the minor class. Let's see how this affects the analysis. A RandomForest classifier will also be used to determine if the prediction results are superior to those predicted by SVM.

## SVM and Unbalanced Datasets

### Equal sample size of 0 and 1

The relevant features of lead time, country of origin and deposit type are selected as the relevant features for determining whether the customer will cancel their booking.

```
y1 = y
x1 = np.column_stack((leadtime,countrycat,deposittypecat))
x1 = sm.add_constant(x1, prepend=True)
```

The data is then split into training and test (technically validation) data:

```
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=0)
```

Here is the SVM model configuration:

```
from sklearn import svm
clf = svm.SVC(gamma='scale')
clf.fit(x1, y1)  
prclf = clf.predict(x1_test)
prclf
```

Here is the classification matrix on the test set:

```
>>> print(confusion_matrix(y1_test,prclf))
>>> print(classification_report(y1_test,prclf))

[[2085  446]
 [ 963 1506]]
              precision    recall  f1-score   support

           0       0.68      0.82      0.75      2531
           1       0.77      0.61      0.68      2469

    accuracy                           0.72      5000
   macro avg       0.73      0.72      0.71      5000
weighted avg       0.73      0.72      0.71      5000
```

The model shows an overall accuracy of **72%** based on the f1-score, and recall for class 1 (recall being the metric that we wish to use when avoiding false negatives) is slightly lower at **61%**. This means that the model demonstrates too many false negatives, i.e. the model predicted certain customers would not cancel their booking, when in fact they did.

Here is how the model predictions perform when assessed against the test set (H2).

```
>>> from sklearn.metrics import classification_report,confusion_matrix
>>> print(confusion_matrix(b,prh2))
>>> print(classification_report(b,prh2))

[[5654 1350]
 [2038 2958]]
              precision    recall  f1-score   support

           0       0.74      0.81      0.77      7004
           1       0.69      0.59      0.64      4996

    accuracy                           0.72     12000
   macro avg       0.71      0.70      0.70     12000
weighted avg       0.71      0.72      0.71     12000
```

We see that while the f1-score accuracy remains at **72%**, recall fell slightly further to 59%. As mentioned, using this model is not ideal as many data points have been discarded to balance the number of data points for each class - which discards potentially valuable data.

### Unequal sample sizes of 0 and 1, 'balanced' class weight

In the original datasets, the sample sizes of 0 and 1 are unequal, with data points for the latter being more numerous than the former.

This time, a 'balanced' class weight can be added to the SVM configuration, which adds a greater penalty to incorrect classifications on the minor class (in this case, the cancellation class).

```
from sklearn import svm
clf = svm.SVC(gamma='scale', 
            class_weight='balanced')
clf.fit(x1, y1)  
prclf = clf.predict(x1_test)
prclf
```

Here is the classification performance of this model on the validation set:

```
[[5954 1312]
 [1078 1671]]
              precision    recall  f1-score   support

           0       0.85      0.82      0.83      7266
           1       0.56      0.61      0.58      2749

    accuracy                           0.76     10015
   macro avg       0.70      0.71      0.71     10015
weighted avg       0.77      0.76      0.76     10015
```

Recall for class 1 remains at 61%, while the f1-score accuracy increases to 76%. Now, let's test the prediction performance on H2 (the test set).

```
[[35632 10596]
 [12028 21074]]
              precision    recall  f1-score   support

           0       0.75      0.77      0.76     46228
           1       0.67      0.64      0.65     33102

    accuracy                           0.71     79330
   macro avg       0.71      0.70      0.70     79330
weighted avg       0.71      0.71      0.71     79330
```

We see that recall for class 1 is now up to 64%, while f1-score accuracy comes in at 71%. This indicates that the SVM is handling the unbalanced classes properly, since recall for class 1 is higher than that predicted when the classes were of equal sample size, and the accuracy of 71% is just slightly lower than the 72% obtained with the previous model. However, this time the analysis is more thorough given that we have not discarded many data points in an attempt to make the sample size equal.

## Random Forest

Can a random forest do any better in predicting the classifications? Let's find out!

The classifier is configured as follows:

```
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(class_weight='balanced')
rfc.fit(x1, y1)  
pr_rfc = rfc.predict(x1_test)
pr_rfc
```

Here is the confusion matrix for the validation set:

```
[[5799 1467]
 [ 518 2231]]
              precision    recall  f1-score   support

           0       0.92      0.80      0.85      7266
           1       0.60      0.81      0.69      2749

    accuracy                           0.80     10015
   macro avg       0.76      0.80      0.77     10015
weighted avg       0.83      0.80      0.81     10015
```

We see that recall for class 1 is quite high at 81%, with f1-score accuracy coming in at 80%.

That said, we see that the accuracy scores dip quite significantly when the predictions are tested against H2:

```
[[33909 12319]
 [15492 17610]]
              precision    recall  f1-score   support

           0       0.69      0.73      0.71     46228
           1       0.59      0.53      0.56     33102

    accuracy                           0.65     79330
   macro avg       0.64      0.63      0.63     79330
weighted avg       0.65      0.65      0.65     79330
```

In this regard, the SVM model demonstrated more success when using a balanced class weight to predict hotel cancellations across the unbalanced dataset.

## Using Time Series Data Points As Features

In the above analysis, it is noteworthy that time features, i.e. month, week number, etc, were not incorporated into the models used for analysis. Would including these features make a significant difference to recall and overall accuracy readings? Let's investigate further.

### Feature Selection with ExtraTreesClassifier

Firstly, the variables *arrivaldateyear*, *arrivaldateweekno*, *arrivaldatemonth*, and *arrivaldatedayofmonth* are included in the analysis, with *arrivaldatemonth* being defined in categorical format, i.e. months of the year are defined as categories.

```
arrivaldateyear = train_df['ArrivalDateYear']
arrivaldateweekno = train_df['ArrivalDateWeekNumber']
arrivaldatedayofmonth = train_df['ArrivalDateDayOfMonth']
arrivaldatemonth = train_df.ArrivalDateMonth.astype("category").cat.codes
arrivaldatemonthcat=pd.Series(arrivaldatemonth)
```

These features, defined as features *2*, *3*, *4*, and *5*, are not ranked in the top five features by the ExtraTreesClassifier:

![top5.png](top5.png)

That said, the time of year may well have at least some effect on cancellation frequency, even if this does not rank among the top features.

In this regard, let's include these four features in the model, and examine what happens to *recall* and *f1-score* accuracy.

```
y1 = y
x1 = np.column_stack((leadtime,countrycat,deposittypecat,arrivaldateyear,arrivaldatemonthcat,arrivaldateweekno,arrivaldatedayofmonth))
x1 = sm.add_constant(x1, prepend=True)
```

### SVM performance on H2 (test set) with time features included

Here is an overview of SVM performance on the test set according to the confusion matrix

```
[[26804 19424]
 [ 8840 24262]]
              precision    recall  f1-score   support

           0       0.75      0.58      0.65     46228
           1       0.56      0.73      0.63     33102

    accuracy                           0.64     79330
   macro avg       0.65      0.66      0.64     79330
weighted avg       0.67      0.64      0.65     79330
```

In this regard, we see that while including the extra features has caused a drop in f1-score accuracy from 71% to 64%.

However, we also note that recall for class 1 has increased significantly from 64% to 73%. Should the hotel be interested in reducing false negatives, i.e. where the model predicts a customer will not cancel but they in fact do, then recall is the superior metric. Accordingly, the SVM model has performed better on a recall basis when the time features are included.

### Random Forest performance on H2 (test set) with time features included

Now, let's examine performance with the time features, but this time using the Random Forest algorithm once again.

```
[[42646  3582]
 [18319 14783]]
              precision    recall  f1-score   support

           0       0.70      0.92      0.80     46228
           1       0.80      0.45      0.57     33102

    accuracy                           0.72     79330
   macro avg       0.75      0.68      0.69     79330
weighted avg       0.74      0.72      0.70     79330
```

We see that while the Random Forest accuracy is higher at 72% according to the f1-score, the recall performance for class 1 is lower at 45%. Therefore, while including the time features increased overall accuracy from 65% to 72%, recall fell from a previous 53% when the time features were not included.

While the choice of model would depend on whether reducing of false negatives is deemed important or not, it is clear that including the time features has proven useful in informing the accuracy of the models.

## Conclusion

In this example, we have seen how support vector machines and random forests can be used to handle unbalanced datasets, and how to interpret confusion matrices for classification accuracy. We also looked at inclusion of time features in the modelling process can also be useful when working with classification data.

Many thanks for reading, and you can also find other machine learning tutorials at [michael-grogan.com](https://www.michael-grogan.com/).

## References

[- Elite Data Science: How to Handle Imbalanced Classes in Machine Learning](https://elitedatascience.com/imbalanced-classes)

[- Predicting Hotel Cancellations with Support Vector Machines and ARIMA](https://towardsdatascience.com/predicting-hotel-cancellations-with-extratreesclassifier-and-logistic-regression-fb9229a95d1e)
