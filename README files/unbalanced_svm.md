# Imbalanced Classes: Predicting Hotel Cancellations with Support Vector Machines

When attempting to build a classification algorithm, one must often contend with the issue of an unbalanced dataset.

An **unbalanced dataset** is one where there is an unequal sample size between classes, which induces significant bias into the predictions of the classifier in question.

In this particular example (available from the references section below), a support vector machine (SVM) classification model is used to classify hotel booking customers in terms of cancellation risk, i.e. **1** if the model predicts that the customer will cancel their booking, **0** if the customer will follow through with the booking.

The H1 dataset is used to train and validate the model, while the predictions from the resulting model are then tested using the H2 data.

In this particular dataset, the sample size for the non-cancellation class (0) is significantly greater than the cancellation class (1). In a previous example, this was dealt with by removing numerous **0** entries in order to have an equal sample size between the two classes. However, this is not necessarily the best approach, as many data points are discarded during this process.

Instead, the SVM model can be modified to penalise wrong predictions on the minor class. Let's see how this affects the analysis.

## SVM and Unbalanced Datasets

### Equal sample size of 0 and 1

The relevant features of lead time, country of origin and deposit type are selected as the relevant features for determining whether the customer will cancel their booking.

```
y1 = y
x1 = np.column_stack((leadtime,countrycat,marketsegmentcat,deposittypecat,customertypecat,rcps,arrivaldateyear,arrivaldatemonthcat,arrivaldateweekno,arrivaldatedayofmonth))
x1 = sm.add_constant(x1, prepend=True)
```

The data is then split into training and test (technically validation) data:

```
x1_train, x1_val, y1_train, y1_val = train_test_split(x1, y1, random_state=0)
```

A 'balanced' class weight can be added to the SVM configuration, which adds a greater penalty to incorrect classifications on the minor class (in this case, the cancellation class).

```
from sklearn import svm
clf = svm.SVC(gamma='scale', 
            class_weight='balanced')
clf.fit(x1_train, y1_train)  
prclf = clf.predict(x1_val)
prclf
```

Here is the classification performance of this model on the validation set:

```
[[5142 2124]
 [ 865 1884]]
              precision    recall  f1-score   support

           0       0.86      0.71      0.77      7266
           1       0.47      0.69      0.56      2749

    accuracy                           0.70     10015
   macro avg       0.66      0.70      0.67     10015
weighted avg       0.75      0.70      0.72     10015
```

Recall for class 1 comes in at 61%, while the f1-score accuracy comes in at 76%. Now, let's test the prediction performance on H2 (the test set).

```
[[25217 21011]
 [ 8436 24666]]
              precision    recall  f1-score   support

           0       0.75      0.55      0.63     46228
           1       0.54      0.75      0.63     33102

    accuracy                           0.63     79330
   macro avg       0.64      0.65      0.63     79330
weighted avg       0.66      0.63      0.63     79330
```

We see that recall for class 1 is now up to 75%, while f1-score accuracy comes in at 63%. Notably, we can see that the f1-score is lower on the test set, but the recall is now higher than on the validation set.

In this regard, if it is assumed that false positives are more tolerable than false negatives in this situation - then one could argue that the model has performed quite well on this basis.

## Conclusion

In this example, we have seen how support vector machines can be used to handle unbalanced datasets, and how to interpret confusion matrices for classification accuracy.

## References

[- Elite Data Science: How to Handle Imbalanced Classes in Machine Learning](https://elitedatascience.com/imbalanced-classes)

[- Predicting Hotel Cancellations with Support Vector Machines and ARIMA](https://towardsdatascience.com/predicting-hotel-cancellations-with-extratreesclassifier-and-logistic-regression-fb9229a95d1e)
