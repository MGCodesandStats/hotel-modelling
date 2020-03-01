# Boosting Techniques in Python: Predicting Hotel Cancellations

Boosting is quite a popular technique in machine learning, which aims to improve prediction accuracy by combining many weak models into one strong model.

For this reason, boosting is referred to as an **ensemble method**.

In this example, boosting techniques are used to determine whether a customer will cancel their hotel booking or not.

## Data Overview and Feature Selection

Hotel cancellations represent the response (or dependent) variable, where 1 = cancel, 0 = follow through with booking.

The features for analysis are as follows:

1. leadtime
2. staysweekendnights
3. staysweeknights
4. adults
5. children
6. babies
7. meal
8. country
9. marketsegment
10. distributionchannel
11. isrepeatedguest
12. previouscancellations
13. previousbookingsnotcanceled
14. reservedroomtype
15. assignedroomtype
16. bookingchanges
17. deposittype
18. dayswaitinglist
19. customertype
20. adr (average daily rate)
21. rcps (required car parking spaces)
22. totalsqr

The relevant features to be included as the *x* variable in the boosting models are identified by the ExtraTreesClassifier.

```
>>> from sklearn.ensemble import ExtraTreesClassifier
>>> model = ExtraTreesClassifier()
>>> model.fit(x, y)
>>> print(model.feature_importances_)

[0.00000000e+00 2.52553177e-02 3.97757078e-03 4.85985259e-03
 2.97916350e-03 2.72472443e-03 3.86899881e-04 3.53988978e-03
 4.06165166e-02 2.14971621e-02 8.89432487e-03 7.53453610e-03
 7.97051828e-03 1.51441200e-03 6.56966716e-03 9.22417386e-03
 4.84711487e-03 2.73077182e-02 7.80494441e-04 1.35137200e-02
 9.38820537e-03 4.08951776e-02 7.61453579e-03 7.48108304e-01]
 ```

The three features identified by the ExtraTreesClassifier (excluding variables deemed to be theoretically irrelevant) are *lead time*, *country* and *deposit type*.

## Boosting Techniques

The following boosting techniques are used in predicting hotel cancellations.

- **AdaBoost (Adaptive Boosting):** AdaBoost works by specifically focusing on training instances that were underfitted by the previous predictor - which in turn allows the classifier to focus on cases that other models may underlook during the course of model building.

- **Gradient Boosting Classifier:** Gradient boosting works in a similar manner to adaptive boosting, but instance weights are no longer tweaked at every iteration as in the case of AdaBoost. Instead, an attempt is made to fit the new predictor to the residual errors that the previous predictor made.

- **XGBoost:** XGBoost is a boosting technique that has become renowned for its execution speed and model performance, and is increasingly being relied upon as a default boosting method - this method implements the gradient boosting decision tree algorithm. 

- **Light Gradient Boosting:** Light Gradient Boosting (also known as LightGBM) also uses a tree based learning algorithm, but has the advantage of using lower memory and is typically best when working with large datasets.

Specifically, the classification accuracy of each technique on the validation dataset is evaluated, with the model deemed most accurate being selected for use on the test set.

## Analysis

The boosting techniques are implemented and the accuracy on both the training and validation sets are calculated.

### Gradient Boosting Classifier

```
>>> from sklearn.ensemble import GradientBoostingClassifier
>>> gbrt = GradientBoostingClassifier(random_state=0)
>>> gbrt.fit(x_train, y_train)

>>> print("Accuracy on training set: {:.3f}".format(gbrt.score(x_train, y_train)))
>>> print("Accuracy on validation set: {:.3f}".format(gbrt.score(x_val, y_val)))

Accuracy on training set: 0.754
Accuracy on validation set: 0.748
```
### LightGBM

```
>>> import lightgbm as lgb
>>> lgb_model = lgb.LGBMClassifier(learning_rate = 0.001, 
>>>                               num_leaves = 65,  
>>>                               n_estimators = 100)                       
>>> lgb_model.fit(x_train, y_train)

>>> print("Accuracy on training set: {:.3f}".format(lgb_model.score(x_train, y_train)))
>>> print("Accuracy on validation set: {:.3f}".format(lgb_model.score(x_val, y_val)))

Accuracy on training set: 0.758
Accuracy on validation set: 0.746
```

### XGBoost

```
>>> import xgboost as xgb
>>> xgb_model = xgb.XGBClassifier(learning_rate=0.001,
>>>                             max_depth = 1, 
>>>                             n_estimators = 100)
>>> xgb_model.fit(x_train, y_train)

>>> print("Accuracy on training set: {:.3f}".format(xgb_model.score(x_train, y_train)))
>>> print("Accuracy on validation set: {:.3f}".format(xgb_model.score(x_val, y_val)))

Accuracy on training set: 0.639
Accuracy on validation set: 0.630
```

### AdaBoost

```
>>> from sklearn.ensemble import AdaBoostClassifier
>>> from sklearn.tree import DecisionTreeClassifier
>>> ada_clf = AdaBoostClassifier(
>>>     DecisionTreeClassifier(max_depth=1), n_estimators=100,
>>>     algorithm="SAMME.R", learning_rate=0.001)
>>> ada_clf.fit(x_train, y_train)

print("Accuracy on training set: {:.3f}".format(ada_clf.score(x_train, y_train)))
print("Accuracy on validation set: {:.3f}".format(ada_clf.score(x_val, y_val)))

Accuracy on training set: 0.639
Accuracy on validation set: 0.630
```

## Confusion Matrices

A confusion matrix is a table that summarizes the performance of a classification algorithm.

Specifically, the confusion matrix allows us to view the mix of true/false positives and negatives, to determine how accurate the model was in predicting whether a customer would cancel their hotel booking. When predicting across the validation set, here are the confusion matrices for each model:

### Gradient Boosting Classifier

```
[[2050  481]
 [ 777 1692]]
              precision    recall  f1-score   support

           0       0.73      0.81      0.77      2531
           1       0.78      0.69      0.73      2469

    accuracy                           0.75      5000
   macro avg       0.75      0.75      0.75      5000
weighted avg       0.75      0.75      0.75      5000
```

### LightGBM

```
[[1921  610]
 [ 662 1807]]
              precision    recall  f1-score   support

           0       0.74      0.76      0.75      2531
           1       0.75      0.73      0.74      2469

    accuracy                           0.75      5000
   macro avg       0.75      0.75      0.75      5000
weighted avg       0.75      0.75      0.75      5000
```

### XGBoost

```
[[ 907 1624]
 [ 226 2243]]
              precision    recall  f1-score   support

           0       0.80      0.36      0.50      2531
           1       0.58      0.91      0.71      2469

    accuracy                           0.63      5000
   macro avg       0.69      0.63      0.60      5000
weighted avg       0.69      0.63      0.60      5000
```

### AdaBoost

```
[[ 907 1624]
 [ 226 2243]]
              precision    recall  f1-score   support

           0       0.80      0.36      0.50      2531
           1       0.58      0.91      0.71      2469

    accuracy                           0.63      5000
   macro avg       0.69      0.63      0.60      5000
weighted avg       0.69      0.63      0.60      5000
```

## Precision vs. Recall and f1-score

When comparing the accuracy scores, we see that numerous readings are provided in each confusion matrix.

However, a particularly important distinction exists between **precision** and **recall**. 

**Precison = ((True Positive)/(True Positive + False Positive))**

**Recall = ((True Positive)/(True Positive + False Negative))**

The two readings are often at odds with each other, i.e. it is often not possible to increase precision without reducing recall, and vice versa.

An assessment as to the ideal metric to use depends in large part on the specific data under analysis. For example, cancer detection screenings that have false negatives (i.e. indicating patients do not have cancer when in fact they do), is a big no-no. Under this scenario, recall is the ideal metric.

However, for emails - one might prefer to avoid false positives, i.e. sending an important email to the spam folder when in fact it is legitimate.

The f1-score takes both precision and recall into account when devising a more general score.

Which would be more important for predicting hotel cancellations?

Well, from the point of view of a hotel - they would likely wish to identify customers who are ultimately going to cancel their booking with greater accuracy - this allows the hotel to better allocate rooms and resources. Identifying customers who are not going to cancel their bookings may not necessarily add value to the hotel's analysis, as the hotel knows that a significant proportion of customers will ultimately follow through with their bookings in any case.

Let's compare the test set results for the four boosting models, which are used to compare predictions on a separate test set to the actual response variable on that test set.

## Test Set - Confusion Matrix Results

### Gradient Boosting Classifier

```
[[6609  395]
 [2829 2167]]
              precision    recall  f1-score   support

           0       0.70      0.94      0.80      7004
           1       0.85      0.43      0.57      4996

    accuracy                           0.73     12000
   macro avg       0.77      0.69      0.69     12000
weighted avg       0.76      0.73      0.71     12000
```

### LightGBM

```
[[5843 1161]
 [2509 2487]]
              precision    recall  f1-score   support

           0       0.70      0.83      0.76      7004
           1       0.68      0.50      0.58      4996

    accuracy                           0.69     12000
   macro avg       0.69      0.67      0.67     12000
weighted avg       0.69      0.69      0.68     12000
```

### XGBoost

```
[[1929 5075]
 [ 360 4636]]
              precision    recall  f1-score   support

           0       0.84      0.28      0.42      7004
           1       0.48      0.93      0.63      4996

    accuracy                           0.55     12000
   macro avg       0.66      0.60      0.52     12000
weighted avg       0.69      0.55      0.50     12000
```

### AdaBoost

```
[[1929 5075]
 [ 360 4636]]
              precision    recall  f1-score   support

           0       0.84      0.28      0.42      7004
           1       0.48      0.93      0.63      4996

    accuracy                           0.55     12000
   macro avg       0.66      0.60      0.52     12000
weighted avg       0.69      0.55      0.50     12000
```

We can see that XGBoost and AdaBoost have a higher recall of 93% for instances of cancellations (a response of 1) - meaning that 93% of all cancellations were identified correctly. However, we see that the overall accuracy based on the weighted average f1-score was 50%. However, with a precision of 48%, this means that only 48% of positive identifications were actually correct - meaning that the model produced a significant number of false positives.

In the case of the Gradient Boosting Classifier and LightGBM, the weighted average f1-scores were 71% and 68%.

In this regard, XGBoost and AdaBoost excel at identifying most of the customers who cancel their booking. However, the model also produces too many false positives - i.e. identifying a customer as cancelling when in fact they don't.

The Gradient Boosting Classifier and LightGBM models give a higher accuracy overall, but the recall is significantly lower at 43% and 50% respectively.

In this regard, choice of model is not only based on overall accuracy, and instead must be chosen with respect to the specific situation.

## Conclusion

In this example, you have seen the use of various boosting methods to predict hotel cancellations. Many thanks for your time, and you can find the associated GitHub repository for the above examples [here](https://github.com/MGCodesandStats/boosting-hotel-cancellations).

## Useful References

- [Classification: Precision and Recall](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall)

- Hands-On Machine Learning with Scikit-Learn & TensorFlow by Aurélien Geron

- [Machine Learning Mastery: A Gentle Introduction to XGBoost for Applied Machine Learning](https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/)

- [What is LightGBM, How to implement it? How to fine tune the parameters?](https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc)
