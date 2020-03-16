# Classification of Hotel Cancellations Using KNN and SMOTE

KNN (K-Nearest Neighbors) is a go-to method for classification purposes.

In this particular example, the KNN algorithm is used to classify hotel bookings in terms of cancellation risk (1 = model predicts that the customer will cancel their booking, 0 = customer is not predicted to cancel their booking).

Given that this dataset is unbalanced, i.e. there are more 0s (non-cancellations) than 1s (cancellations), the Synthetic Minority Oversampling Technique (SMOTE) is used to balance the classes in order to apply the KNN algorithm.

## SMOTE Oversampling Technique

As mentioned, the dataset in question is unbalanced. As a result, it is necessary to oversample the minor class (1 = cancellations) in order to ensure that the KNN results are not skewed towards the major class.

This can be done by using the SMOTE technique.

After having imported and scaled the data using MinMaxScaler, SMOTE can be imported from the imblearn library:

```
import imblearn
print(imblearn.__version__)
from imblearn.over_sampling import SMOTE
from collections import Counter
```

Counter is imported for the purposes of summarizing the class distributions.

For instance, the original class distribution comprised of 0: 28938 and 1: 11122. However, after applying the SMOTE oversampling technique, we now see that the number of observations in each class are equal.

(1)

A train-test split is then invoked to separate the data into training and validation data.

```
x1_train, x1_val, y1_train, y1_val = train_test_split(x_scaled, y1, random_state=0)
```

Specifically, this model is being trained on the H1 hotel dataset (with a subset of this dataset being used for validation), while H2 is being used as the test data to compare predictions with the actual responses.

## K-Fold Cross Validation and KNN

One important caveat regarding the train-test split method is that depending on the nature of the split, we may find a situation where the model performs well on one random set of validation data but not on another.

Therefore, should we find a situation where the model is performing well in predicting one validation set - it does not necessarily mean that it will do so for all sets.

In this regard, a K-fold cross-validation technique is used whereby the data is split into k subsamples with an equal number of observations. A single subsample is used as the validation data while the remaining k-1 subsamples are used as training data. This process is then repeated, with each subsample being used once as the validation data, with the process repeated k times.

Using trial and error, I decided on k=10 as the appropriate number of folds.

Here is the training and test score when a simple train-test split is used.

The model is configured as follows:

```
# KNN
knn = KNeighborsClassifier(n_neighbors=10)
model=knn.fit(x1_train, y1_train)
pred = model.predict(x1_val)
pred
print("Training set score: {:.2f}".format(knn.score(x1_train, y1_train)))
print("Validation set score: {:.2f}".format(knn.score(x1_val, y1_val)))

# KNN Plot
mglearn.plots.plot_knn_classification(n_neighbors=10)
plt.show()
```

The training and test set scores are generated:

```
Training set score: 0.87
Validation set score: 0.83
```

Here is a visual of the training classes versus test predictions as illustrated by the KNN model:

(2)

While the validation set score of 0.84 was impressive, how do we know whether we were simply lucky in selecting the validation set that validated the predictions of the model? To ensure that this is not just a random fluke, the k-fold cross validation technique can be used.

The value of k (number of subsets) is set to 10 in this instance.

```
# Cross Validation: 
from sklearn.model_selection import cross_val_score, cross_val_predict
scores = cross_val_score(model, x_scaled, y1, cv=10)
print ("Cross-validated scores:", scores)
print("Mean score: {}".format(np.mean(scores)))
```

The mean score across the 10 folds comes in at 0.67:

```
Cross-validated scores: [0.80597789 0.79111956 0.71510021 0.71129924 0.69212163 0.63337941
 0.6412649  0.59858303 0.6251944  0.54207707]
Mean score: 0.6756117332309184
```

Here is a breakdown of the model performance according to a confusion matrix:

```
[[5754 1529]
 [ 824 6362]]
              precision    recall  f1-score   support

           0       0.87      0.79      0.83      7283
           1       0.81      0.89      0.84      7186

    accuracy                           0.84     14469
   macro avg       0.84      0.84      0.84     14469
weighted avg       0.84      0.84      0.84     14469
```

The accuracy according to the f1-score is reasonably good at 84%.

### Precision vs. Recall

However, when working with classification data, one must also pay attention to the precision versus recall readings, as opposed to simply overall accuracy.

```
Precision = ((True Positive)/(True Positive + False Positive))
Recall = ((True Positive)/(True Positive + False Negative))
```

The two readings are often at odds with each other, i.e. it is often not possible to increase precision without reducing recall, and vice versa.

An assessment as to the ideal metric to use depends in large part on the specific data under analysis. For example, cancer detection screenings that have false negatives (i.e. indicating patients do not have cancer when in fact they do), is a big no-no. Under this scenario, recall is the ideal metric.

However, for emails — one might prefer to avoid false positives, i.e. sending an important email to the spam folder when in fact it is legitimate.

The f1-score takes both precision and recall into account when devising a more general score.
Which would be more important for predicting hotel cancellations?

Well, from the point of view of a hotel — they would likely wish to identify customers who are ultimately going to cancel their booking with greater accuracy — this allows the hotel to better allocate rooms and resources. Identifying customers who are not going to cancel their bookings may not necessarily add value to the hotel’s analysis, as the hotel knows that a significant proportion of customers will ultimately follow through with their bookings in any case.

## Test Data

Let us see how the results look when the model makes predictions on H2 (the test set).

Here, we see that the f1-score accuracy has decreased significantly to 55%.

```
[[15958 30270]
 [ 5602 27500]]
              precision    recall  f1-score   support

           0       0.74      0.35      0.47     46228
           1       0.48      0.83      0.61     33102

    accuracy                           0.55     79330
   macro avg       0.61      0.59      0.54     79330
weighted avg       0.63      0.55      0.53     79330
```

However, the recall for the cancellation class (1) stands at 83%. As mentioned, precision and recall are often at odds with each other simply due to the fact that false positives tend to increase recall, while false negatives tend to increase precision.

Assuming that the hotel would like to maximise recall (i.e. tolerate a certain number of false positives while at the same time identifying all customers who will cancel their booking), then this model meets that criteria.

Of all customers who cancel their booking, this model correctly identifies 83% of those customers.

## Conclusion

In this example, you have seen:

- The use of KNN as a classification algorithm
- How K-fold cross-validation can provide a better overview of model performance
- Importance of precision versus recall in judging model performance.

Many thanks for your time, and the associated GitHub repository for this example can be found [here](https://github.com/MGCodesandStats/hotel-modelling).

## References

- [GitHub repository (Msanjayds): Cross-Validation calculation](https://github.com/Msanjayds/Scikit-learn/blob/master/CrossValidation.ipynb)
- [Machine Learning Mastery: SMOTE Oversampling for Imbalanced Classification with Python](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)
