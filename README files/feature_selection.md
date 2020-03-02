# Feature Selection and Classification Methods: Predicting Hotel Cancellations

One of the most important features (no pun intended) of a machine learning project is feature selection.

Feature selection allows for identification of the most important or influencing factors on the response (or dependent) variable. In this example, feature selection techniques are used to predict the most important influencing factors on whether a customer chooses to cancel their hotel booking or not.

## Feature Selection Tools

Three different feature selection tools are used to analyse this dataset:

- **ExtraTreesClassifier:** The purpose of the ExtraTreesClassifier is to fit a number of randomized decision trees to the data, and in this regard is a from of ensemble learning. Particularly, random splits of all observations are carried out to ensure that the model does not overfit the data.

- **Step forward and backward feature selection**: This is a **"wrapper-based"** feature selection method, where the feature selection is based on a specific machine learning algorithm (in this case, the RandomForestClassifier). For forward-step selection, each individual feature is added to the model one at a time, and the features with the highest ROC_AUC score are selected as the best features. When conducting backward feature selection, this process happens in reverse - whereby each feature is dropped from the model one at a time, i.e. the features with the lowest ROC_AUC scores are dropped from the model.

## Background and Data Manipulation

The purpose of using these algorithms is to identify features that best help to predict whether a customer will cancel their hotel booking. This is the dependent variable, where (1 = cancel, 0 = follow through with booking).

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

With regard to these features, certain features such as lead time are **interval** - in other words they can take on a wide range of values and are not necessarily constrained by a particular scale.

However, certain variables such as **customertype** are categorical variables. In this regard, *cat.codes* is used to identify these variables as categorical and ensure that they are not erronously ranked in the eventual analysis. As an example, consider the following variable: 1 = apple, 2 = banana, 3 = orange. This variable is categorical and the numbers have no inherent rank - therefore it is important to specify as such.

In this regard - using **customertype** as an example, the variable is first converted to categorical and then stored as a pandas Series:

```
customertypecat=train_df.CustomerType.astype("category").cat.codes
customertypecat=pd.Series(customertypecat)
```
The IsCanceled variable is the response variable:

```
IsCanceled = train_df['IsCanceled']
y = IsCanceled
```

Once the features have been loaded into Python, they are then stored as a numpy stack (or a sequence of arrays):

```
x = np.column_stack((leadtime,arrivaldateyear,arrivaldatemonthcat,arrivaldateweekno,arrivaldatedayofmonth,staysweekendnights,staysweeknights,adults,children,babies,mealcat,countrycat,marketsegmentcat,distributionchannelcat,isrepeatedguest,previouscancellations,previousbookingsnotcanceled,reservedroomtypecat,assignedroomtypecat,bookingchanges,deposittypecat,dayswaitinglist,customertypecat,adr,rcps,totalsqr,reservationstatuscat))
x = sm.add_constant(x, prepend=True)
```

Now that the **x** and **y** variables have been defined, the feature selection methods are used to identify which variables have the greatest influence on hotel cancellations.

Specifically, once the relevant features have been identified, the SVM (support vector machines) model is used for classification. The identified features from the three techniques outlined above are fed separately into the model to determine which feature selection tool is doing the best job at identifying the important features - which is assumed to be reflected by a higher AUC score.

## ExtraTreesClassifier

The ExtraTreesClassifier is generated:

```
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x, y)
print(model.feature_importances_)
```
Here are the results:

```
[0.00000000e+00 2.48809571e-02 5.38765461e-03 2.83449418e-03
 3.79810248e-03 3.47059669e-03 3.13400622e-03 3.76015081e-03
 2.55919654e-03 2.61790593e-03 2.87548922e-04 2.75263105e-03
 3.71420453e-02 1.84569752e-02 5.83394041e-03 4.33738703e-03
 8.51023083e-03 6.71433130e-04 3.82846350e-03 7.24333726e-03
 4.09623483e-03 3.40961337e-02 5.06608127e-04 1.07736457e-02
 7.16088830e-03 1.98534730e-02 6.99116062e-03 7.75014798e-01]
 ```

Let's sort this into a data frame and take a look at the top features:

```
ext=pd.DataFrame(model.feature_importances_,columns=["extratrees"])
ext
ext.sort_values(['extratrees'], ascending=True)
```

![image1amended.png](https://github.com/MGCodesandStats/hotel-modelling/blob/master/images/image1amended.png)

The top identified features are features 1, 12, 13, 21, 23, 25 (lead time, country of origin, market segment, deposit type, customer type, and required car parking spaces). Note that feature **27** (reservation status) is not valid in this case, since this effectively represents the same thing as the response variable - i.e. whether a customer cancelled or followed through with their booking. In this case, including the feature in the analysis would be erroneous.

## Step forward and backward feature selection

As previously described, this feature selection method is based on the RandomForestClassifier. In terms of step forward feature selection, the ROC_AUC score is assessed for each feature as it is added to the model, i.e. the features with the highest scores are added to the model. For step backward feature selection, the process is reversed - features are dropped from the model based on those with the lowest ROC_AUC scores. The top six features are being selected from the dataset using this feature selection tool.

The forward feature selection is implemented as follows:

```
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector

forward_feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1),
           k_features=6,
           forward=True,
           verbose=2,
           scoring='roc_auc',
           cv=4)
           
fselector = forward_feature_selector.fit(x, y)                     
```

Here is the generated output:

```
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    2.0s remaining:    0.0s
[Parallel(n_jobs=1)]: Done  28 out of  28 | elapsed:   40.8s finished

[2020-03-01 19:01:14] Features: 1/6 -- score: 1.0[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    1.3s remaining:    0.0s
[Parallel(n_jobs=1)]: Done  27 out of  27 | elapsed:   37.4s finished

[2020-03-01 19:01:52] Features: 2/6 -- score: 1.0[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    1.5s remaining:    0.0s
[Parallel(n_jobs=1)]: Done  26 out of  26 | elapsed:   37.3s finished

...

[2020-03-01 19:03:49] Features: 5/6 -- score: 1.0[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    1.9s remaining:    0.0s
[Parallel(n_jobs=1)]: Done  23 out of  23 | elapsed:   40.7s finished

[2020-03-01 19:04:30] Features: 6/6 -- score: 1.0
```

We can identify the feature names (or numbers in this case, as they are stored in the array) as follows:

```
>>> fselector.k_feature_names_
('0', '1', '2', '3', '4', '27')
```

The backward feature selection method is more computationally-intensive, as all features in the dataset are being considered.

We implement this by simply setting *forward=False*.

```
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score

backward_feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1),
           k_features=6,
           forward=False,
           verbose=2,
           scoring='roc_auc',
           cv=4)
           
bselector = backward_feature_selector.fit(x, y)
```
Here is the generated output:

```
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    3.4s remaining:    0.0s
[Parallel(n_jobs=1)]: Done  28 out of  28 | elapsed:  1.1min finished

[2020-03-01 19:05:39] Features: 27/6 -- score: 1.0[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    3.2s remaining:    0.0s
[Parallel(n_jobs=1)]: Done  27 out of  27 | elapsed:  1.0min finished

...

[2020-03-01 19:17:46] Features: 7/6 -- score: 1.0[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    2.7s remaining:    0.0s
[Parallel(n_jobs=1)]: Done   7 out of   7 | elapsed:   13.0s finished

[2020-03-01 19:17:59] Features: 6/6 -- score: 1.0
```

Here are the identified features:

```
>>> bselector.k_feature_names_
('0', '1', '3', '4', '5', '27')
```

As we can see, the identified features are the same as for the forward feature selection. The ExtraTreesClassifier also identified feature no. 1 (lead time) as an important feature, while this method identified features 3, 4, 5 (arrivaldatemonth, arrivaldateweekno, arrivaldatedayofmonth).

In this regard, this feature selection method is indicating the time features in the dataset are of greater importance than the ExtraTreesClassifier method is suggesting.



## Conclusion

To summarize, we have looked at:

- Feature selection using ExtraTreesClassifier
- Forward and backward feature selection methods
- Use of SVM with balanced classes to predict cancellation incidences
- Assessment of prediction accuracy using AUC

Many thanks for your time, and you can also find the relevant GitHub repository for this example [here](https://github.com/MGCodesandStats/feature-selection).

## Useful References

- [Feature selection using Wrapper methods in Python](https://towardsdatascience.com/feature-selection-using-wrapper-methods-in-python-f0d352b346f)

- [How to Perform Feature Selection with Categorical Data](https://machinelearningmastery.com/feature-selection-with-categorical-data/)

- [ANOVA F-value For Feature Selection](https://chrisalbon.com/machine_learning/feature_selection/anova_f-value_for_feature_selection/)
