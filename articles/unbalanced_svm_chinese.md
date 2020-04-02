[Home](https://mgcodesandstats.github.io/) |
[Medium](https://medium.com/@firstclassanalyticsmg) |
[LinkedIn](https://www.linkedin.com/in/michaeljgrogan/) |
[GitHub](https://github.com/mgcodesandstats) |
[Speaking Engagements](https://mgcodesandstats.github.io/speaking-engagements/) |
[Terms](https://mgcodesandstats.github.io/terms/) |
[E-mail](mailto:contact@michael-grogan.com)

# 分类和: 支持向量机 (SVM)

*This is a translated summary of the original article (in my best Chinese!): [Imbalanced Classes: Predicting Hotel Cancellations with Support Vector Machines](https://www.michael-grogan.com/hotel-modelling/articles/unbalanced_svm)*

不平衡的种类是一个问题. 为什么？

在Jupyter Notebook中执行SVM时, 次要分类比重大的分类不相等的。

所以呢，我们要用SVM跟一个 class_weight='balanced'，为了平等对待这两个分类。

在这个例子中要想预测客户是否会取消他们的酒店预订。

消除是次要分类, 不消除是重大分类。

以下是变量：

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

# 什么是SVM？

SVM是可用于分类和回归任务的监督学习模型。

当定义两个班级之间的决策极限时，SVM模型可以评估每个训练点的重要性。

！[svm-plane.png](svm-plane.png)

所选的位于两个类别之间的决策边界上的几个训练点称为支持向量。

## "Precision" 和 "Recall" 和 "f1-score"

**Precision** und **Recall** 计算为:

```
Precision = ((真阳性)/(真阳性 + 假阳性))

Recall = ((真阳性)/(真阳性 + 假阴性))
```

Precision的增加通常会导致Recall的减少，反之亦然。

在这种情况下哪个最好？

说实话假阳性在这个情况下, 因为我们要发现什么客户有取消风险。

第H1资料集是我们的训练集.

```
y1 = y
x1 = np.column_stack((leadtime,countrycat,marketsegmentcat,deposittypecat,customertypecat,rcps,arrivaldateyear,arrivaldatemonthcat,arrivaldateweekno,arrivaldatedayofmonth))
x1 = sm.add_constant(x1, prepend=True)

x1_train, x1_val, y1_train, y1_val = train_test_split(x1, y1, random_state=0)

from sklearn import svm
clf = svm.SVC(gamma='scale', 
            class_weight='balanced')
clf.fit(x1_train, y1_train)  
prclf = clf.predict(x1_val)
prclf
```

这是验证集的结果:

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

这是测试集的结果 (第H2资料集)

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

第H2资料集的 "f1-score accuracy"降至63％, 但是"recall"增加到75％。

就是说, SVM确定了取消酒店预订的所有客户中的75％。

## 参考

数据集和笔记本可在此处获得: [MGCodesandStats GitHub Repository](https://github.com/MGCodesandStats/hotel-modelling).

另一个有用的参考: [Elite Data Science: How to Handle Imbalanced Classes in Machine Learning](https://elitedatascience.com/imbalanced-classes)