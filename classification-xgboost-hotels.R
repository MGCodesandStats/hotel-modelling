# Reference: https://xgboost.readthedocs.io/en/latest/R-package/xgboostPresentation.html 

# TRAINING AND VALIDATION

require(xgboost)
library(Matrix)

H1<-read.csv("H1.csv")
H1

class(H1$Country)
class(H1$LeadTime)
class(H1$MarketSegment)
class(H1$DepositType)

IsCanceled<-as.numeric(factor(H1$IsCanceled))
IsCanceled
IsCanceled[IsCanceled == "1"] <- "0"
IsCanceled[IsCanceled == "2"] <- "1"
IsCanceled<-as.numeric(IsCanceled)
IsCanceled

leadtime<-as.numeric(H1$LeadTime)
leadtime
country<-as.numeric(factor(H1$Country))
country
marketsegment<-as.numeric(factor(H1$MarketSegment))
marketsegment
deposittype<-as.numeric(factor(H1$DepositType))
deposittype
customertype<-as.numeric(factor(H1$CustomerType))
customertype
rcps<-as.numeric(H1$RequiredCarParkingSpaces)
rcps
week<-as.numeric(H1$ArrivalDateWeekNumber)
week

df<-data.frame(leadtime,country,marketsegment,deposittype,customertype,rcps,week)
attach(df)
df<-as.matrix(df)

train <- df[1:32000,]
val <- df[32001:40060,]

train=as(train, "dgCMatrix")
train

val=as(val, "dgCMatrix")
val

IsCanceled_train=IsCanceled[1:32000]
IsCanceled_val=IsCanceled[32001:40060]

bst <- xgboost(data = train, label = IsCanceled_train, max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
pred <- predict(bst, val)


print(length(pred))
print(head(pred))

prediction <- as.numeric(pred > 0.5)
print(head(prediction))

err <- mean(as.numeric(pred > 0.5) != IsCanceled_val)
print(paste("val-error=", err))

# Importance Matrix
importance_matrix <- xgb.importance(model = bst)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix)



# TEST

H2<-read.csv("H2.csv")
H2

leadtime<-as.numeric(H2$LeadTime)
leadtime
country<-as.numeric(factor(H2$Country))
country
marketsegment<-as.numeric(factor(H2$MarketSegment))
marketsegment
deposittype<-as.numeric(factor(H2$DepositType))
deposittype
customertype<-as.numeric(factor(H2$CustomerType))
customertype
rcps<-as.numeric(H2$RequiredCarParkingSpaces)
rcps
week<-as.numeric(H2$ArrivalDateWeekNumber)
week

test<-data.frame(leadtime,country,marketsegment,deposittype,customertype,rcps,week)
attach(test)
test<-as.matrix(test)

test=as(test, "dgCMatrix")
test

IsCanceled_H2<-as.numeric(factor(H2$IsCanceled))
IsCanceled_H2
IsCanceled_H2[IsCanceled_H2 == "1"] <- "0"
IsCanceled_H2[IsCanceled_H2 == "2"] <- "1"
IsCanceled_H2<-as.numeric(IsCanceled_H2)
IsCanceled_H2

pred <- predict(bst, test)

print(length(pred))
print(head(pred))

prediction <- as.numeric(pred > 0.5)
print(head(prediction))

err <- mean(as.numeric(pred > 0.5) != IsCanceled_H2)
print(paste("test-error=", err))