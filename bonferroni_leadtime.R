# https://www.statology.org/bonferroni-correction-in-r/

# Compare lead time and ADR

data<-read.csv("H1_random_sampling.csv")
attach(data)

#view first six rows of data frame
head(data)

boxplot(ADR ~ DistributionChannel,
        data = data,
        main = "ADR By Group",
        xlab = "Distribution Channel",
        ylab = "ADR",
        col = "green",
        border = "black")

#fit the one-way ANOVA model
model <- aov(ADR ~ DistributionChannel, data = data) # low p-value: lead time not equal among groups

#view model output
summary(model)

#perform pairwise t-tests with Bonferroni's correction
pairwise.t.test(data$ADR, data$DistributionChannel, p.adjust.method="bonferroni")