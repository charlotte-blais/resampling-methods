# Resampling Methods

library(MASS)
library(ISLR)

# Example 1: Validation Sets
# Resampling methods can be useful for estimating test error 
# Here we will estimate the test error of a logistic regression model that uses two features, income and balance, 
# in order to predict the probability of default. We use the validation set approach. 
set.seed(2)
train = sample(1:nrow(Default),.75*nrow(Default))
lrfit=glm(default~income+balance,data=Default, subset=train, family=binomial)
# For classification problems we generally calculate the average misclassification 
# Let's start out with our threshold probability as 0.5. (<0.5 is "No", >0.5 is "Yes")
x=predict(lrfit,Default[-train,],type="response")
pred = ifelse(x>0.5,"Yes","No")
# We get a 2.44% error rate which seems good at first glance, but actually a default only happens 3% of the time so our error rate is not that much better than predicting "No" every  time. 
(1/nrow(Default[-train,]))*sum(Default$default[-train]!=pred)
sum(Default$default[-train]=="Yes")/nrow(Default[-train,])
# Looking more closely we see that at the current threshold we miss 70% of defaulters
table(Default$default[-train],pred)
55/(55+23)
# Let's try lowering the threshold to 0.2 to see if that will help
pred = ifelse(x>0.2,"Yes","No")
#Now we get an overall error rate of 4.5%, but we have decreased the number of defaulters the model misses to 43.6%
(1/nrow(Default[-train,]))*sum(Default$default[-train]!=pred)
sum(Default$default[-train]=="Yes")/nrow(Default[-train,])
table(Default$default[-train],pred)
34/(34+44)
# Try with a different validation set:
set.seed(10)
train2 = sample(1:nrow(Default),.75*nrow(Default))
lrfit2 = glm(default~income+balance,data=Default, subset=train2, family=binomial)
pred2 = ifelse(predict(lrfit2,Default[-train2,],type="response")>0.5,"Yes","No")
# Here we have a 2.76% error rate (compared to 2.4% on previous validation set)
(1/nrow(Default[-train2,]))*sum(Default$default[-train2]!=pred2)
# Try with a different validation set: 
set.seed(25)
train3 = sample(1:nrow(Default),.75*nrow(Default))
lrfit3 = glm(default~income+balance,data=Default, subset=train3, family=binomial)
pred3 = ifelse(predict(lrfit3,Default[-train3,],type="response")>0.5,"Yes","No")
# Here we have a 2.4% error rate 
(1/nrow(Default[-train3,]))*sum(Default$default[-train3]!=pred3)
# Based on our three validation sets we estimate the error rate to be 2.533% when the threshold is 0.5
(.0276+.0244+.024)/3

# Experiment using another feature in the model:
lrfit4 = glm(default~income+balance+student,data=Default, subset=train3, family=binomial)
pred4 = ifelse(predict(lrfit4,Default[-train3,],type="response")>0.5,"Yes","No")
lrfit5 = glm(default~income+balance+student,data=Default, subset=train2, family=binomial)
pred5 = ifelse(predict(lrfit5,Default[-train2,],type="response")>0.5,"Yes","No")
lrfit6 = glm(default~income+balance+student,data=Default, subset=train, family=binomial)
pred6 = ifelse(predict(lrfit6,Default[-train,],type="response")>0.5,"Yes","No")
# Here we get an average 2.67% error rate. 
err4 = (1/nrow(Default[-train3,]))*sum(Default$default[-train3]!=pred4)
err5 = (1/nrow(Default[-train2,]))*sum(Default$default[-train2]!=pred5)
err6 = (1/nrow(Default[-train,]))*sum(Default$default[-train]!=pred6)
(err4+err5+err6)/3
# Let's look at this student predictor more closely to see why our error rate increased:
# Students have a higher balance and a lower income
plot(Default$student,Default$balance, main="Student Balance")
plot(Default$student,Default$income, main="Student Income")
defaultrate = cumsum(Default$balance)
# income was significant in the model without student. When student is included income is no longer significant, but student is. 
summary(lrfit3)
summary(lrfit4)
# Default rate among students according to balance intervals (groups)
groups = seq(0,2500,length=15)
drstudent = c(0)
drnonstud = c(0)
for(i in 1:14){
drstudent = c(drstudent,sum(as.numeric(subset(Default$default,as.numeric(Default$student)==2&Default$balance<=groups[i]&Default$balance>groups[i-1]))-1)/nrow(subset(Default,as.numeric(Default$student)==2&Default$balance<=groups[i]&Default$balance>groups[i-1])))
drnonstud = c(drnonstud,sum(as.numeric(subset(Default$default,as.numeric(Default$student)==1&Default$balance<=groups[i]&Default$balance>groups[i-1]))-1)/nrow(subset(Default,as.numeric(Default$student)==1&Default$balance<=groups[i]&Default$balance>groups[i-1])))
}
# Since students tend to hold higher balances they default more overall. But, at the same balance a non-student is actually more likely to default than a student. We have a confounding effect. 
par(mfrow=c(1,1))
plot(groups,drstudent,type="o")
matlines(groups,drnonstud,type="o",col="red")
drstudentoverall = sum(as.numeric(subset(Default$default,as.numeric(Default$student)==2))-1)/nrow(subset(Default,as.numeric(Default$student)==2))
drnonstudoverall = sum(as.numeric(subset(Default$default,as.numeric(Default$student)==1))-1)/nrow(subset(Default,as.numeric(Default$student)==1))
matlines(groups,rep(drstudentoverall,length(groups)),type="o",lty=3)
matlines(groups,rep(drnonstudoverall,length(groups)),type="o",lty=3,col="red")


# Example 2: Bootstrapping
# Compute estimates for the standard errors of the income and balance logistic regression coefficients using the bootstrap and the standard glm() function.
set.seed(2)
train = sample(1:nrow(Default),.75*nrow(Default))
lrfit=glm(Default$default~income+balance,data=Default, subset=train, family=binomial)
# Coefficient Estimate, Standard Error: (Intercept) -1.173e+01,  5.061e-01 (Income) 2.479e-05  5.776e-06 (Balance) 5.700e-03  2.614e-04
summary(lrfit)
lrfit$coefficients
# Make a function, boot.fn, to give the coefficient standard errors:
boot.fn = function(data,index){
  coefficients(glm(default~income+balance,data=Default,family=binomial,subset=index))
}
# Use boot() function to estimate the standard error of the coefficient estimates using 1000 bootstrap estimates
# note: there is a discrepancy between these coefficient estimates and the ones from the summary() function. This is because the bootstrap method doesn't assume that the observations are fixed 
#       and all variation comes from the error terms nor that the linear model choice is correct which is an assumption the noise variance depends on (and the equation for the standard error of the coefficients uses the noise variance )
library(boot)
boot(Default, boot.fn, R=1000)


# Example 3: Bootstrapping (another example)
# The estimate of the population mean of medv is 22.53281
mean(Boston$medv)
# Compute standard error of the estimated mean of medv: 0.4088611
sd(Boston$medv)/sqrt(nrow(Boston))
# Estimate standard error of population mean estimate using the bootstrap: 0.4022274
boot.fn = function(data,index){
  mean(data[index])
}
boot.est = boot(Boston$medv,boot.fn,R=1000)
# 95% confidence interval for mean of medv: (21.72953, 23.33608)
t.test(Boston$medv)
# 95% confidence interval for mean of medv using bootstrapped estimate: (21.76, 23.33 )
boot.ci(boot.est,conf=0.95,type="norm")

# Example 4: Leave-One-Out Cross Validation (LOOCV)
# We can use the function cv.glm() to do this, but we will do it by hand to see how LOOCV works.
set.seed(2)
error = rep(0,nrow(Weekly))
index = seq(1,nrow(Weekly),length=nrow(Weekly))
for(i in 1:nrow(Weekly)){
  lrfit = glm(Direction~Lag1+Lag2,data=Weekly,subset=index[-i],family="binomial")
  pred = ifelse(predict.glm(lrfit,newdata=Weekly[i,],type="response")>0.5,"Up","Down")
  if(pred != Weekly[i,9])
    error[i] = 1
}
# Compute the error
sum(error)/nrow(Weekly)
set.seed(2)
glmm = glm(Direction~Lag1+Lag2,data=Weekly,family="binomial")
# This is the cost function for a binary classification - pi is the prediction where default = 0.  
cost <- function(Direction,pi=0) mean((pi < 0.5)&Direction==1 | (pi>0.5)&Direction==0)
cv = cv.glm(Weekly,glmm,cost=cost)
# Same result as doing the for loop.
cv$delta[1]




