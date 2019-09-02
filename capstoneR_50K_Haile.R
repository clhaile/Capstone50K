# Preamble items
# Install standard packages if not present
if(!require(tidyverse)) install.packages("tidyverse") 
if(!require(caret)) install.packages("caret")
if(!require(dslabs)) install.packages("dslabs")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(pdftools)) install.packages("pdftools")
if(!require(lubridate)) install.packages("lubridate")
if(!require(matrixStats)) install.packages("matrixStats")
if(!require(data.table)) install.packages("data.table")
if(!require(InformationValue)) install.packages("InformationValue")
if(!require(ROCR)) install.packages("ROCR")
if(!require(randomForest)) install.packages("randomForest")
if(!require(readr)) install.packages("readr")

# Load standard libraries

library(tidyverse)
library(dslabs)
library(caret)
library(dslabs)
library(tidyverse)
library(pdftools)
library(lubridate)
library(matrixStats)
library(data.table)
library(forcats)
library(InformationValue)
library(ROCR)
library(randomForest)
library(readr)

## load  and view head of data file
## This data is from the Kaggle list of curated datasets
## https://www.kaggle.com/uciml/adult-census-income

adult <- read_csv("https://raw.githubusercontent.com/clhaile/Capstone50K/master/adult.csv")
head(adult)

##Exploratory analysis

## count of >=50K vs <50k
adult %>% group_by(income) %>% summarize(count=n())

##histogram of age vs income, 5 year interval on age
adult %>% ggplot()+aes(x=age,group=income,fill=income)+geom_histogram(binwidth=5)
adult %>% ggplot()+aes(x=age,group=income,fill=income)+geom_histogram(binwidth=5,position="fill")

## summarize the number of workers in each class

adult %>% group_by(workclass) %>% summarize(count=n())

table(adult$workclass,adult$income)

## collapse a few small factors into a single factor
adult$workclass<-adult$workclass %>% fct_collapse(other = c("Never-worked","Without-pay","?"))
table(adult$workclass,adult$income)

## summarize and invesigate effect of variable "fnlwgt"
summary(adult$fnlwgt)

adult %>% ggplot()+aes(x=fnlwgt,group=income,fill=income)+geom_histogram()
adult %>% ggplot()+aes(x=fnlwgt,group=income,fill=income)+geom_histogram(position="fill")


## summarize and invesigate effect of variable education.num
table(adult$education.num,adult$income) 
adult %>% group_by(education) %>% summarize(size=length(income)) %>% arrange(desc(size))
adult %>% ggplot()+aes(x=education.num,group=income,fill=income)+geom_bar()
adult %>% ggplot()+aes(x=education.num,group=income,fill=income)+geom_bar(position="fill")

#investigate marital status
table(adult$marital.status,adult$income)


#reduce levels of marital status
adult$marital.status<-adult$marital.status %>% fct_collapse(married_together = c("Married-AF-spouse","Married-civ-spouse"),not_together=c("Divorced","Married-spouse-absent","Never-married","Separated","Widowed"))
table(adult$marital.status,adult$income)


#investigate occupation
table(adult$occupation,adult$income)
#reduce levels of occupation
adult$occupation<-adult$occupation %>% fct_collapse(Blue_Collar=c("Craft-repair","Farming-fishing","Handlers-cleaners","Machine-op-inspct","Transport-moving"),White_Collar=c("Adm-clerical","Sales","Tech-support","Protective-serv"),Exec_mgr_prof=c("Exec-managerial","Prof-specialty"), Service_other =c("?", "Armed-Forces", "Other-service", "Priv-house-serv"))
table(adult$occupation,adult$income)


adult %>% ggplot()+aes(x=occupation,group=income,fill=income)+geom_bar()
adult %>% ggplot()+aes(x=occupation,group=income,fill=income)+geom_bar(position="fill")

#investigate relationship (are they a husband to someone, etc.)
table(adult$relationship,adult$income)


#investigate race
table(adult$race,adult$income)

#investigate sex
table(adult$sex,adult$income)

#histogram capital gain
summary(adult$capital.gain)
adult %>% ggplot()+aes(x=capital.gain,group=income,fill=income)+geom_histogram(binwidth=10000)

#histogram capital loss
summary(adult$capital.loss)
adult %>% ggplot()+aes(x=capital.loss,group=income,fill=income)+geom_histogram(binwidth=1000)

summary(adult$capital.gain)
sum(adult$capital.gain==0)/length(adult$capital.gain)
summary(adult$capital.loss)
sum(adult$capital.loss==0)/length(adult$capital.loss)


#histogram hours per week
adult %>% ggplot()+aes(x=hours.per.week,group=income,fill=income)+geom_histogram(binwidth=5)

#investigate native country
table(adult$native.country,adult$income)
# reduce factors of native country
adult$native.country[adult$native.country!="United-States"]  <- "Outside_US" 
adult$native.country[adult$native.country=="United-States"] <- "US" 
table(adult$native.country,adult$income)

#reduce the dataset for variables considered in model
adult <- adult %>% select(age,workclass,education.num,marital.status,occupation,race,sex,hours.per.week,native.country,income)
head(adult)

## transform outcome variable income into binary variable with
##   "<=50K" = 0 and ">50K" = 1
# convenient for ROC curve

adult$income_b[adult$income == "<=50K"] <- 0
adult$income_b[adult$income == ">50K"] <- 1
adult$income_b<-as.numeric(adult$income_b)

#### Begin analysis section

### split into training (train_set)/validation(test_set) of 75:25


set.seed(1)
test_index<-createDataPartition(adult$income,times=1,p=0.25,list=FALSE)

train_set<-adult[-test_index, ]
test_set<-adult[test_index, ]



## GLM Binary Logistic Model

default_glm_mod = train(
  form = income ~ age+workclass+education.num+marital.status+occupation+race+sex+hours.per.week+native.country,
  data = train_set,
  trControl = trainControl(method = "cv", number = 5),
  method = "glm",
  family = "binomial"
)

##GLM Model summary
summary(default_glm_mod)


##accuracay
calc_acc = function(actual, predicted) {
  mean(actual == predicted)
}

acc_glm<-calc_acc(actual = test_set$income,
         predicted = predict(default_glm_mod, newdata = test_set))


##confusion Matrix
y_hat_glm<-predict(default_glm_mod, newdata = test_set)

tbg<-table(predicted=y_hat_glm,actual=test_set$income)

## F1 score

f1_glm<-F_meas(factor(y_hat_glm),factor(test_set$income))
f1_glm

## ROC Curve

## predict probabilities rather than binary
p_hat_glm<-predict(default_glm_mod, newdata = test_set, type = "prob")

pr1 <- prediction(p_hat_glm[2], test_set$income_b)
prf_glm <- performance(pr1, measure = "tpr", x.measure = "fpr")
plot(prf_glm)


## KNN model

default_knn_mod = train(
  income ~ age+workclass+education.num+marital.status+occupation+race+sex+hours.per.week+native.country,
  data = train_set,
  method = "knn",
  trControl = trainControl(method = "cv", number = 5),
  preProcess = c("center", "scale"),
  tuneGrid = expand.grid(k = seq(5, 25, by = 2))
)

default_knn_mod$results

default_knn_mod$finalModel

#knn accuracy and F1 score

acc_knn<-calc_acc(actual = test_set$income,
         predicted = predict(default_knn_mod, newdata = test_set))

y_hat_knn<-predict(default_knn_mod, newdata = test_set)

f1_knn<-F_meas(factor(y_hat_knn),factor(test_set$income))
f1_knn

##confusion Matrix for knn

y_hat_knn<-predict(default_knn_mod, newdata = test_set)

table(predicted=y_hat_knn,actual=test_set$income)

## ROC Curve for knn

## predict probabilities rather than binary
p_hat_knn<-predict(default_knn_mod, newdata = test_set, type = "prob")

pr2 <- prediction(p_hat_knn[2], test_set$income_b)
prf_knn <- performance(pr2, measure = "tpr", x.measure = "fpr")
plot(prf_knn)


##Random Forest Model

## convert some of the categorical variables

train_set$sex_f<-factor(train_set$sex)
test_set$sex_f<-factor(test_set$sex)

train_set$race_f<-factor(train_set$race)
test_set$race_f<-factor(test_set$race)                         

train_set$native.country_f<-factor(train_set$native.country)
test_set$native.country_f<-factor(test_set$native.country)

## Random Forest model

rf <- randomForest(as.factor(income) ~ age+workclass+education.num+marital.status+occupation+hours.per.week+sex_f+native.country_f+race_f, data = train_set, ntree = 1000)
rf.pred.prob <- predict(rf, newdata = test_set, type = 'prob')
rf.pred <- predict(rf, newdata = test_set, type = 'class')

# confusion matrix 
tb <- table(rf.pred, test_set$income)
tb

## accuracy and F1 score

acc_rf<-calc_acc(actual = test_set$income,
         predicted = predict(rf, newdata = test_set))
acc_rf

f1_rf<-F_meas(rf.pred,factor(test_set$income))
f1_rf


# ROC curve

p_hat_rf<-as.data.frame(rf.pred.prob)
pr3 <- prediction(p_hat_rf[2], test_set$income_b)
prf_rf <- performance(pr3, measure = "tpr", x.measure = "fpr")
plot(prf_rf)


## summary of all models

acc_results<-bind_rows(
  tibble(method="GLM",Accuracy = acc_glm,F1=f1_glm),
  tibble(method="KNN",Accuracy = acc_knn,F1=f1_knn),
  tibble(method="Random Forest",Accuracy = acc_rf,F1=f1_rf))

acc_results %>% knitr::kable()

## graphs of ROC curves together

dd1 <- data.frame(FP = prf_glm@x.values[[1]], TP = prf_glm@y.values[[1]])
dd2 <- data.frame(FP = prf_knn@x.values[[1]], TP = prf_knn@y.values[[1]])
dd3 <- data.frame(FP = prf_rf@x.values[[1]], TP = prf_rf@y.values[[1]])

g <- ggplot() + 
  geom_line(data = dd1, aes(x = FP, y = TP, color = 'GLM')) + 
  geom_line(data = dd2, aes(x = FP, y = TP, color = 'KNN')) + 
  geom_line(data = dd3, aes(x = FP, y = TP, color = 'Random Forest')) +
  ggtitle('ROC Curve') + 
  labs(x = 'False Positive Rate', y = 'True Positive Rate') 
g

# Area under curve computation and table
auc <- rbind(
             performance(pr1, measure = 'auc')@y.values[[1]],
             performance(pr2, measure = 'auc')@y.values[[1]],
             performance(pr3, measure = 'auc')@y.values[[1]])
             
rownames(auc) <- (c('GLM', 'KNN', 
                    'Random Forest'))
colnames(auc) <- 'Area Under ROC Curve'
round(auc, 4)
