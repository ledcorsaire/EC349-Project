#EC349 Assignment Rscript 
#ID:2102296

#Clear
cat("\014")  
rm(list=ls())

#Load Packages required.

library(glmnet)
library(ggplot2)
library(tidyverse)
library(jsonlite)
library(syuzhet)
library(tree)
library(rpart)
library(rpart.plot)
library(randomForest)
library(pROC)
library(caret)

setwd("/Users/manas/Desktop/Uni/Study/Year 3/EC349 Data Science for Economists/Assignment")

#Load data required. Smaller versions of datasets used to save computational power and memory usage.

business_data <- stream_in(file("/Users/manas/Desktop/Uni/Study/Year 3/EC349 Data Science for Economists/Assignment/Main Datasets/yelp_academic_dataset_business.json"))  #note that stream_in reads the json lines (as the files are json lines, not json)

review_data  <- load("/Users/manas/Desktop/Uni/Study/Year 3/EC349 Data Science for Economists/Assignment/Small Datasets/yelp_review_small.Rda")

user_data <- load("/Users/manas/Desktop/Uni/Study/Year 3/EC349 Data Science for Economists/Assignment/Small Datasets/yelp_user_small.Rda")


#Make a dataframe for average ratings by user

user_averagestars <- subset(user_data_small, select = c(1,11))

#Use the previous dataframe to merge with the review data 

refined_review_data1 <- left_join(user_averagestars , review_data_small, by = "user_id")

#Remove observations that do not include reviews and only include user average ratings.Final review dataframe is produced. 

refined_review_data<- refined_review_data1 %>% filter(!is.na(stars))


#Split the dataframe into a test and training set

set.seed(1) 


test_index <- sample(1:nrow(refined_review_data), 10000)
test_set <- refined_review_data[test_index, ]
train_set <- refined_review_data[-test_index, ]


#Manipulating training set to include average stars of the business from the business data 

names(business_data)[names(business_data) == "stars"] <- "businessaveragestars"
business_stars  <- subset(business_data, select = c(1,9))

intermediate1 <- subset(train_set, select = c(1,2,4,5,9) )

intermediate2 <- left_join(intermediate1, business_stars, by = "business_id")

##Converting the text reviews into numerical sentiment scores

sentiment_scores_training <- get_sentiment(intermediate2$text, method = "syuzhet")

intermediate2$text <- sentiment_scores_training

names(intermediate2)[names(intermediate2) == "text"] <- "sentiment_scores"


#Remove unwanted items that are not required in further analysis

intermediate2$business_id <- NULL
intermediate2$user_id <- NULL

#Produce final training dataset 

final_training_dataset <- intermediate2

#Configuring final test set using the same methodology as the training set

intermediatetest1 <- subset(test_set, select = c(1,2,4,5,9) )

intermediatetest2 <- left_join(intermediatetest1, business_stars, by = "business_id")

sentiment_scores_test <- get_sentiment(intermediatetest2$text, method = "syuzhet")

intermediatetest2$text <- sentiment_scores_test

names(intermediatetest2)[names(intermediatetest2) == "text"] <- "sentiment_scores"

#Remove unwanted items
intermediatetest2$business_id <- NULL
intermediatetest2$user_id <- NULL

final_test_dataset <- intermediatetest2

#Classifying star ratings as categorical variables 

final_training_dataset$stars <- as.factor(final_training_dataset$stars)
final_test_dataset$stars <- as.factor(final_test_dataset$stars)

#Plot distribution of stars in training dataset
ggplot(final_training_dataset, aes(x=stars)) + 
  geom_bar() +
  labs(title="Distribution of stars", x="Ratings", y="Frequency") +
  theme_minimal()


#Producing a classification tree using RandomForest package 

rforest_stars<- randomForest(stars~ businessaveragestars+ average_stars+sentiment_scores, data =final_training_dataset) 


#Cross Validation to verify model configuration

train_control <- trainControl(method = "cv", number = 10)

rforest_stars_cv <- train(stars ~ businessaveragestars+ average_stars+sentiment_scores, data = final_training_dataset, 
                          method = "rf", 
                          trControl = train_control,
                          ntree = 50)


#Using the model to predict ratings for test set 
predictions <- predict(rforest_stars, newdata = final_test_dataset, type = "class")

#Evaluating accuracy of model

actualstars <- final_test_dataset$stars

accuracy <- sum(predictions == actualstars) / length(actualstars)

#Setting up a confusion matrix to review results and evaluate beyond accuracy 

conf_matrix1 <- confusionMatrix(predictions, final_test_dataset$stars)


#Looking at model OOB error 

rforest_stars[["err.rate"]][nrow(rforest_stars[["err.rate"]]), "OOB"]

#Looking at variable importance and see how much they reduce Gini Index

importance_measures <- importance(rforest_stars)




