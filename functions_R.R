# Import librerias

library(readr)
library(papeR)
library(dplyr)
library(ggplot2)
library(ggpubr)
library(plotly)
library(reshape2)
library(tidyr)
library(gridExtra)
library(mvoutlier)
library(caTools)
library(standardize)
library(caret)

# Librerias para modelar
library(rpart)
library(rpart.plot)
library(ROCR)
library(glmnet)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# DATA CLEANING
# Estas funciones sirven para encontrar nulos, limpiarlos, imputarlos, et.

find_nulls <- function(df){
  columnas_nas <- list()
  
  for (i in colnames(df)){
    nans <- nrow(filter(select(df,i), !complete.cases(select(df,i))))
    
    if (nans != 0){
      
      columnas_nas[[i]] <- (nans/nrow(df))*100
    }
    
  }
  return(columnas_nas)
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Log function for values lower than 0

log_function <- function(df,target_val){
  
  # Esta función toma un dataframe y hace una transformación logarítmica sobre
  # las variables numéricas, excluyendo la target
  for (v in setdiff(names(select_if(df, is.numeric)), target_val)) {
    df[[v]] <- log(df[[v]] + 1 - min(df[[v]]))
  }
  
  return(df)
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
scale_function <- function(df,target_val){
  
  # Esta función toma un dataframe y hace una normalizacion de
  # las variables numéricas, excluyendo la target
  for (v in setdiff(names(select_if(df, is.numeric)), target_val)) {
    df[[v]] <- scale(df[[v]])
  }
  
  return(df)
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
predict_function <- function(model,df,objetivo){
  
  pred <- predict(model,df, type = 'prob')
  pred$pred <- factor(if_else(pred$churn >= .5, 'churn','no_churn'))
  pred$obs <- df[,objetivo]
  
  test_results_benchmark <- twoClassSummary(pred, lev = levels(pred$obs))
  return(test_results_benchmark)
}