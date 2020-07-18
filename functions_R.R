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
library(randomForest)
library(xgboost)

# Librerias de feature importance
library(vip)
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

one_hot_sparse <- function(data_set) {
  
  require(Matrix)
  
  created <- FALSE
  
  if (sum(sapply(data_set, is.numeric)) > 0) {  # Si hay, Pasamos los numéricos a una matriz esparsa (sería raro que no estuviese, porque "Label"  es numérica y tiene que estar sí o sí)
    out_put_data <- as(as.matrix(data_set[,sapply(data_set, is.numeric), with = FALSE]), "dgCMatrix")
    created <- TRUE
  }
  
  if (sum(sapply(data_set, is.logical)) > 0) {  # Si hay, pasamos los lógicos a esparsa y lo unimos con la matriz anterior
    if (created) {
      out_put_data <- cbind2(out_put_data,
                             as(as.matrix(data_set[,sapply(data_set, is.logical),
                                                   with = FALSE]), "dgCMatrix"))
    } else {
      out_put_data <- as(as.matrix(data_set[,sapply(data_set, is.logical), with = FALSE]), "dgCMatrix")
      created <- TRUE
    }
  }
  
  # Identificamos las columnas que son factor (OJO: el data.frame no debería tener character)
  fact_variables <- names(which(sapply(data_set, is.factor)))
  
  # Para cada columna factor hago one hot encoding
  i <- 0
  
  for (f_var in fact_variables) {
    
    f_col_names <- levels(data_set[[f_var]])
    f_col_names <- gsub(" ", ".", paste(f_var, f_col_names, sep = "_"))
    j_values <- as.numeric(data_set[[f_var]])  # Se pone como valor de j, el valor del nivel del factor
    
    if (sum(is.na(j_values)) > 0) {  # En categóricas, trato a NA como una categoría más
      j_values[is.na(j_values)] <- length(f_col_names) + 1
      f_col_names <- c(f_col_names, paste(f_var, "NA", sep = "_"))
    }
    
    if (i == 0) {
      fact_data <- sparseMatrix(i = c(1:nrow(data_set)), j = j_values,
                                x = rep(1, nrow(data_set)),
                                dims = c(nrow(data_set), length(f_col_names)))
      fact_data@Dimnames[[2]] <- f_col_names
    } else {
      fact_data_tmp <- sparseMatrix(i = c(1:nrow(data_set)), j = j_values,
                                    x = rep(1, nrow(data_set)),
                                    dims = c(nrow(data_set), length(f_col_names)))
      fact_data_tmp@Dimnames[[2]] <- f_col_names
      fact_data <- cbind(fact_data, fact_data_tmp)
    }
    
    i <- i + 1
  }
  
  if (length(fact_variables) > 0) {
    if (created) {
      out_put_data <- cbind(out_put_data, fact_data)
    } else {
      out_put_data <- fact_data
      created <- TRUE
    }
  }
  return(out_put_data)
}