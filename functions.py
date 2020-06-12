#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# IMPORTA FUNCIONES UTILES

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import train_test_split

import lightgbm as lgb
from sklearn import metrics

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# VARIAS

def random_numbers_list(start, stop, n = 1):
	# Genera un numero aleatorio entre dos limites pasados por parametro

	if start == 0 and stop == 1:

		return np.random.random(n)
	else:

		return np.random.randint(start,stop,n)

# Estas dos funciones devuelven las duplas con un cierto nivel de correlación especificado (muy útil cuando se tienen muchas variables)

def get_redundant_pairs(df):
	# Importante, solo tiene que haber datos numericos
    '''Obtengo la diagonal y la parte inferior de la matriz'''

    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, corr = -2):

	# Esta funcion es la que obtiene los pares de variable con una correlación >= corr| correlacion <= -corr
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[(au_corr > corr) | (au_corr < -corr)]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FEATURE ENGINEERING

def feature_engineering(df,categorical,label_encoding = False, dummy = False):

	# Convierto las variables categoricas en numericas segun se quiera o no

	if label_encoding:
		encoder = LabelEncoder()
		
		# Convierto las variables categoricas pasadas en la funcion para codificar
		for feature in categorical:
			encoded = encoder.fit_transform(df[feature])
			df[feature + '_labels'] = encoded
	# Creo variables dummies si asi se desea
	if dummy:

		df = pd.get_dummies(df[categorical])

	return df

def interaction_features(df, categorical):

	import itertools
	# Esta funcion genera interacciones entre todas las variables categoricas, creando nuevas variables del estilo cat1_cat2
	# esto lo hace para cada combinacion de categoricas.

	# Itero para cada par de features y las combino creando nuevas
	
	for col1, col2 in itertools.combinations(categorical, 2):		# Itero cada combinacion posible (lo hace itertools.combinatios, devuelve cada par (o cantidad que yo le indique) posible)
		new_col_name = '_'.join([col1,col2])						# Creo el nombre para la nueva columna
		new_values = df[col1].map(str) + '_' + df[col2].map(str)	# Creo la nueva categorica combinando dos
		encoder = preprocessing.LabelEncoder()						# Creo la instancia LabelEncoder para codificar las nuevas categoricas
		df[new_col_name] = encoder.fit_transform(new_values)		# Transformo los valores

	return df


def count_encoding(X_train, X_test, y_train, y_test, categorical):
	# Esta funcion hace 'Count Encoding'

	# Create the count encoder
	count_enc = ce.CountEncoder(cols = categorical)

	# Learn encoding from the training set
	count_enc.fit(X_train[cat_features])

	# Apply encoding to the train and validation sets as new columns

	X_train = X_train.join(count_enc.transform(X_train[cat_features]).add_suffix('_count'))
	X_test = X_test.join(count_enc.transform(X_test[cat_features]).add_suffix('_count'))

	return X_train, X_test, y_train, y_test

def target_encoding(X_train, X_test, y_train, y_test, categorical):
	# Esta funcion hace 'Target Encoding', calculando la probabilidad de la clase dado ciertos atributos. Solo se aplica sobre el entrenamiento, el encoding luego se aplica al test
	# Creo el coder
	
	target_enc = ce.TargetEncoder(cols=categorical)

	# Hago fit del encoder sobre los datos de entrenamiento unicamente. Si lo hago sobre todos los datos hago data leakeage
	
	target_enc.fit(X_train[categorical], y_train)

	# Aplico el transform
	
	X_train = X_train.join(target_enc.transform(X_train[categorical]).add_suffix('_target'))
	X_test = X_test.join(target_enc.transform(X_test[categorical]).add_suffix('_target'))

	return X_train, X_test, y_train, y_test


def catboost_encoding(X_train, X_test, y_train, y_test, categorical):
	# Esta funcion hace 'Catboost Encoding',similar al target encoding, pero el cálculo de la probabilidad lo hace con las instancias anteriores, no con todo el dataset
	# Creo el coder
	
	target_enc = ce.CatBoostEncoder(cols=categorical)

	# Hago fit del encoder sobre los datos de entrenamiento unicamente. Si lo hago sobre todos los datos hago data leakeage
	
	target_enc.fit(X_train[categorical], y_train)

	# Aplico el transform
	
	X_train = X_train.join(target_enc.transform(X_train[categorical]).add_suffix('_cb'))
	X_test = X_test.join(target_enc.transform(X_test[categorical]).add_suffix('_cb'))

	return X_train, X_test, y_train, y_test

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# FEATURE SELECTION
# Funciones para hacer seleccion de atributos con tecnicas univariables o multivariables

def univariate_selection(X_train,y_train, features, k_feat, method):

	# Esta función compara la correlación que hay entre las features que se le pasan y la variable target
	# de forma individual (uno contra uno). Puedo seleccionar la k features que deseo mantener. Existen distintas
	# metricas para medir la correlacion, por defecto está la 'f_classif' que hace un f score, pero puede ser
	# X^2, ANOVA o mutual information score (que captura relaciones no lineales). Ojo que las tecnicas cambian
	# segun sea clasificacion o regresion

	from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif, f_regression, mutual_info_regression
	# Create the selector, keeping k features
	
	selector = SelectKBest(method, k = k_feat)
	# Use the selector to retrieve the best features
	X_new = selector.fit_transform(X_train[features], y_train) 

	# Get back the kept features as a DataFrame with dropped columns as all 0s
	selected_features = pd.DataFrame(selector.inverse_transform(X_new),
								index = X_train.index,
                                columns = features)

	# Find the columns that must be kept
	keep_columns = selected_features.columns[selected_features.var() != 0]

	return keep_columns

def l1_regularization_selection(X_train,y_train, features, reg_parameter, rand_state):

	# Esta funcion utiliza la regularizacion L1 para seleccionar las mejores features
	# Nota: esta funcion no trae las mejore k features, sino las que quedan seleccionadas como mas relevantes
	# luego de aplicarles la regularizacion

	from sklearn.linear_model import LogisticRegression
	from sklearn.feature_selection import SelectFromModel

	logistic = LogisticRegression(C = reg_parameter, penalty = 'l1', random_state = rand_state).fit(X_train, y_train)
	model = SelectFromModel(logistic, prefit = True)
	X_new = model.transform(X_train)

	selected_features = pd.DataFrame(model.inverse_transform(X_new),
						index = X_train.index,
						columns = X_train.columns)

	cols_to_keep = selected_features.columns[selected_features.var() != 0]

	return cols_to_keep

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FEATURE IMPORTANCE

def permutation_importance(X_train, y_train, X_test, y_test, model, rand_state = 1):

	# Lo que hace esta funcion es aplicar una tecnica para determinar la importancia de las variables
	# segun el siguiente metodo: 
	# Dentro del conjunto de validacion (con un modelo ya entrenado) se mezclan una a una las variables (como en un mazo de cartas)
	# y se predicen los resultados y se evalua que tanto baja la performance. Cuanto mas baje, mas importante era esa variable

	perm = PermutationImportance(model, random_state = rand_state).fit(X_test, y_test)	      										# Creo una instancia PermutationImportance (solo se aplica a testing)
	eli5.show_weights(perm, feature_names = X_test.columns.tolist())    															# Mostrar la importancia de las variables

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# TRAIN TEST SPLIT

def train_test_split_func(df, len_size = 0.2, semilla = 42):

	X = df.drop(columns = 'target', axis = 1, inplace = False)
	y = df['target']

	X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = len_size,random_state = semilla)

	return X_train, X_test, y_train, y_test



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Usando LightGBM

#def light_gm(X_train,y_train,X_test,y_test):
#	# Importante primero separar en tres conjuntos: Train, valid y test

	#feature_cols = [#columnas que se quieran incluir en el modelo]

#	dtrain = lgb.Dataset(X_train[feature_cols], label = y_train)
#	dvalid = lgb.Dataset(X_valid[feature_cols], label = y_valid)

#	param = {'num_leaves': 64, 'objective': 'binary', 
#	             'metric': 'auc', 'seed': 7} # Buscar en la documentacion por el resto de los hiperparametros

#	num_round = 1000		# Cantidad de arboles
#	bst =  lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10) # Entreno el modelo con los parámetros especificados

	# Evaluo el modelo entrenado sobre el conjunto de testeo
#	ypred = bst.predict(X_test[feature_cols])
#	score = metrics.roc_auc_score(y_test, ypred)
#	print(f"Validation AUC score: {score:.4f}")
