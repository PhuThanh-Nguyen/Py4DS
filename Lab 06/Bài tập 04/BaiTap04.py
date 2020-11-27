'''
	Python for Data Science
	Lab 6: Feature Selection vs Dimensionality Reduction - Excercise 04
	
	Using Santander_train.csv, Santander_test.csv
	Perform data dimensionality reduction and feature selection (at least 5 techniques)
	Build model
'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def ConstantFeatures(X_train, X_test, threshold = 0):
  '''
	Removing Constant Features using Variance Threshold
	------------------
	Parameters:
		threshold: float, default = 0
			threshold parameter to identify the variable as constant
		X_train, X_test: DataFrame object
			Training and testing data to evaluate 
	------------------
	Returns: 
		train data, test data after applying filter methods
  '''
  # Create the VarianceThreshold object.
  vs_constant = VarianceThreshold(threshold)

  # Select the numerical columns only.
  numerical_x_train = X_train[X_train.select_dtypes([np.number]).columns]

  # Fit the object to our data.
  vs_constant.fit(numerical_x_train)

  # Get the constant colum names.
  constant_columns = [column for column in numerical_x_train.columns
                      if column not in numerical_x_train.columns[vs_constant.get_support()]]

  # Detect constant categorical variables.
  constant_cat_columns = [column for column in X_train.columns \
                          if (X_train[column].dtype == "O" and len(X_train[column].unique())  == 1 )]

  # Concatenating the two lists.
  all_constant_columns = constant_cat_columns + constant_columns

  # Drop the constant columns
  X_train = X_train.drop(labels = all_constant_columns, axis = 1)
  X_test = X_test.drop(labels = all_constant_columns, axis = 1)
  
  return X_train, X_test

def correlationFilterMethod(X_train, X_test, thres = 0.8):
	'''
		Perform Correlation filter method
		------------------
		Parameters:
			X_train, X_test: DataFrame objects
				Training and testing data
			thres: float, default: 0.8
				Threshold of absolute value of correlation, must between 0 and 1
		------------------
		Returns:
			X_train, X_test after applying correlation filter method
	'''
	# Creating set to hold the correlated features
	corr_features = set()

	# Create the correlation matrix (default to pearson)
	corr_matrix = X_train.corr()

	for i in range(len(corr_matrix.columns)):
		for j in range(i):
			if abs(corr_matrix.iloc[i, j]) > thres:
				colname = corr_matrix.columns[i]
				corr_features.add(colname)
				
	X_train = X_train.drop(labels = corr_features, axis = 1)
	X_test = X_test.drop(labels = corr_features, axis = 1)
	return X_train, X_test

def PCAMethod(num_components, X_train, X_test):
	'''
		Perform Principle Component Analysis (PCA) Method
		------------------
		Parameters:
			num_components:	int
				Number of components to keep
			X_train, X_test: DataFrame objects
		------------------
		Returns:
			Training and testing data after applying PCA Method
	'''
	pca = PCA(n_components = num_components)
	pca.fit(X_train)
	return pca.transform(X_train), pca.transform(X_test)

def BackwardFeatureElimination(X_train, y_train, X_test, y_test, clf):
	
	'''
		Perform backward feature elimination
		------------------
		Parameters:
			X_train, y_train, X_test, y_test: DataFrame object
				Training and testing data
			clf: sklearn classification model
				Model to train, test and calculate error
		------------------
		Returns:
			X_train, X_test after applying backward feature elimination algorithm
	'''
	
	def errorCalculate(columns):
		'''
			A wrapper function to calculate error on training, testing data on given columns
			------------------
			Parameters:
				columns: list-like
					Set of columns to calculate error
			------------------
			Returns:
				Error (1 - accuracy_score) when train model using such columns 
		'''
		X_train_col = X_train[columns]
		X_test_col = X_test[columns]
		
		clf.fit(X_train_col, y_train)
		y_pred = clf.predict(X_test_col)
		
		return 1 - accuracy_score(y_test, y_pred)
	
	min_err = errorCalculate(X_train.columns)
	cols = list(X_train.columns)
	
	for column in X_train.columns:
		temp = cols.copy()
		temp.remove(column)
		
		err = errorCalculate(temp)
		
		if err < min_err:
			min_err = err
		else:
			return X_train[cols], X_test[cols]
		cols = temp
	return X_train, X_test

def ForwardFeatureElimination(X_train, y_train, X_test, y_test, clf):
	
	'''
		Perform forward feature elimination
		------------------
		Parameters:
			X_train, y_train, X_test, y_test: Data frame
				Training and testing data
			clf: sklearn classification model
				Model to train, test and calculate error
		------------------
		Returns:
			X_train, X_test after applying forward feature elimination algorithm
	'''
	
	def errorCalculate(columns):
		'''
			A wrapper function to calculate error on training, testing data on given columns
			------------------
			Parameters:
				columns: list-like
					Set of columns to calculate error
			------------------
			Returns:
				Error (1 - accuracy_score) when train model using such columns 
		'''
		X_train_col = X_train[columns]
		X_test_col = X_test[columns]
		
		clf.fit(X_train_col, y_train)
		y_pred = clf.predict(X_test_col)
		
		return 1 - accuracy_score(y_test, y_pred)
	
	# Start with empty set of columns
	min_err, cols, i = 0, [], 0
	max_cols = list(X_train.columns)
	
	while cols != max_cols:
		temp = cols.copy()
		temp.append(X_train.columns[i])
		err = errorCalculate(temp)
		
		if err < min_err and min_err != 0:
			min_err = err
		elif min_err == 0:
			min_err = err
		elif err > min_err and min_err != 0:
			return X_train[cols], X_test[cols]
		i += 1
		cols = temp
	return X_train[max_cols], X_test[max_cols]

def decisionTreeModel(X_train, X_dev, y_train, y_dev):
	'''
	Performs Decision Tree model using sklearn library
	-----------------
	Parameters:
		X_train, X_dev, y_train, y_dev: array-like
			Training and development data
	-----------------
	Returns:
		A tuple (accuracy, score) where `accuracy` is the accuracy of model based on development data and
		`score` is the score of model based on training data
	'''
	clf = DecisionTreeClassifier(random_state = 0)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_dev)
	return accuracy_score(y_dev, y_pred), clf.score(X_train, y_train) 

def main():
	########################## Exploratory Data Analysis (EDA) ###############################
	data_train, data_test = pd.read_csv('../dataset/Santander_train.csv'), pd.read_csv('../dataset/Santander_test.csv')
	
	print('#'*50 + ' ORIGINAL DATA TRAIN ' + '#'*50)
	print('>'*10 + f' Data shape: {data_train.shape}')
	print('>'*10 + ' Data head:')
	print(data_train.head())
	print('>'*10 + ' Data info:')
	print(data_train.info())
	print('>'*10 + ' Data describe:')
	print(data_train.describe())

	print('#'*50 + ' ORIGINAL DATA TEST ' + '#'*50)
	print('>'*10 + f' Data test shape: {data_test.shape}')
	print('>'*10 + ' Data test head:')
	print(data_test.head())
	print('>'*10 + ' Data test info:')
	print(data_test.info())
	print('>'*10 + ' Data test describe:')
	print(data_test.describe())

	# Plot label of data train
	# Unbalanced between two labels in data train
	fig = plt.figure(figsize=(10, 8))
	ax = data_train['TARGET'].value_counts(normalize=True).plot(kind='bar')
	fig.suptitle('Barplot of Target (label)', fontsize=16)
	fig.savefig('Barplot-target.png')

	# Checking for NaNs or empty cells by features of data train
	print('#'*50 + f' MISSING VALUE PERCENTAGE OF DATA TRAIN' + '#'*50)
	percent_missing = data_train.isnull().sum() * 100 / len(data_train)
	missing_value_df_train = pd.DataFrame({'column_name': data_train.columns,
		                                   'percent_missing': percent_missing})
	print(missing_value_df_train.sort_values(by = 'percent_missing', ascending = False))
	print('#'*50 + ' DATA TRAIN INFOMATION BEFORE HANDLING MISSING VALUES ' + '#'*50)
	print('>'*10 + f' Data shape: {data_train.shape}')
	print('>'*10 + ' Data train head:')
	print(data_train.head())
	print('>'*10 + ' Data train info:')
	print(data_train.info())
	print('>'*10 + ' Data train describe:')
	print(data_train.describe())

	# Checking for NaNs or empty cells by features of data train
	print('#'*50 + f' MISSING VALUE PERCENTAGE OF DATA TEST' + '#'*50)
	percent_missing = data_test.isnull().sum() * 100 / len(data_test)
	missing_value_df_test = pd.DataFrame({'column_name': data_test.columns,
		                                  'percent_missing': percent_missing})
	print(missing_value_df_test.sort_values(by = 'percent_missing', ascending = False))
	print('#'*50 + ' DATA TEST INFOMATION BEFORE HANDLING MISSING VALUES ' + '#'*50)
	print('>'*10 + f' Data test shape: {data_test.shape}')
	print('>'*10 + ' Data test head:')
	print(data_test.head())
	print('>'*10 + ' Data test info:')
	print(data_test.info())
	print('>'*10 + ' Data test describe:')
	print(data_test.describe())
	# There is no missing value in both data train and data test
	
	# Checking for duplicate values
	print('#'*50 + ' CHECKING DUPLICATE VALUES ON DATA TRAIN ' + '#'*50)
	duplicateRowsTrain = data_train[data_train.duplicated()]
	total_duplicates = len(duplicateRowsTrain)
	print(
		f"Duplicate Rows (Data Train) (total: {total_duplicates}) except first occurrence based on all columns are :")
	print(duplicateRowsTrain)
	print('#'*50 + ' DATA TRAIN INFO BEFORE DROPPING DUPLICATE VALUES ' + '#'*50)
	print('>'*10 + f' Data train shape: {data_train.shape}')
	print('>'*10 + ' Data train info:')
	print(data_train.info())

	print('#'*50 + ' CHECKING DUPLICATE VALUES ON DATA TEST ' + '#'*50)
	duplicateRowsTest = data_test[data_test.duplicated()]
	total_duplicates = len(duplicateRowsTest)
	print(
		f"Duplicate Rows (Data Test) (total: {total_duplicates}) except first occurrence based on all columns are :")
	print(duplicateRowsTest)
	print('#'*50 + ' DATA TEST INFO BEFORE DROPPING DUPLICATE VALUES ' + '#'*50)
	print('>'*10 + f' Data test shape: {data_test.shape}')
	print('>'*10 + ' Data test info:')
	print(data_test.info())

	# Drop duplicate values on data train and data test
	data_train.drop_duplicates(
		subset=data_train.columns.values[:-1], keep='first', inplace=True)
	print('#'*50 + ' DATA TRAIN INFO AFTER DROPPING DUPLICATE VALUES ' + '#'*50)
	print('>'*10 + f' Data train shape: {data_train.shape}')
	print('>'*10 + ' Data train info:')
	print(data_train.info())

	data_test.drop_duplicates(
		subset=data_test.columns.values[:-1], keep='first', inplace=True)
	print('#'*50 + ' DATA TEST INFO AFTER DROPPING DUPLICATE VALUES ' + '#'*50)
	print('>'*10 + f' Data test shape: {data_test.shape}')
	print('>'*10 + ' Data test info:')
	print(data_test.info())

	########################## Feature Selection & Dimensionality Reduction ###############################
	X, y = data_train.drop(['TARGET'], axis = 1), data_train['TARGET']
	X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size = 0.25, random_state = 0)

	X_test = data_test

	# Using ConstantFeatures method
	X_train_temp, X_dev_temp = ConstantFeatures(X_train, X_dev)
	accuracy, score = decisionTreeModel(X_train_temp, X_dev_temp, y_train, y_dev)
	print('#' * 50 + ' USING CONSTANT FEATURES METHOD ' + '#' * 50)
	print(f'>> Accuracy using Constant Features method = {accuracy}')
	print(f">> Model's score using Constant Features method = {score}")

	# Using Correlation Filter Method
	X_train_temp, X_dev_temp = correlationFilterMethod(X_train, X_dev, thres = 0.8)
	accuracy, score = decisionTreeModel(X_train_temp, X_dev_temp, y_train, y_dev)
	print('#' * 50 + ' USING CORRELATION FILTER METHOD ' + '#' * 50)
	print(f'>> Accuracy using Correlation Filter method = {accuracy}')
	print(f">> Model's score using Correlation Filter method = {score}")

	# Using PCA
	X_train_temp, X_dev_temp = PCAMethod(10, X_train, X_dev)
	accuracy, score = decisionTreeModel(X_train_temp, X_dev_temp, y_train, y_dev)
	print('#' * 50 + ' USING PCA ' + '#' * 50)
	print(f'>> Accuracy using PCA = {accuracy}')
	print(f">> Model's score using PCA = {score}")

	# Using Backward Feature Elimination
	clf = DecisionTreeClassifier(random_state = 0)
	X_train_temp, X_dev_temp = BackwardFeatureElimination(X_train, y_train, X_dev, y_dev, clf)
	accuracy, score = decisionTreeModel(X_train_temp, X_dev_temp, y_train, y_dev)
	print('#' * 50 + ' USING BACKWARD FEATURE ELIMINATION ' + '#' * 50)
	print(f'>> Accuracy using Backward Feature Elimination = {accuracy}')
	print(f">> Model's score using Backward Feature Elimination = {score}")
	
	# Using Forward Feature Elimination
	clf = DecisionTreeClassifier(random_state = 0)
	X_train_temp, X_dev_temp = ForwardFeatureElimination(X_train, y_train, X_dev, y_dev, clf)
	accuracy, score = decisionTreeModel(X_train_temp, X_dev_temp, y_train, y_dev)
	print('#' * 50 + ' USING FORWARD FEATURE ELIMINATION ' + '#' * 50)
	print(f'>> Accuracy using Forward Feature Elimination = {accuracy}')
	print(f">> Model's score using Forward Feature Elimination = {score}")
	
	'''
	With random state = 0
		1. USING CONSTANT FEATURES METHOD
	>> Accuracy using Constant Features method = 0.924177847934754
	>> Model's score using Constant Features method = 1.0
		2. USING CORRELATION FILTER METHOD
	>> Accuracy using Correlation Filter method = 0.9263877926861352
	>> Model's score using Correlation Filter method = 1.0
		3. USING PCA
	>> Accuracy using PCA = 0.9322283609576427
	>> Model's score using PCA = 0.9870560378847671
		4. USING BACKWARD FEATURE ELIMINATION
	>> Accuracy using Backward Feature Elimination = 0.9272822941331229
	>> Model's score using Backward Feature Elimination = 0.9959308953784092
		5. USING FORWARD FEATURE ELIMINATION
	>> Accuracy using Forward Feature Elimination = 0.9274927650618259
	>> Model's score using Forward Feature Elimination = 1.0
	'''
if __name__ == '__main__':
	main()
