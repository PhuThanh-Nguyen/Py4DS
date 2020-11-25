'''
	Python for Data Science
	Lab 6: Feature Selection vs Dimensionality Reduction - Excercise 01
	
	1. Defining the functions correspond with the feature selection or dimensionality reduction
technologies.
	2. Giving examples to demonstrate your function works
'''
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold

def Constant_Features(X_train, X_test, threshold = 0):
  """
	Removing Constant Features using Variance Threshold
	
	Parameters:
		threshold: float, default = 0
			threshold parameter to identify the variable as constant
		X_train, X_test: DataFrame object
			Training and testing data to evaluate 
	
	Returns: 
		train data, test data after applying filter methods
  """
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
  
  return x_train, x_test

def quasiConstantFeatures(X_train, X_test, thres = 0.98):
	
	'''
		Perform Quasi-constant feature selection
		Parameters:
			X_train, X_test: DataFrame objects
				Training and testing data
			thres: float, between 0 and 1
				threshold to drop feature
		Returns:
			X_train, X_test after applying Quasi - Constant feature selection
	'''
	
	# Create empty list
	quasi_constant_feature = []

	# Loop over all the columns
	for feature in X_train.columns:
		# Calculate the ratio.
		predominant = (X_train[feature].value_counts() / np.float(len(X_train))).sort_values(ascending=False).values[0]
		
		# Append the column name if it is bigger than the threshold
		if predominant >= thres:
		    quasi_constant_feature.append(feature)   

	# Drop the quasi constant columns
	X_train = X_train.drop(labels = quasi_constant_feature, axis = 1)
	X_test = X_test.drop(labels = quasi_constant_feature, axis = 1)
	
	return X_train, X_test

def duplicateFeatures(X_train, X_test):
	'''
		Drop duplicated columns from traning, testing data
		Parameters:
			X_train, X_test: DataFrame objects
				Training and testing data
		Returns:
			X_train, X_test after dropping duplicated columns
	'''
	# Transpose the feature matrice
	train_features_T = X_train.T

	# Select the duplicated features columns names
	duplicated_columns = train_features_T[train_features_T.duplicated()].index.values

	# Drop those columns
	X_train = X_train.drop(labels = duplicated_columns, axis = 1)
	X_test = X_test.drop(labels = duplicated_columns, axis = 1)
	return X_train, X_test

def BackwardFeatureElimination(X_train, y_train, X_test, y_test, clf):
	
	'''
		Perform backward feature elimination
		
		Parameters:
			X_train, y_train, X_test, y_test: DataFrame object
				Training and testing data
			clf: sklearn classification model
				Model to train, test and calculate error
		
		Returns:
			X_train, X_test after applying backward feature elimination algorithm
	'''
	
	def errorCalculate(columns):
		'''
			A wrapper function to calculate error on training, testing data on given columns
			Parameters:
				columns: list-like
					Set of columns to calculate error
			Returns:
				Error (1 - accuracy_score) when train model using such columns 
		'''
		X_train_col = X_train[columns]
		X_test_col = X_test[columns]
		
		clf.fit(X_train_col, y_train)
		y_pred = clf.predict(X_test_col)
		
		return 1 - accuracy_score(y_test, y_pred)
	
	min_err = errorCalculate(X_train.columns)
	cols = X_train.columns
	
	for column in X_train.columns:
		temp = cols.copy()
		temp.remove(column)
		
		err = errorCalculate(temp)
		
		if err < min_err:
			min_err = err
		else:
			return X_train[cols], X_test[col]
		cols = temp

def ForwardFeatureElimination(X_train, y_train, X_test, y_test):
	
	'''
		Perform forward feature elimination
		Parameters:
			X_train, y_train, X_test, y_test: Data frame
				Training and testing data
			clf: sklearn classification model
				Model to train, test and calculate error
		Returns:
			X_train, X_test after applying forward feature elimination algorithm
	'''
	
	def errorCalculate(columns):
		'''
			A wrapper function to calculate error on training, testing data on given columns
			Parameters:
				columns: list-like
					Set of columns to calculate error
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
	
	while cols != X_train.columns:
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
	return X_train[cols], X_test[cols]

def main():
	pass

if __name__ '__main__':
	main()
