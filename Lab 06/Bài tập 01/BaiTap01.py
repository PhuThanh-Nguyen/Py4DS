'''
	Python for Data Science
	Lab 6: Feature Selection vs Dimensionality Reduction - Excercise 01
	
	1. Defining the functions correspond with the feature selection or dimensionality reduction
technologies.
	2. Giving examples to demonstrate your function works
'''
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest, chi2
from sklearn.decomposition import PCA

def ConstantFeatures(X_train, X_test, threshold = 0):
  '''
	Removing Constant Features using Variance Threshold
	
	Parameters:
		threshold: float, default = 0
			threshold parameter to identify the variable as constant
		X_train, X_test: DataFrame object
			Training and testing data to evaluate 
	
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

def correlationFilterMethod(X_train, X_test, thres = 0.8):
	'''
		Perform Correlation filter method
		Parameters:
			X_train, X_test: DataFrame objects
				Training and testing data
			thres: float, default: 0.8
				Threshold of absolute value of correlation, must between 0 and 1
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

def mutualInfomation(numFeatures, X_train, y_train, X_test):

	'''
		Perform Mutual information filter method
		Parameters:
			numFeatures: int
				Number of features to retain after filter
			X_train, X_test, y_train: DataFrame/Series objects
				Training data and testing data
		Returns:
			Training data and testing data after applying mutual information
	'''

	# Get only the numerical features.
	numerical_X_train = X_train[X_train.select_dtypes([np.number]).columns]

	# Create the SelectKBest with the mutual info strategy.
	selection = SelectKBest(mutual_info_classif, k = numFeatures).fit(numerical_x_train, y_train)

	features = X_train.columns[selection.get_support()]
	
	return X_train[features], X_test[features]

def chiSquareMethod(numFeatures, X_train, y_train, X_test):
	'''
		Perform Chi-Square method
		Parameters:
			numFeatures: int
				Number of features to retain after filter
			X_train, X_test, y_train: DataFrame/Series objects
				Training data and testing data
		Returns:
			Training data and testing data after applying chisquare method

	'''
	# Apply the chi2 score on the data and target (target should be binary).  
	selection = SelectKBest(chi2, k = numFeatures).fit(X_train, y_train)

	features = X_train.columns[selection.get_support()]
	return X_train[features], X_test[features]

def PCAMethod(num_components, X_train):
	pca = PCA(n_components = num_components)
	pca.fit(X_train)
	X_train = pca.transform(X_train)
	return X_train

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

def ForwardFeatureElimination(X_train, y_train, X_test, y_test, clf):
	
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
	# ------------------- Test Costant Features function -------------------
	# Create dummy data to test
	data = pd.DataFrame({'a' : [1,1,1,1], 'b' : [1,2,3,4], 'c' : [2,1,3,4], 'y' : [0,0,1,1]})
	print(data)
	print(data.info())
	print(data.describe())

	# Split dummy data
	X, y = data.drop('y', axis = 1), data['y']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

	print('*' * 20)
	print('>> Before apply constant features filter')
	print('>> X_train:')
	print(X_train)
	print('>> X_test:')
	print(X_test)
	# X_train_new and X_test_new must drop column `a`
	X_train_new, X_test_new = ConstantFeatures(X_train, X_test)
	print('>> After apply constant features filter')
	print('>> X_train:')
	print(X_train_new)
	print('>> X_test:')
	print(X_test_new)

	# ------------------- Test quasiConstantFeatures function -------------------
	# Create dummy data to test
	data = pd.DataFrame({'a' : [1,1,2,1], 'b' : [1,2,3,3], 'c' : [2,1,3,4], 'y' : [0,0,1,1]})
	print(data)
	print(data.info())
	print(data.describe())

	# Split dummy data
	X, y = data.drop('y', axis = 1), data['y']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

	print('*' * 20)
	print('>> Before apply Quasi-constant features filter')
	print('>> X_train:')
	print(X_train)
	print('>> X_test:')
	print(X_test)
	# X_train_new and X_test_new must drop column `a`
	X_train_new, X_test_new = quasiConstantFeatures(X_train, X_test, thres = 0.8)
	print('>> After apply Quasi-constant features filter')
	print('>> X_train:')
	print(X_train_new)
	print('>> X_test:')
	print(X_test_new)

	# ------------------- Test correlationFilterMethod function -------------------
	rnd = np.random.RandomState(0)
	# Create dummy data to test
	data = pd.DataFrame({'a' : rnd.randn(10), 'b' : rnd.randn(10),\
						 'c' : rnd.randn(10), 'd' : rnd.randn(10),\
						 'y' : rnd.binomial(1, 0.4, size = 10)})
	data['e'] = 2 * data['a'] + 1
	data['f'] = data['b'] + 2 * data['a']
	# Split data
	X, y = data.drop('y', axis = 1), data['y']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = rnd)

	print('*' * 20)
	print('>> Before apply correlation filter method')
	print('>> X_train:')
	print(X_train)
	print('>> X_test:')
	print(X_test)
	# X_train_new and X_test_new must drop column `e` and `f`
	X_train_new, X_test_new = correlationFilterMethod(X_train, X_test, thres = 0.8)
	print('>> After apply correlation filter method')
	print('>> X_train:')
	print(X_train_new)
	print('>> X_test:')
	print(X_test_new)

if __name__ == '__main__':
	main()