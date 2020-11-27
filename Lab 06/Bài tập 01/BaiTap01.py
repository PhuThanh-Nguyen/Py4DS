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
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest, chi2, SelectFromModel, RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

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

def quasiConstantFeatures(X_train, X_test, thres = 0.98):
	
	'''
		Perform Quasi-constant feature selection
		------------------
		Parameters:
			X_train, X_test: DataFrame objects
				Training and testing data
			thres: float, between 0 and 1
				threshold to drop feature
		------------------
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
		------------------
		Parameters:
			X_train, X_test: DataFrame objects
				Training and testing data
		------------------
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

def mutualInfomation(numFeatures, X_train, X_test, y_train, y_test):

	'''
		Perform Mutual information filter method
		------------------
		Parameters:
			numFeatures: int
				Number of features to retain after filter
			X_train, X_test, y_train, y_test: DataFrame/Series objects
				Training data
		------------------
		Returns:
			Training data and testing data after applying mutual information
	'''

	# Get only the numerical features.
	numerical_X_train = X_train[X_train.select_dtypes([np.number]).columns]
	numerical_X_test = X_test[X_test.select_dtypes([np.number]).columns]

	# Create the SelectKBest with the mutual info strategy.
	selection_XTrain = SelectKBest(mutual_info_classif, k = numFeatures).fit(numerical_X_train, y_train)
	selection_XTest = SelectKBest(mutual_info_classif, k = numFeatures).fit(numerical_X_test, y_test)
	
	return selection_XTrain.transform(X_train), selection_XTest.transform(X_test)

def chiSquareMethod(numFeatures, X_train, X_test, y_train, y_test):
	'''
		Perform Chi-Square method
		------------------
		Parameters:
			numFeatures: int
				Number of features to retain after filter
			X_train, X_test, y_train: DataFrame/Series objects
				Training data and testing data
		------------------
		Returns:
			Training data and testing data after applying chisquare method

	'''
	# Apply the chi2 score on the data and target (target should be binary).  
	selection_XTrain = SelectKBest(chi2, k = numFeatures).fit(X_train, y_train)
	selection_XTest = SelectKBest(chi2, k = numFeatures).fit(X_test, y_test)

	return selection_XTrain.transform(X_train), selection_XTest.transform(X_test)

def selectFromModel(mdl, X_train, y_train, X_test):
	'''
		Using Select From Model from sklearn.model_selection
		------------------
		Parameters:
			mdl: object
				The base estimator from which the transformer is built.
				The estimator must have either a feature_importances_ or coef_ attribute after fitting
			X_train, y_train, X_test: DataFrame/Series objects
		------------------
		Returns:
			Training and testing data after using sklearn.model_selection.SelectFromModel method
	'''
	select_model = SelectFromModel(mdl)
	fit = select_model.fit(X_train, y_train)
	return fit.transform(X_train), fit.transform(X_test)
	
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

def RFEMethod(mdl, n_features, X_train, X_test, y_train, y_test):
	'''
		Perform Recursive Feature Elimination (RFE) Method
		------------------
		Parameters:
			mdl: object
				A supervised learning estimator with a fit method that provides information about feature importance 
				either through a coef_ attribute or through a feature_importances_ attribute
			n_features:	int
				The number of features to select
			X_train, X_test, y_train, y_test: DataFrame/Series objects
		------------------
		Returns:
			Training and testing data after applying RFE Method
	'''
	# Define model
	rfe = RFE(estimator = mdl, n_features_to_select = n_features)
	# Fit the model
	rfe.fit(X_train, y_train)
	return rfe.transform(X_train), rfe.transform(X_test)

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

def main():
	rnd = np.random.RandomState(0)
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

	# ------------------- Test mutualInfomation function -------------------
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
	print('>> Before apply mutual information filter method')
	print('>> X_train:')
	print(X_train)
	print('>> X_test:')
	print(X_test)
	# Apply Mutual Information Method
	X_train_new, X_test_new = mutualInfomation(4, X_train, X_test, y_train, y_test)
	print('>> After apply mutual information filter method')
	print('>> X_train:')
	print(X_train_new)
	print('>> X_test:')
	print(X_test_new)

	# ------------------- Test Chi Square function -------------------
	# Create dummy data to test
	data = pd.DataFrame({'a' : np.abs(rnd.randn(10)), 'b' : np.abs(rnd.randn(10)),\
						 'c' : np.abs(rnd.randn(10)), 'd' : np.abs(rnd.randn(10)),\
						 'y' : rnd.binomial(1, 0.4, size = 10)})
	data['e'] = 2 * data['a'] + 1
	data['f'] = data['b'] + 2 * data['a']
	# Split data
	X, y = data.drop('y', axis = 1), data['y']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = rnd)

	print('*' * 20)
	print('>> Before apply Chi Square filter method')
	print('>> X_train:')
	print(X_train)
	print('>> X_test:')
	print(X_test)
	# Apply Chi Square method
	X_train_new, X_test_new = chiSquareMethod(4, X_train, X_test, y_train, y_test)
	print('>> After apply Chi Square filter method')
	print('>> X_train:')
	print(X_train_new)
	print('>> X_test:')
	print(X_test_new)

	# ------------------- Test selectFrommodel -------------------
	# Create dummy data to test
	data = pd.DataFrame({'a' : np.abs(rnd.randn(10)), 'b' : np.abs(rnd.randn(10)),\
						 'c' : np.abs(rnd.randn(10)), 'd' : np.abs(rnd.randn(10)),\
						 'y' : rnd.binomial(1, 0.4, size = 10)})
	data['e'] = 2 * data['a'] + 1
	data['f'] = data['b'] + 2 * data['a']
	# Split data
	X, y = data.drop('y', axis = 1), data['y']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = rnd)

	print('*' * 20)
	print('>> Before apply Select From Model method')
	print('>> X_train:')
	print(X_train)
	print('>> X_test:')
	print(X_test)
	# Apply Select From Model method
	rfc = RandomForestClassifier(n_estimators = 100)
	X_train_new, X_test_new = selectFromModel(rfc, X_train, y_train, X_test)
	print('>> After apply Select From Model method')
	print('>> X_train:')
	print(X_train_new)
	print('>> X_test:')
	print(X_test_new)

	# ------------------- Test PCA -------------------
	data = pd.DataFrame({'a' : np.abs(rnd.randn(10)), 'b' : np.abs(rnd.randn(10)),\
						 'c' : np.abs(rnd.randn(10)), 'd' : np.abs(rnd.randn(10)),\
						 'y' : rnd.binomial(1, 0.4, size = 10)})
	data['e'] = 2 * data['a'] + 1
	data['f'] = data['b'] + 2 * data['a']
	# Split data
	X, y = data.drop('y', axis = 1), data['y']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = rnd)

	print('*' * 20)
	print('>> Before apply PCA method')
	print('>> X_train:')
	print(X_train)
	print('>> X_test:')
	print(X_test)
	# Apply PCA Method
	X_train_new, X_test_new = PCAMethod(2, X_train, X_test)
	print('>> After apply PCA method')
	print('>> X_train:')
	print(X_train_new)
	print('>> X_test:')
	print(X_test_new)

	# ------------------- Test RFE -------------------
	data = pd.DataFrame({'a' : np.abs(rnd.randn(10)), 'b' : np.abs(rnd.randn(10)),\
						 'c' : np.abs(rnd.randn(10)), 'd' : np.abs(rnd.randn(10)),\
						 'y' : rnd.binomial(1, 0.4, size = 10)})
	data['e'] = 2 * data['a'] + 1
	data['f'] = data['b'] + 2 * data['a']
	# Split data
	X, y = data.drop('y', axis = 1), data['y']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = rnd)

	print('*' * 20)
	print('>> Before apply RFE method')
	print('>> X_train:')
	print(X_train)
	print('>> X_test:')
	print(X_test)
	# Apply RFE Method
	rfc = RandomForestClassifier(n_estimators = 100)
	X_train_new, X_test_new = RFEMethod(rfc, 2, X_train, X_test, y_train, y_test)
	print('>> After apply RFE method')
	print('>> X_train:')
	print(X_train_new)
	print('>> X_test:')
	print(X_test_new)

	# ------------------- Test Backward Feature Elimination method -------------------
	data = pd.DataFrame({'a' : np.abs(rnd.randn(10)), 'b' : np.abs(rnd.randn(10)),\
						 'c' : np.abs(rnd.randn(10)), 'd' : np.abs(rnd.randn(10)),\
						 'y' : rnd.binomial(1, 0.4, size = 10)})
	data['e'] = 2 * data['a'] + 1
	data['f'] = data['b'] + 2 * data['a']
	# Split data
	X, y = data.drop('y', axis = 1), data['y']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = rnd)

	print('*' * 20)
	print('>> Before apply Backward Feature Elimination method')
	print('>> X_train:')
	print(X_train)
	print('>> X_test:')
	print(X_test)
	# Apply Backward Feature Elimination Method
	rfc = RandomForestClassifier(n_estimators = 100)
	X_train_new, X_test_new = BackwardFeatureElimination(X_train, y_train, X_test, y_test, rfc)
	print('>> After apply Backward Feature Elimination method')
	print('>> X_train:')
	print(X_train_new)
	print('>> X_test:')
	print(X_test_new)

	# ------------------- Test Forward Feature Elimination method -------------------
	data = pd.DataFrame({'a' : np.abs(rnd.randn(10)), 'b' : np.abs(rnd.randn(10)),\
						 'c' : np.abs(rnd.randn(10)), 'd' : np.abs(rnd.randn(10)),\
						 'y' : rnd.binomial(1, 0.4, size = 10)})
	data['e'] = 2 * data['a'] + 1
	data['f'] = data['b'] + 2 * data['a']
	# Split data
	X, y = data.drop('y', axis = 1), data['y']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = rnd)

	print('*' * 20)
	print('>> Before apply Forward Feature Elimination method')
	print('>> X_train:')
	print(X_train)
	print('>> X_test:')
	print(X_test)
	# Apply Forward Feature Elimination Method
	rfc = RandomForestClassifier(n_estimators = 100)
	X_train_new, X_test_new = ForwardFeatureElimination(X_train, y_train, X_test, y_test, rfc)
	print('>> After apply Forward Feature Elimination method')
	print('>> X_train:')
	print(X_train_new)
	print('>> X_test:')
	print(X_test_new)

if __name__ == '__main__':
	main()