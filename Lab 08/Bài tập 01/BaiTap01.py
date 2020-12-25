import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import VarianceThreshold

def decisionTreeModel(X_train, X_test, y_train, y_test, **kwargs):
	'''
		Fitting decision tree model to training dataset and calculate accuracy score on testing dataset
		------------------------
		Parameters:
			X_train, y_train: DataFrame object
				Training dataset with corresponding labels
			X_test, y_test: DataFrame object
				Testing dataset with corresponding labels
			kwargs: dictionary
				Keyword arguments to pass in Decision Tree model in scikit-learn
	'''
	
	clf = DecisionTreeClassifier(**kwargs).fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	return accuracy_score(y_test, y_pred)

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
	numerical_x_train = X_train

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

def main():
	# Read data
	data = pd.read_csv('sonar.csv', header = None)
	
	# Basic first view of dataset
	print('>> FIRST 5 ROWS OF DATASET:')
	print(data.head())
	
	print('>> ORIGINAL DATA INFO:')
	print(data.info())
	print('>> DESCRIBE ORIGINAL DATA (NUMERICAL COLUMNS):')
	print(data.describe().T)
	print('>> DESCRIBE ORIGINAL DATA (OBJECT COLUMNS):')
	print(data.describe(include = 'O').T)
	
	# Preprocessing data
	
	# Handle missing values and duplicate values
	# No missing values in any columns
	print('>> COUNT NUMBER OF MISSING VALUES IN EACH COLUMNS OF DATASET:')
	print(data.isna().sum())
	
	# No duplicated rows
	print(f'>> COUNT NUMBER OF DUPLICATED ROWS IN DATASET: {data.duplicated().any().sum()}')
	
	# Label encoder
	le = LabelEncoder()
	data.iloc[:, -1] = le.fit_transform(data.iloc[:, -1])
	
	# EDA
	# Dữ liệu khá cân bằng về phân bố giữa các lớp
	sns.countplot(x = data.iloc[:, -1])
	plt.savefig('Class countplot.png')
	
	# Normalized data
	X, y = data.iloc[:, :-1], data.iloc[:, -1]
	
	# Split dataset
	X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 0, test_size = 0.25)
	
	# Feature engineering
	
	# Drop constant features
	X_train, X_test = ConstantFeatures(X_train, X_test)
	
	# Drop columns that have high correlation
	X_train, X_test = correlationFilterMethod(X_train, X_test)
	
	# Training model and tuning hyper-parameters
	
	kf = StratifiedKFold(n_splits = 5)
	clf = DecisionTreeClassifier(random_state = 0)
	print('>> CROSS VALIDATION SCORE ON TRAINING DATASET: ')
	print(cross_val_score(clf, X_train, y_train, cv = kf))
	
	accuracy = decisionTreeModel(X_train, X_test, y_train, y_test, random_state = 0)
	
	print(f'>> ACCURACY SCORE OF DECISION TREE MODEL ON TESTING DATASET (BEFORE TUNING HYPERPARAMETERS): {accuracy}')
	
	clf = DecisionTreeClassifier()
	searchSpace = dict()
	searchSpace['criterion'] = ['gini', 'entropy']
	searchSpace['splitter'] = ['best', 'random']
	searchSpace['max_features'] = ['auto', 'sqrt', 'log2']
	searchSpace['random_state'] = list(range(10, 20))
	
	search = GridSearchCV(
		clf, searchSpace, scoring = make_scorer(accuracy_score), n_jobs = -1, cv = kf
	)
	
	result = search.fit(X_train, y_train)
	print('>> TUNING HYPERPARAMETERS RESULT:')
	print(f'>> BEST SCORE: {result.best_score_}')
	print(f'>> BEST HYPERPARAMETERS: {result.best_params_}')
	
	accuracy = decisionTreeModel(X_train, X_test, y_train, y_test, **result.best_params_)
	print(f'>> ACCURACY SCORE OF DECISION TREE MODEL ON TESTING DATASET (AFTER TUNING HYPERPARAMETERS): {accuracy}')
	
	'''
		>> CROSS VALIDATION SCORE ON TRAINING DATASET: 
			[0.5625     0.74193548 0.83870968 0.77419355 0.4516129 ]
		>> ACCURACY SCORE OF DECISION TREE MODEL ON TESTING DATASET (BEFORE TUNING HYPERPARAMETERS): 0.5769230769230769
		>> TUNING HYPERPARAMETERS RESULT:
			>> BEST SCORE: 0.7568548387096774
			>> BEST HYPERPARAMETERS: {'criterion': 'entropy', 'max_features': 'auto', 'random_state': 18, 'splitter': 'best'}
		>> ACCURACY SCORE OF DECISION TREE MODEL ON TESTING DATASET (AFTER TUNING HYPERPARAMETERS): 0.7884615384615384
	'''
	
if __name__ == '__main__':
	main()
