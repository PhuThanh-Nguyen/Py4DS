'''
	Python for Data Science: Laboratory 04 - Excercise 03
	Perform an Exploratory Data Analysis (EDA), Data cleaning, Building clustering models for prediction, Presenting results using 		   dataset `mushrooms.csv`
'''
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

def kMeansModel(n_clusters, X_train, X_test, y_test, random_num):
	'''
		Performs KMeans clustering using sklearn library
		Parameters:
			n_clusters: int
				The number of clusters to form as well as the number of centroids to generate
			X_train: array-like
				Training instances to cluster
			X_test: array-like
				New data to predict
			y_test: array_like
				Ground truth for X_test
			random_num: int
				Random state which will pass to KMeans to initialize centroids for each cluster
		Returns:
			Accuracy of KMeans Clustering model at specified random_state
	'''
	kmeans = KMeans(n_clusters = n_clusters, random_state = random_num).fit(X_train)
	y_pred = kmeans.predict(X_test)
	return metrics.accuracy_score(y_test, y_pred)

def agglomerativeClustering(n_clusters, X_train, X_test, y_test):
	'''
		Performs Agglomerative Clustering using sklearn library
		Parameters:
			n_clusters: int
				The number of clusters to find
			X_train: array-like
				Training instances to cluster
			X_test: array-like
				New data to predict
			y_test: array_like
				Ground truth for X_test
		Returns:
			Accuracy of Agglomerative Clustering algorithm
	'''
	cluster = AgglomerativeClustering(n_clusters = n_clusters, linkage = 'single', affinity = 'euclidean').fit(X_train)
	y_pred = cluster.fit_predict(X_test)
	return metrics.accuracy_score(y_test, y_pred)

def BIRCHClustering(n_clusters, X_train, X_test, y_test):
	'''
		Performs BIRCH Clustering using sklearn library
		Parameters:
			n_clusters: int
				The number of clusters to find
			X_train: array-like
				Training instances to cluster
			X_test: array-like
				New data to predict
			y_test: array_like
				Ground truth for X_test
		Returns:
			Accuracy of BIRCH Clustering algorithm
	'''
	cluster = Birch(n_clusters = n_clusters)
	cluster.fit(X_train)
	y_pred = cluster.predict(X_test)
	return metrics.accuracy_score(y_test, y_pred)

def main():
	# Read data
	data = pd.read_csv('../dataset/mushrooms.csv')
	# EDA
	'''
		Nhận xét:
			Cột veil-type chỉ có một giá trị riêng biệt nên ta có thể bỏ cột này mà không ảnh hưởng đến kết quả
	'''
	for column in data.columns:
		print(column, data[column].nunique())
	
	# Preprocessing
	
	# Drop columns via analysis in EDA:
	data.drop('veil-type', axis = 'columns', inplace = True)
	
	# Drop missing values via description of dataset:
	data = data.loc[data['stalk-root'] != '?'] 
	
	# Encode the categorical features
	label_encoder = LabelEncoder()
	for column in data.columns:
		data[column] = label_encoder.fit_transform(data[column])
	
	# Drop missing values:
	print('>> Data before drop missing values: ')
	print(data.info())
	print(data.describe())
	data.dropna()
	print('>> Data after drop missing values: ')
	print(data.info())
	print(data.describe())
	
	# Drop duplicate values
	print('>> Data before drop duplicate values: ')
	print(data.info())
	print(data.describe())
	data.drop_duplicates(subset = data.columns.values[:-1], keep='first', inplace=True)
	print('>> Data after drop duplicate values: ')
	print(data.info())
	print(data.describe())
	
	# Handle outliers
	Q1 = data.quantile(0.25)
	Q3 = data.quantile(0.75)
	IQR = Q3 - Q1
	print('>> Data before drop outliers: ')
	print(data.info())
	print(data.describe())
	data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
	print('>> Data after drop outliers: ')
	print(data.info())
	print(data.describe())
	
	# Prepare data for training and testing
	X = data.drop('class', axis = 1)
	y = data['class']
	random_state = 843
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = random_state)
	number_clusters = 2
	
	# Using Kmeans clustering
	accuracy = kMeansModel(number_clusters, X_train, X_test, y_test, random_state)
	print(f'Accuracy using KMeans Clustering: {accuracy}')

	# Using Agglomerative Clustering
	accuracy = agglomerativeClustering(number_clusters, X_train, X_test, y_test)
	print(f'Accuracy using Agglomerative Clustering: {accuracy}')
	
	# Using BIRCH Clustering
	accuracy = BIRCHClustering(number_clusters, X_train, X_test, y_test)
	print(f'Accuracy using BIRCH Clustering: {accuracy}')
	
	'''
	With random_state == 843:
		Accuracy using KMeans Clustering: 0.9547511312217195
		Accuracy using Agglomerative Clustering: 0.9909502262443439
		Accuracy using BIRCH Clustering: 0.9457013574660633
	'''

if __name__ == '__main__':
	main()
