'''
	Python for Data Science: Laboratory 04 - Excercise 04
	Perform an Exploratory Data Analysis (EDA), Data cleaning, Building clustering models for prediction, 
	Presenting results using dataset `xAPI-Edu-Data.csv`
'''
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
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
	cluster = AgglomerativeClustering(n_clusters = n_clusters, linkage = 'complete', affinity = 'euclidean').fit(X_train)
	y_pred = cluster.fit_predict(X_test)
	return metrics.accuracy_score(y_test, y_pred)

def MiniBatchKMeansClustering(n_clusters, X_train, X_test, y_test, random_num):
	'''
		Performs MiniBatch Clustering using sklearn library
		Parameters:
			n_clusters: int
				The number of clusters to find
			X_train: array-like
				Training instances to cluster
			X_test: array-like
				New data to predict
			y_test: array_like
				Ground truth for X_test
			random_num : int
				Random state which will pass to KMeans to initialize centroids for each cluster 
		Returns:
			Accuracy of MiniBatch KMeans Clustering algorithm
	'''
	cluster = MiniBatchKMeans(n_clusters = n_clusters, random_state = random_num)
	cluster.fit(X_train)
	y_pred = cluster.fit_predict(X_test)
	return metrics.accuracy_score(y_test, y_pred)

def main():
	# Read data
	
	data = pd.read_csv('../dataset/xAPI-Edu-Data.csv')
	print(data.info())
	data.rename(index = str,
				columns={'gender':'Gender', 'NationalITy': 'Nationality',
				'raisedhands': 'RaisedHands', 'VisITedResources': 'VisitedResources'}, 
				inplace=True)
	
	# EDA
	
	'''
		Nhận xét: 
			Nhận thấy số lượng dữ liệu 'M' chiếm hơn 60% nên 
			dataset có xu hướng chệch nhiều sang 'M' (giới tính nam) hơn là 'F' (giới tính nữ)
	'''
	fig = plt.figure(figsize = (10,8))
	ax = data['Gender'].value_counts(normalize = True).plot(kind = 'bar')
	fig.savefig('Barplot-gender.png')
	
	'''
	Nhận xét:
		+) Phân vị thứ 3 của dữ liệu 'M' khoảng 60%, còn phân vị dữ liệu 'F' hơn 70%
		--> Đa phần các nữ sinh tham gia bàn luận nhiều hơn so với các nam sinh
		+) Boxplot của giới tính Nam có phân vị thứ 2 gần với phân vị thứ nhất -> Dữ liệu của nam sinh có thể lệch phải
		+) Boxplot của giới tính Nữ có phân vị thứ 2 gần với phân vị thứ nhất, nhưng không quá nhiều -> Dữ liệu của nữ sinh khá đều
	'''
	fig = plt.figure(figsize = (10,8))
	sns.boxplot(x = 'Gender', y = 'Discussion', data = data)
	fig.savefig('Boxplot Gender vs Discussion.png')
	
	'''
	Nhạn xét:
		+) Biều đồ phân phối của nam lệch phải, đa đỉnh
		+) Biều đồ phân phối của nữ  lệch phải nhưng không nhiều, đa đỉnh
	'''
	
	FacetGrid = sns.FacetGrid(data, hue = 'Gender', height = 6)
	FacetGrid.map(sns.kdeplot, 'Discussion', shade = True)
	FacetGrid.set(xlim = (0, data['Discussion'].max()))
	FacetGrid.add_legend()
	FacetGrid.savefig('KDE plot Gender vs Discussion.png')
	
	'''
	Nhận xét:
		+) Đa số học sinh giỏi, trung bình nghỉ dưới 7 ngày
		+) Đa số học sinh yếu và trung bình nghỉ trên 7 ngày, số lượng học sinh yếu nhiều hơn so với số lượng học sinh trung bình
	'''
	fig = plt.figure(figsize = (10,8))
	sns.countplot(x = 'StudentAbsenceDays', hue = 'Class', data = data)
	fig.savefig('Countplot: AbsenceDays - Class.png')
	
	# Preprocessing
	
	# Encode the categorical features
	columns = data.dtypes[data.dtypes.values == object].index
	
	label_encoder = LabelEncoder()
	for column in columns:
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
	X = data.drop('Class', axis = 1)
	y = data['Class']
	random_state = 133
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = random_state)
	number_clusters = 3
	# Using Kmeans clustering
	accuracy = kMeansModel(number_clusters, X_train, X_test, y_test, random_state)
	print(f'Accuracy using Kmeans Clustering: {accuracy}')

	# Using Agglomerative Clustering
	accuracy = agglomerativeClustering(number_clusters, X_train, X_test, y_test)
	print(f'Accuracy using Agglomerative Clustering: {accuracy}')
	
	# Using miniBatch Clustering
	accuracy = MiniBatchKMeansClustering(number_clusters, X_train, X_test, y_test, random_state)
	print(f'Accuracy using MiniBatch KMeans Clustering: {accuracy}')
	
	'''
	With random_state == 133:
		Accuracy using Kmeans Clustering: 0.797752808988764
		Accuracy using Agglomerative Clustering: 0.7640449438202247
		Accuracy using MiniBatch KMeans Clustering: 0.0449438202247191
	'''

if __name__ == '__main__':
	main()
