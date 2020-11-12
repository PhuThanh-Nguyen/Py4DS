import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler

def LinearRegression_Model(X_train,X_test,y_train,y_test):
	clf = LinearRegression()
	clf.fit(X_train,y_train)
	y_pred = clf.predict(X_test)
	return (
		np.sqrt(metrics.mean_squared_error(y_test,y_pred)),
		r2_score(y_test,y_pred)*100,
		mean_absolute_error(y_test,y_pred),
		mean_squared_error(y_test,y_pred)
	)

def main():
	nyc_df = pd.read_csv('../dataset/AB_NYC_2019.csv')
	
	# Drop duplicate values
	print('>> Before drop duplicate values: ')
	print(f'>> nyc_df shape: {nyc_df.shape}')
	print(nyc_df.info())
	print(nyc_df.describe())
	print('>> After drop duplicate values:')
	print(f'>> nyc_df shape: {nyc_df.shape}')
	print(nyc_df.info())
	print(nyc_df.describe())
	print('>> Before drop missing values:')
	print(f'>> nyc_df shape: {nyc_df.shape}')
	print(nyc_df.info())
	
	print(nyc_df.head())
	# Missing values
	missing_values = nyc_df.isnull().sum().sort_values()
	percentage_missing_values = (missing_values/len(nyc_df))*100
	missing_values = pd.concat([missing_values, percentage_missing_values], axis = 1, keys= ['Missing values', '% Missing'])
	print(missing_values)
	# Drop columns have no impact on the model prediction
	nyc_df.drop(['id', 'name', 'host_name', 'last_review'], axis=1, inplace=True)
	nyc_df.reviews_per_month.fillna(0, inplace=True)
	print('>> After drop missing values:')
	print(f'>> nyc_df shape: {nyc_df.shape}')
	print(nyc_df.info())
	print(nyc_df.describe())
	nyc_df.drop_duplicates(subset=nyc_df.columns.values[:-1], keep='first', inplace=True)
	corr = nyc_df.corr()
	corr = corr.filter(items=['price'])
	plt.figure(figsize=(15, 8))
	sns.heatmap(corr, annot=True)
	plt.show()
	"""
	As we see through above heatmap. The coefficient correlation of feature with each other is not high.
	So we don't need to remove more feature for prediction
	"""
	# Encode the categorical features
	categorical_features = nyc_df.select_dtypes(include= [np.object]).columns
	label_enc = LabelEncoder()
	for col in categorical_features:
		nyc_df[col] = label_enc.fit_transform(nyc_df[col])
    
	fig, axes = plt.subplots(ncols=3, nrows=4, figsize=(15, len(nyc_df.columns)/2))
	for i, c in zip(axes.flatten(), nyc_df.columns):
		sns.boxplot(nyc_df[c], ax = i)
	fig.subplots_adjust(hspace=.5)
	fig.suptitle('Before removing outliers', fontsize = 16)
	plt.show()

	# Detect outlier values
	Q1 = nyc_df.quantile(0.25)
	Q3 = nyc_df.quantile(0.75)
	IQR = Q3 - Q1
	print(IQR)
	# Remove outliers
	nyc_df = nyc_df[~((nyc_df < (Q1 - 1.5 * IQR)) | (nyc_df > (Q3 + 1.5 * IQR))).any(axis=1)]
	print(f'>> nyc_df shape after removing outliers: {nyc_df.shape}')

	fig, axes = plt.subplots(ncols = 3, nrows = 4, figsize = (15,len(nyc_df.columns)/2))
	for i, c in zip(axes.flatten(), nyc_df.columns):
		sns.boxplot(nyc_df[c], ax = i)
	fig.subplots_adjust(hspace=.5)
	fig.suptitle('After removing outliers', fontsize = 16)
	plt.show()

	X = nyc_df.drop(['price'], axis=1)
	y = nyc_df.price

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

	MSE,R2,MAE,MSRE = LinearRegression_Model(X_train,X_test,y_train,y_test)
	
	print('\n>> Before normalizing <<\n')
	print(f'Mean Squared Error: {MSE}')
	print(f'R2 Score: {R2}')
	print(f'Mean Absolute Error: {MAE}')
	print(f'Mean Squareroot Error: {MSRE}')

	print('\n>> After Scale <<')
	# Using Standard Scaler
	scalerX = StandardScaler()
	scalerX.fit(X_train)
	X_train_standardScaler = scalerX.transform(X_train)

	scalerX.fit(X_test)
	X_test_standardScaler = scalerX.transform(X_test)

	MSE,R2,MAE,MSRE = LinearRegression_Model(X_train_standardScaler,X_test_standardScaler,y_train,y_test)
	print(f'\nMean Squared Error using Standard Scaler: {MSE}')
	print(f'R2 Score using Standard Scaler: {R2}')
	print(f'Mean Absolute Error using Standard Scaler: {MAE}')
	print(f'Mean Squareroot Error using Standard Scaler: {MSRE}')

	# Using Robust Scaler
	scalerX = RobustScaler()
	scalerX.fit(X_train)
	X_train_RobustScaler = scalerX.transform(X_train)

	scalerX.fit(X_test)
	X_test_RobustScaler = scalerX.transform(X_test)

	MSE,R2,MAE,MSRE = LinearRegression_Model(X_train_RobustScaler,X_test_RobustScaler,y_train,y_test)
	print(f'\nMean Squared Error using Robust Scaler: {MSE}')
	print(f'R2 Score using Robust Scaler: {R2}')
	print(f'Mean Absolute Error using Robust Scaler: {MAE}')
	print(f'Mean Squareroot Error using Robust Scaler: {MSRE}')

	# Using Normalizer
	scalerX = Normalizer()
	scalerX.fit(X_train)
	X_train_NormalizeScaler = scalerX.transform(X_train)

	scalerX.fit(X_test)
	X_test_NormalizeScaler = scalerX.transform(X_test)

	MSE,R2,MAE,MSRE = LinearRegression_Model(X_train_NormalizeScaler,X_test_NormalizeScaler,y_train,y_test)
	print(f'\nMean Squared Error using Normalizer: {MSE}')
	print(f'R2 Score using Normalizer: {R2}')
	print(f'Mean Absolute Error using Normalizer: {MAE}')
	print(f'Mean Squareroot Error using Normalizer: {MSRE}')
	'''
	With random_state == 1:
		Mean Squared Error using Standard Scaler: 49.246496857803805
		R2 Score using Standard Scaler: 47.08939720751063
		Mean Absolute Error using Standard Scaler: 36.81807466190845
		Mean Squareroot Error using Standard Scaler: 2425.2174527656803

		Mean Squared Error using Robust Scaler: 86.38395049787574
		R2 Score using Robust Scaler: -62.801404373231804
		Mean Absolute Error using Robust Scaler: 72.03624303878996
		Mean Squareroot Error using Robust Scaler: 7462.186903619447

		Mean Squared Error using Normalizer: 68.41117331653514
		R2 Score using Normalizer: -2.1047867248628727
		Mean Absolute Error using Normalizer: 55.1159828976059
		Mean Squareroot Error using Normalizer: 4680.08863454501

	'''
if __name__ == '__main__':
	main()
