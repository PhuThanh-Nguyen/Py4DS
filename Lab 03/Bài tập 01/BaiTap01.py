import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

def decisionTreeModel(X_train, X_test, y_train, y_test):
	clf = DecisionTreeClassifier(random_state = 1)
	clf = clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	return metrics.accuracy_score(y_test, y_pred)

def main():
	# Read data
	data = pd.read_csv('../dataset/creditcard.csv')
	# EDA
	for i in range(len(data.columns)):
		print(data.iloc[:,i].value_counts())
		print("*"*20)
	'''
		Nhận xét: Hệ số tương quan giữa các biến không lớn nên ta không bỏ cột
	'''
	plt.figure(figsize=(14,12))
	sns.heatmap(data.corr(), linewidths=.1, cmap="YlGnBu", annot = False)
	plt.yticks(rotation=0)
	plt.show()
	
	plt.figure(figsize=(14,12))
	sns.countplot(x="Class", data=data, linewidth=2, edgecolor=sns.color_palette("dark"))
	plt.show()
	
	avg_amount = data.groupby("Class")[['Amount']].agg("mean").reset_index()
	plt.figure(figsize=(14,12))
	sns.barplot(x='Class', y= "Amount", data = avg_amount)
	plt.show()
    
	plt.figure(figsize=(14,12))
	Amount = sns.boxplot(x="Class", y="Amount", data=data)
	Amount.set(ylim=(data['Amount'].min(),300))
	plt.show()
	
	# Remove missing values, duplicates
	# Remove missing values:
	print('>> Before drop missing values: ')
	print(f'>> Data shape: {data.shape}')
	print(data.info())
	print(data.describe())
	data = data.dropna()
	
	print('>> After drop missing values:')
	print(f'>> Data shape: {data.shape}')
	print(data.info())
	print(data.describe())
	
	# Remove duplicate values
	print('>> Before drop duplicate values: ')
	print(f'>> Data shape: {data.shape}')
	print(data.info())
	print(data.describe())
	
	data.drop_duplicates(subset = data.columns.values[:-1], keep = 'first', inplace = True)
	print('>> After drop duplicate values:')
	print(f'>> Data shape: {data.shape}')
	print(data.info())
	print(data.describe())
	
	# Train model with differents normalizer
	X = data.drop(['Class'], axis = 'columns')
	y = data['Class']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
	
	# Before normalize
	accuracy = decisionTreeModel(X_train, X_test, y_train, y_test)
	print(f'Decision Tree model accuracy before normalization: {accuracy}')
	
	# Using Standard Scaler
	scalerX = StandardScaler()
	scalerX.fit(X_train)
	X_train_standardScaler = scalerX.transform(X_train)
	
	scalerX.fit(X_test)
	X_test_standardScaler = scalerX.transform(X_test)
	
	accuracy = decisionTreeModel(X_train_standardScaler, X_test_standardScaler, y_train, y_test)
	print(f'Decision Tree model accuracy using Standard Scaler: {accuracy}')
	
	# Using Robust Scaler
	scalerX = RobustScaler()
	scalerX.fit(X_train)
	X_train_RobustScaler = scalerX.transform(X_train)
	
	scalerX.fit(X_test)
	X_test_RobustScaler = scalerX.transform(X_test)
	
	accuracy = decisionTreeModel(X_train_RobustScaler, X_test_RobustScaler, y_train, y_test)
	print(f'Decision Tree model accuracy using Robust Scaler: {accuracy}')
	
	# Using Normalizer
	scalerX = Normalizer()
	scalerX.fit(X_train)
	X_train_NormalizeScaler = scalerX.transform(X_train)
	
	scalerX.fit(X_test)
	X_test_NormalizeScaler = scalerX.transform(X_test)
	
	accuracy = decisionTreeModel(X_train_NormalizeScaler, X_test_NormalizeScaler, y_train, y_test)
	print(f'Decision Tree model accuracy using Normalizer: {accuracy}')
	'''
	With random_state == 1:
		Decision Tree model accuracy before normalization: 0.9990131393447246
		Decision Tree model accuracy using Standard Scaler: 0.9991964134664185
		Decision Tree model accuracy using Robust Scaler: 0.9990695313821688
		Decision Tree model accuracy using Normalizer: 0.9986042970732533
	'''
if __name__ == '__main__':
	main()
