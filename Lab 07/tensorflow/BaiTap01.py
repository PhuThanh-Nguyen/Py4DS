'''
	Python for Data Science
	Lab 7: Deep Learning
	Using tensorflow library
'''

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing

# Read data
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight','Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names = column_names, na_values = '?', comment = '\t', sep = ' ', skipinitialspace = True)

dataset = raw_dataset.copy()
print(dataset.head())

for i in range(1,17):
	print(dataset.iloc[:,i].value_counts())
    print("*"*20)

# Preprocessing data

# Drop missing values
print('>> Before drop missing values:')
print(dataset.info())
print(dataset.describe())

print('>> Count missing values in each column:')
print(dataset.isna().sum())

dataset = dataset.dropna()
print('>> After drop missing values:')
print(dataset.info())
print(dataset.describe())


# Drop duplicates
print('>> Before drop duplicate values:')
print(dataset.info())
print(dataset.describe())


dataset.drop_duplicates(subset = dataset.columns.values[:-1], keep = 'first', inplace = True)
print('>> After drop duplicate values:')
print(dataset.info())
print(dataset.describe())

# One hot encoding 
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, prefix = '', prefix_sep = '')
print(dataset.head())

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# EDA
sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
plt.savefig('pairplot.png')

print(train_dataset.describe().transpose())

# Train - Test split
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

# Normalize layer
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())

# Create linear model
linear_model = tf.keras.models.Sequential([
	normalizer,
	tf.keras.layers.Dense(unit = 1)
])

linear_model.compile(
	optimizer = tf.optimizers.Adam(learning_rate = 0.1),
	loss = 'mean_absolute_error'
)

linear_model.fit(
	train_features, train_labels, 
	epochs=100,
	verbose=0,
	# Calculate validation results on 20% of the training data
	validation_split = 0.2
)

print(linear_model.evaluate(test_features, test_labels, verbose = 2))
