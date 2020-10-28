# Add code here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def randomForestModel(X_train, X_test, y_train, y_test):
	clf = RandomForestClassifier()
	clf = clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	return metrics.accuracy_score(y_test, y_pred)

def main():
    path = '/home/duc-hoang/Documents/Junior-Year/Py4DS/LAB/Lab 03/dataset/FIFA2018Statistics.csv'
    data = pd.read_csv(path)
    print('>> Original data infomation')
    print(f'>> data shape: {data.shape}')
    print(data.info())
    print(data.describe())

    # Drop 2 unnecessary columns 
    missing_cols = ['Own goals', 'Own goal Time']
    data = data.drop(missing_cols,axis=1)
    # Drop missing values by rows
    data = data.dropna(axis=0)
    print('>> After dropping missing values')
    print(f'>> data shape: {data.shape}')

    # Encode the object columns
    label_enc = LabelEncoder()
    Cat_Columns = data.dtypes.pipe(lambda X: X[X == 'object']).index
    for col in Cat_Columns:
        data[col] = label_enc.fit_transform(data[col])

    # Selecting features based on correlation
    plt.figure(figsize=(18, 12))
    sns.heatmap(data.corr(), linewidths=.1, cmap="YlGnBu", annot=True)
    plt.tight_layout()
    plt.show()

    corr = data.corr()
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i, j] >= 0.9:
                if columns[j]:
                    columns[j] = False
    selected_columns = data.columns[columns]
    data = data[selected_columns]

    # Handle outlier values
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    print(IQR)

    # Box plot each columns to determine the outlier values
    boxplot = data[[col for col in data.columns]]
    f, ax = plt.subplots(ncols = 4, nrows = 6, figsize=(10,len(boxplot.columns)/3))
    for i, c in zip(ax.flatten(), boxplot.columns):
        sns.boxplot(boxplot[c], ax = i)
    f.tight_layout()
    plt.show()

    data_out = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]
    print(f'data shape after removing outliers: {data_out.shape}')

    X = data_out.drop(['Man of the Match'],axis=1)
    y = data_out['Man of the Match']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

    accuracy = randomForestModel(X_train, X_test, y_train, y_test)
    print(f'Decision Tree Accuracy before normalizing {accuracy}')

    # Using Standard Scaler
    scalerX = StandardScaler()
    scalerX.fit(X_train)
    X_train_standardScaler = scalerX.transform(X_train)
	
    scalerX.fit(X_test)
    X_test_standardScaler = scalerX.transform(X_test)
	
    accuracy = randomForestModel(X_train_standardScaler, X_test_standardScaler, y_train, y_test)
    print(f'Random Forest model accuracy using Standard Scaler: {accuracy}')
	
    # Using Robust Scaler
    scalerX = RobustScaler()
    scalerX.fit(X_train)
    X_train_standardScaler = scalerX.transform(X_train)
	
    scalerX.fit(X_test)
    X_test_standardScaler = scalerX.transform(X_test)
	
    accuracy = randomForestModel(X_train_standardScaler, X_test_standardScaler, y_train, y_test)
    print(f'Random Forest model accuracy using Robust Scaler: {accuracy}')
	
    # Using Normalizer
    scalerX = Normalizer()
    scalerX.fit(X_train)
    X_train_standardScaler = scalerX.transform(X_train)
	
    scalerX.fit(X_test)
    X_test_standardScaler = scalerX.transform(X_test)
	
    accuracy = randomForestModel(X_train_standardScaler, X_test_standardScaler, y_train, y_test)
    print(f'Random Forest model accuracy using Normalizer: {accuracy}')
if __name__ == "__main__":
    main()
