# Add code here
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def rfModel(X_train, X_test, y_train, y_test):
	clf = RandomForestClassifier(random_state=1)
	clf = clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	return metrics.accuracy_score(y_test, y_pred)

def main():
    path = '../dataset/FIFA2018Statistics.csv'
    data = pd.read_csv(path)
    print('>> Original data infomation')
    print(f'>> data shape: {data.shape}')
    print(data.info())
    print(data.describe())

    # Selecting features based on correlation
    numerical_features = data.select_dtypes(include = [np.number]).columns
    plt.figure(figsize=(18, 10))
    sns.heatmap(data[numerical_features].corr(), linewidths=.1, cmap="YlGnBu", annot=True)
    plt.tight_layout()
    plt.title('numerical columns correlation')
    plt.show()

    """
    As we see, the correlation coefficient (corr) of the features with label what we can decide to choose. 
    We can choose features with which 'Man of the Match is highly correlated as 'Goal Scored', 'On-Target', 'Corners', 'Attempts', 'free Kicks', 'Yellow Card', 'red', 'Fouls Committed', 'Own goal Time'.
    """
    data['Man of the Match'] = data['Man of the Match'].map({'Yes': 1, 'No': 0})
    corr = data.corr()
    corr = corr.filter(items = ['Man of the Match'])
    plt.figure(figsize=(15,8))
    sns.heatmap(corr, annot=True)
    plt.show()

    """
    weight_features = ['Goal Scored', 'On-Target', 'Corners', 'Attempts', 'Free Kicks', 'Yellow Card', 'Red', 'Fouls Committed', 'Own goal Time']

    - Because 'Corners','On-Target' and 'Fouls Committed' have low collinearity with 'Man of the Match' and high collinearity with other features, so we can remove them.
    - As 'own Goal Time' and 'Own goals' are having > 90% missing values, filling them with any combination will lead predictive model to false direction. So, dropping them is the best option
    - I believe that 'Date' and '1st Goal' should not impact model's prediction.
    """
    drop_features = ['Own goal Time', 'Own goals', 'On-Target', 'Corners', 'Fouls Committed', 'Date', '1st Goal']
    data.drop(drop_features,axis=1, inplace = True)

    # Encode the categorical features
    categorical_features = data.select_dtypes(include= [np.object]).columns
    label_enc = LabelEncoder()
    for col in categorical_features:
        data[col] = label_enc.fit_transform(data[col])
    
    # Detect outlier values
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    print(IQR)

    # Box plot each columns to detect outliers
    f, ax = plt.subplots(ncols = 4, nrows = 5, figsize=(15,len(data.columns)/2))
    for i, c in zip(ax.flatten(), data.columns):
        sns.boxplot(data[c], ax = i)
    f.tight_layout()
    plt.title('>> Before removing outliers')
    plt.show()

    # remove outliers
    data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    print(f'>> data shape after removing outliers: {data.shape}')

    f, ax = plt.subplots(ncols = 4, nrows = 5, figsize=(15,len(data.columns)/2))
    for i, c in zip(ax.flatten(), data.columns):
        sns.boxplot(data[c], ax = i)
    f.tight_layout()
    plt.title('>> After removing outliers')
    plt.show()

    cleaned_data = pd.get_dummies(data)
    print(f'>> cleaned data shape {cleaned_data.shape}')
    X = cleaned_data.drop(['Man of the Match'],axis=1)
    y = cleaned_data['Man of the Match']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

    accuracy = rfModel(X_train, X_test, y_train, y_test)
    print(f'>> Random Forest Accuracy before normalizing {accuracy}')

    # Using Standard Scaler
    scalerX = StandardScaler()
    scalerX.fit(X_train)
    X_train_standardScaler = scalerX.transform(X_train)
	
    scalerX.fit(X_test)
    X_test_standardScaler = scalerX.transform(X_test)
	
    accuracy = rfModel(X_train_standardScaler, X_test_standardScaler, y_train, y_test)
    print(f'>> Random Forest model accuracy using Standard Scaler: {accuracy}')
	
    # Using Robust Scaler
    scalerX = RobustScaler()
    scalerX.fit(X_train)
    X_train_standardScaler = scalerX.transform(X_train)
	
    scalerX.fit(X_test)
    X_test_standardScaler = scalerX.transform(X_test)
	
    accuracy = rfModel(X_train_standardScaler, X_test_standardScaler, y_train, y_test)
    print(f'>> Random Forest model accuracy using Robust Scaler: {accuracy}')
	
    # Using Normalizer
    scalerX = Normalizer()
    scalerX.fit(X_train)
    X_train_standardScaler = scalerX.transform(X_train)
	
    scalerX.fit(X_test)
    X_test_standardScaler = scalerX.transform(X_test)
	
    accuracy = rfModel(X_train_standardScaler, X_test_standardScaler, y_train, y_test)
    print(f'>> Random Forest model accuracy using Normalizer: {accuracy}')
    """
    Random Forest Accuracy before normalizing 0.6842105263157895
    Random Forest model accuracy using Standard Scaler: 0.7894736842105263
    Random Forest model accuracy using Robust Scaler: 0.6842105263157895
    Random Forest model accuracy using Normalizer: 0.7894736842105263
    """
if __name__ == "__main__":
    main()
