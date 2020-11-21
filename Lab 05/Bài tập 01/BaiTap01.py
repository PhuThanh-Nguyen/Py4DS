'''
    Perform an Exploratory Data Analysis (EDA), Data cleaning, Building models for prediction,
    Presenting resultsusing on the 'Titanic datasets'
'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def RandomForestModel(X_train, y_train, X_test, rd_num):
    '''
    Performs Random Forest model using sklearn library 
    Parameters:
            X_train: array-like
                Training instances to cluster
            y_train: array-like
                Training labels
            X_test: array-like
                New (unseen) data to predict
            rd_num: int
                Controls both the randomness of the bootstrapping of the samples used when building trees
    '''
    clf = RandomForestClassifier(random_state=rd_num).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf.score(X_train, y_train)


def DecisionTreeModel(X_train, y_train, X_test, rd_num):
    '''
    Performs Decision Tree model using sklearn library 
    Parameters:
            X_train: array-like
                Training instances to cluster
            y_train: array-like
                Training labels
            X_test: array-like
                New (unseen) data to predict
            rd_num: int
                Controls the randomness of the estimator
    '''
    clf = DecisionTreeClassifier(random_state=rd_num).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf.score(X_train, y_train)


def main():
    ########################## Exploratory Data Analysis (EDA) ###############################
    # Read data_train
    train_path = '../dataset/titanic_train.csv'
    test_path = '../dataset/titanic_test.csv'
    data_train = pd.read_csv(train_path)
    data_test = pd.read_csv(test_path)

    print('#'*50 + ' ORIGINAL DATA TRAIN ' + '#'*50)
    print('>'*10 + f' Data shape: {data_train.shape}')
    print('>'*10 + ' Data head:')
    print(data_train.head())
    print('>'*10 + ' Data info:')
    print(data_train.info())
    print('>'*10 + ' Data describe:')
    print(data_train.describe())

    print('#'*50 + ' ORIGINAL DATA TEST ' + '#'*50)
    print('>'*10 + f' Data test shape: {data_test.shape}')
    print('>'*10 + ' Data test head:')
    print(data_test.head())
    print('>'*10 + ' Data test info:')
    print(data_test.info())
    print('>'*10 + ' Data test describe:')
    print(data_test.describe())

    # Plot label of data train
    fig = plt.figure(figsize=(10, 8))
    ax = data_train['Survived'].value_counts(normalize=True).plot(kind='bar')
    fig.suptitle('Barplot of Survived (label)', fontsize=16)
    fig.savefig('Barplot-label.png')

    # Checking for NaNs or empty cells by features of data train
    print('#'*50 + f' MISSING VALUE PERCENTAGE OF DATA TRAIN' + '#'*50)
    percent_missing = data_train.isnull().sum() * 100 / len(data_train)
    missing_value_df_train = pd.DataFrame({'column_name': data_train.columns,
                                           'percent_missing': percent_missing})
    print(missing_value_df_train)
    print('#'*50 + ' DATA TRAIN INFOMATION BEFORE HANDLING MISSING VALUES ' + '#'*50)
    print('>'*10 + f' Data shape: {data_train.shape}')
    print('>'*10 + ' Data train head:')
    print(data_train.head())
    print('>'*10 + ' Data train info:')
    print(data_train.info())
    print('>'*10 + ' Data train describe:')
    print(data_train.describe())

    # Checking for NaNs or empty cells by features of data train
    print('#'*50 + f' MISSING VALUE PERCENTAGE OF DATA TEST' + '#'*50)
    percent_missing = data_test.isnull().sum() * 100 / len(data_test)
    missing_value_df_test = pd.DataFrame({'column_name': data_test.columns,
                                          'percent_missing': percent_missing})
    print(missing_value_df_test)
    print('#'*50 + ' DATA TEST INFOMATION BEFORE HANDLING MISSING VALUES ' + '#'*50)
    print('>'*10 + f' Data test shape: {data_test.shape}')
    print('>'*10 + ' Data test head:')
    print(data_test.head())
    print('>'*10 + ' Data test info:')
    print(data_test.info())
    print('>'*10 + ' Data test describe:')
    print(data_test.describe())
    # Handle missing values
    '''
    As we see from above frame, the column 'Cabin' has more 77% of missing values. We absolutely do not want to \n
    lead model to false direction, so drop this column is the best choice.
    '''
    data_train = data_train.drop('Cabin', axis=1)
    data_test = data_test.drop('Cabin', axis=1)
    # Next, we need to keep and fill missing values into the column 'Age' by mean, because this column has not too much missing values and
    # gets high correlated with our target.
    data_train['Age'] = data_train['Age'].fillna(29)
    data_test['Age'] = data_test['Age'].fillna(29)
    # Fill NaNs with the most frequent value from 'Embarked' column.
    data_train['Embarked'] = data_train['Embarked'].fillna(
        data_train['Embarked'].value_counts().index[0])
    data_test['Fare'] = data_test['Fare'].fillna(
        data_test['Fare'].value_counts().index[0])

    print('#'*50 + ' DATA TRAIN INFOMATION AFTER HANDLING MISSING VALUES ' + '#'*50)
    print('>'*10 + f' Data train shape: {data_train.shape}')
    print('>'*10 + ' Data train head:')
    print(data_train.head())
    print('>'*10 + ' Data train info:')
    print(data_train.info())
    print('>'*10 + ' Data train describe:')
    print(data_train.describe())

    print('#'*50 + ' DATA TEST INFOMATION AFTER HANDLING MISSING VALUES ' + '#'*50)
    print('>'*10 + f' Data test shape: {data_test.shape}')
    print('>'*10 + ' Data test head:')
    print(data_test.head())
    print('>'*10 + ' Data test info:')
    print(data_test.info())
    print('>'*10 + ' Data test describe:')
    print(data_test.describe())

    ########################## PREPROCESSING ###############################

    # Select and drop low weight features.
    low_weight_features = ['PassengerId', 'Name', 'Ticket', 'Embarked']
    data_train = data_train.drop(low_weight_features, axis=1)
    data_test = data_test.drop(low_weight_features, axis=1)
    print('#'*50 + ' DATA TRAIN SHAPE AFTER DROPPING LOW WEIGHT FEATURES ' + '#'*50)
    print('>'*10 + f' Data train shape: {data_train.shape}')
    print('>'*10 + ' Data train head:')
    print(data_train.head())

    print('#'*50 + ' DATA TEST SHAPE AFTER DROPPING LOW WEIGHT FEATURES ' + '#'*50)
    print('>'*10 + f' Data test shape: {data_test.shape}')
    print('>'*10 + ' Data test head:')
    print(data_test.head())

    # Extract categorical features
    print('#'*50 + ' DATA TRAIN INFO AFTER ENCODING CATEGORICAL FEATURES ' + '#'*50)
    print('>'*10 + f' Data train shape: {data_train.shape}')
    print('>'*10 + ' Data train head:')
    print(data_train.head())

    print('#'*50 + ' DATA TEST INFO AFTER ENCODING CATEGORICAL FEATURES ' + '#'*50)
    print('>'*10 + f' Data test: {data_test.shape}')
    print('>'*10 + ' Data test head:')
    print(data_test.head())

    # Encode data_train and data_test
    categorical_features = data_train.select_dtypes(
        include=[np.object]).columns
    label_enc = LabelEncoder()
    for col in categorical_features:
        data_train[col] = label_enc.fit_transform(data_train[col])
    for col in categorical_features:
        data_test[col] = label_enc.fit_transform(data_test[col])

    print('#'*50 + ' DATA TRAIN INFO AFTER ENCODING CATEGORICAL FEATURES ' + '#'*50)
    print('>'*10 + f' Data train shape: {data_train.shape}')
    print('>'*10 + ' Data train head:')
    print(data_train.head())

    print('#'*50 + ' DATA TEST INFO AFTER ENCODING CATEGORICAL FEATURES ' + '#'*50)
    print('>'*10 + f' Data test shape: {data_test.shape}')
    print('>'*10 + ' Data test head:')
    print(data_test.head())

    # Checking for duplicate values
    print('#'*50 + ' CHECKING DUPLICATE VALUES ON DATA TRAIN ' + '#'*50)
    duplicateRowsTrain = data_train[data_train.duplicated()]
    total_duplicates = len(duplicateRowsTrain)
    print(
        f"Duplicate Rows (Data Train) (total: {total_duplicates}) except first occurrence based on all columns are :")
    print(duplicateRowsTrain)
    print('#'*50 + ' DATA TRAIN INFO BEFORE DROPPING DUPLICATE VALUES ' + '#'*50)
    print('>'*10 + f' Data train shape: {data_train.shape}')
    print('>'*10 + ' Data train info:')
    print(data_train.info())

    print('#'*50 + ' CHECKING DUPLICATE VALUES ON DATA TEST ' + '#'*50)
    duplicateRowsTest = data_test[data_test.duplicated()]
    total_duplicates = len(duplicateRowsTest)
    print(
        f"Duplicate Rows (Data Test) (total: {total_duplicates}) except first occurrence based on all columns are :")
    print(duplicateRowsTest)
    print('#'*50 + ' DATA TEST INFO BEFORE DROPPING DUPLICATE VALUES ' + '#'*50)
    print('>'*10 + f' Data test shape: {data_test.shape}')
    print('>'*10 + ' Data test info:')
    print(data_test.info())

    # Drop duplicate values on data train and data test
    data_train.drop_duplicates(
        subset=data_train.columns.values[:-1], keep='first', inplace=True)
    print('#'*50 + ' DATA TRAIN INFO AFTER DROPPING DUPLICATE VALUES ' + '#'*50)
    print('>'*10 + f' Data train shape: {data_train.shape}')
    print('>'*10 + ' Data train info:')
    print(data_train.info())

    data_test.drop_duplicates(
        subset=data_test.columns.values[:-1], keep='first', inplace=True)
    print('#'*50 + ' DATA TEST INFO AFTER DROPPING DUPLICATE VALUES ' + '#'*50)
    print('>'*10 + f' Data test shape: {data_test.shape}')
    print('>'*10 + ' Data test info:')
    print(data_test.info())

    # Detect outliers
    '''
    We don't need to remove outliers within data test
    '''
    fig, axes = plt.subplots(
        ncols=2, nrows=4, figsize=(15, len(data_train.columns)))
    for i, c in zip(axes.flatten(), data_train.columns):
        sns.boxplot(data_train[c], ax=i)
    fig.subplots_adjust(hspace=1.5)
    fig.suptitle('Before removing outliers (train)', fontsize=16)
    plt.savefig('Before_remove_outliers_train.png')

    print('#'*50 + ' DATA TRAIN SHAPE BEFORE REMOVING OUTLIERS ' + '#'*50)
    print('>'*20 + f' Data train shape: {data_train.shape}')
    print('>'*20 + ' Data train info')
    print(data_train.info())
    print(data_train.describe())

    Q1 = data_train.quantile(0.25)
    Q3 = data_train.quantile(0.75)
    IQR = Q3 - Q1
    data_train = data_train[~((data_train < (Q1 - 1.5 * IQR)) | (data_train > (Q3 + 1.5 * IQR))).any(axis=1)]

    fig, axes = plt.subplots(
        ncols=2, nrows=4, figsize=(15, len(data_train.columns)))
    for i, c in zip(axes.flatten(), data_train.columns):
        sns.boxplot(data_train[c], ax=i)
    fig.subplots_adjust(hspace=1.5)
    fig.suptitle('After removing outliers (train)', fontsize=16)
    plt.savefig('After_remove_outliers_train.png')

    print('#'*50 + ' DATA TRAIN SHAPE AFTER REMOVING OUTLIERS ' + '#'*50)
    print('>'*20 + f' Data train shape: {data_train.shape}')
    print('>'*20 + ' Data train info')
    print(data_train.info())
    print(data_train.describe())

    # Prepare data for training model
    X_train = data_train.drop('Survived', axis=1)
    y_train = data_train['Survived']
    X_test = data_test

    # Build and train model
    accuracy = RandomForestModel(X_train, y_train, X_test, 645)
    print(f'>> Accuracy using Random Forest Classifier model: {accuracy}')

    accuracy = DecisionTreeModel(X_train, y_train, X_test, 645)
    print(f'>> Accuracy using Decision Tree Classifier model: {accuracy}')

    '''
    With random_state == 645:
        Accuracy using Random Forest Classifier model: 0.9855769230769231
        Accuracy using Decision Tree Classifier model: 0.9879807692307693	
    '''


if __name__ == "__main__":
    main()
