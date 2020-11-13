'''
    Python for Data Science: Laboratory 04 - Excercise 02
    Perform an Exploratory Data Analysis (EDA), Data cleaning, 
        Building clustering models for prediction, Presenting results using dataset 'diabetes.csv'
'''
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, MiniBatchKMeans
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')
def kMeansClustering(n_clusters, X_train, X_test, y_test, random_num):
    '''
    Performs KMeans clustering using sklearn library
    Parameters:
            n_clusters: int
                The number of clusters to form as well aAffinityPropagations the number of centroids to generate.
            X_train: array-like
                Training instances to cluster
            X_test: array-like
                New (unseen) data to predict
            y_test: array-like
                Ground truth for X_test
    '''
    cluster = KMeans(n_clusters=n_clusters, random_state=random_num).fit(X_train)
    y_pred = cluster.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)

def agglomerativeClustering(n_clusters, X_train, X_test, y_test):
    '''
    Performs AgglomerativeClustering using sklearn library
    Parameters:
            n_clusters: int
                The number of clusters to find
            X_train: array-like
                Training instances to cluster
            X_test: array-like
                New (unseen) data to predict
            y_test: array-like
                Ground truth for X_test
    '''
    cluster = AgglomerativeClustering(n_clusters = n_clusters, affinity = 'l2', linkage = 'average').fit(X_train)
    y_pred = cluster.fit_predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)

def miniBatchKmeansClustering(n_clusters, random_num, X_train, X_test, y_test):
    '''
    Performs MiniBatchKmeansClustering using sklearn library
    Parameters:
            n_clusters: int
                The number of clusters to form as well as the number of centroids to generate
            random_num: int
                Determines random number generation for centroid initialization and random reassignment. 
                Use an int to make the randomness deterministic
            X_train: array-like
                Training instances to cluster
            X_test: array-like
                New (unseen) data to predict
            y_test: array-like
                Ground truth for X_test
    '''
    cluster = MiniBatchKMeans(n_clusters = n_clusters, random_state=random_num).fit(X_train)
    y_pred = cluster.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)

def BIRCHClustering(n_clusters, X_train, X_test, y_test):
    '''
    Perform BIRCHClustering using sklearn library
    Parameters:
            n_clusters: int
                Number of clusters after the final clustering step, which treats the subclusters from the leaves as new samples.
            X_train: array-like
                Training instances to cluster
            X_test: array-like
                New (unseen) data to predict
            y_test: array-like
                Ground truth for X_test 
    '''
    cluster = Birch(n_clusters = n_clusters)
    cluster.fit(X_train)
    y_pred = cluster.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)
def main():
    # Read data
    ######################## EDA #################################
    path = '../dataset/diabetes.csv'
    data = pd.read_csv(path)

    # Show original data info
    print('#'*50 + ' ORIGINAL DATA INFOMATION ' + '#'*50)
    print('>'*20 + f' Data shape: {data.shape} ')
    print('>'*20 + ' Data info')
    print(data.info())
    print(data.describe())

    # Checking for NaNs or empty cells by features
    percent_missing = data.isnull().sum() * 100 / len(data)
    missing_value_df = pd.DataFrame({'column_name': data.columns,
                                     'percent_missing': percent_missing})
    print('#'*50 + f' MISSING VALUE PERCENTAGE ' + '#'*50)
    print(missing_value_df)
    
    # Checking for duplicate values
    print('#'*50 + ' CHECKING DUPLICATE VALUES ' + '#'*50)
    duplicateRowsDF = data[data.duplicated()]
    total_duplicates = len(duplicateRowsDF)
    print(f"Duplicate Rows (total: {total_duplicates}) except first occurrence based on all columns are :")
    print(duplicateRowsDF)

    ########################## PREPROCESSING ###############################
    
    # Detect outliers
    fig, axes = plt.subplots(ncols=2, nrows=4, figsize=(10, len(data.columns)/2))
    for i, c in zip(axes.flatten(), data.columns):
        sns.boxplot(data[c], ax = i)
    fig.subplots_adjust(hspace=1)
    fig.suptitle('Before removing outliers', fontsize = 16)
    plt.savefig('Before_remove_outliers.png')

    print('#'*50 + ' DATA SHAPE BEFORE REMOVING OUTLIERS ' + '#'*50)
    print('>'*20 + f' Data shape: {data.shape}')
    print('>'*20 + ' Data info')
    print(data.info())
    print(data.describe())

    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

    fig, axes = plt.subplots(ncols=2, nrows=4, figsize=(10, len(data.columns)/2))
    for i, c in zip(axes.flatten(), data.columns):
        sns.boxplot(data[c], ax = i)
    fig.subplots_adjust(hspace=1)
    fig.suptitle('After removing outliers', fontsize = 16)
    plt.savefig('After_remove_outliers.png')

    print('#'*50 + ' DATA SHAPE AFTER REMOVING OUTLIERS ' + '#'*50)
    print('>'*20 + f' Data shape: {data.shape}')
    print('>'*20 + ' Data info')
    print(data.info())
    print(data.describe())
    
    # Remove space characters in column names
    data.columns = data.columns.str.replace(' ', '')

    # Prepare data for training model
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    # Using Robust Scaler
    scalerX = RobustScaler()
    X_train_RobustScaler = scalerX.fit_transform(X_train)
    X_test_RobustScaler = scalerX.transform(X_test)

    # Choose parameters for training model 
    randomstate = 17
    number_clusters = 2

    ###################################### BUILD MODEL ###########################
    # Using K-Mean clustering
    accuracy = kMeansClustering(number_clusters, X_train_RobustScaler, X_test_RobustScaler, y_test, randomstate)
    print(f'Accuracy using KMeans Clustering: {accuracy}')

    # Using Agglomerative Clustering 
    accuracy = agglomerativeClustering(number_clusters, X_train_RobustScaler, X_test_RobustScaler, y_test)
    print(f'Accuracy using Agglomerative Clustering: {accuracy}')
    
    # Using BIRCH Clustering
    accuracy = BIRCHClustering(number_clusters, X_train_RobustScaler, X_test_RobustScaler, y_test)
    print(f'Accuracy using BIRCH Clustering: {accuracy}')

    # Using MiniBatch Clustering
    accuracy = miniBatchKmeansClustering(number_clusters, randomstate, X_train_RobustScaler, X_test_RobustScaler, y_test)
    print(f'Accuracy using MiniBatch Clustering: {accuracy}')

    '''
    With random_state == 17 and number_clusters == 2:
        Accuracy using KMeans Clustering: 0.75
        Accuracy using Agglomerative Clustering: 0.71875
        Accuracy using BIRCH Clustering: 0.7625
        Accuracy using MiniBatch Clustering: 0.76875
    '''
if __name__ == "__main__":
    main()