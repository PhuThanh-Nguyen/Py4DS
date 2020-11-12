'''
    Python for Data Science: Laboratory 04 - Excercise 01
    Perform an Exploratory Data Analysis (EDA), Data cleaning, 
        Building clustering models for prediction, Presenting results using dataset 'spam.csv'
'''
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.model_selection import train_test_split
from sklearn import metrics

def kMeanCluster(n_clusters, X_train, X_test, y_test, random_num):
    '''
    Performs KMeans clustering using sklearn library
    Parameters:
            n_clusters: int
                The number of clusters to form as well as the number of centroids to generate.
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
    cluster = AgglomerativeClustering(n_clusters = n_clusters, affinity = 'l1', linkage = 'average').fit(X_train)
    y_pred = cluster.fit_predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)

def BIRCHClustering(n_clusters, X_train, X_test, y_test):
    cluster = Birch(n_clusters = n_clusters)
    cluster.fit(X_train)
    y_pred = cluster.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)
def main():
    # Read data
    path = '../dataset/spam_original.csv'
    data = pd.read_csv(path)

    # Show original data info
    print('>>>>>> ORIGINAL DATA: ')
    print(data.head())
    print(f'>>>>>> DATA SHAPE: {data.shape}')
    print('>>>>>> DATA INFO:')
    print(data.info())

    # Detect missing values
    percent_missing = data.isnull().sum() * 100 / len(data)
    missing_value_df = pd.DataFrame({'column_name': data.columns,'percent_missing': percent_missing})
    print('>>>>>> MISSING VALUES PERCENTAGE: ')
    print(missing_value_df)

    # Remove space char in column names:
    data.columns = data.columns.str.replace(' ','')

    # Drop duplicate values
    print('>>>>>> DATA BEFORE DROP DUPLICATE VALUES: ')
    print(f'>>>>>> DATA SHAPE: {data.shape}')
    print('>>>>>>> DATA INFO:')
    print(data.info())
    print(data.describe())
    data.drop_duplicates(subset = data.columns.values[:-1], keep='first', inplace=True)
    print('>>>>>> DATA AFTER DROP DUPLICATE VALUES: ')
    print(f'>>>>>> DATA SHAPE: {data.shape}')
    print('>>>>>>> DATA INFO:')
    print(data.info())
    print(data.describe())
    
    # Detect and drop outliers
    print('>>>>>> DATA BEFORE REMOVING OUTLIERS')
    print(f'>>>>>> DATA SHAPE: {data.shape}')
    print('>>>>>> DATA INFO')
    print(data.info())
    print(data.describe())
    
    fig, axes = plt.subplots(ncols=9, nrows=7, figsize=(15, len(data.columns)/2))
    for i, c in zip(axes.flatten(), data.columns):
        sns.boxplot(data[c], ax = i)
    fig.subplots_adjust(hspace=.5)
    fig.suptitle('Before removing outliers', fontsize = 16)
    plt.savefig('Before_remove_outliers.png')

    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

    fig, axes = plt.subplots(ncols=9, nrows=7, figsize=(15, len(data.columns)/2))
    for i, c in zip(axes.flatten(), data.columns):
        sns.boxplot(data[c], ax = i)
    fig.subplots_adjust(hspace=.5)
    fig.suptitle('After removing outliers', fontsize = 16)
    plt.savefig('After_remove_outliers.png')

    print('>>>>>> DATA AFTER REMOVING OUTLIERS')
    print(f'>>>>>> DATA SHAPE: {data.shape}')
    print('>>>>>> DATA INFO')
    print(data.info())
    print(data.describe())
    
    # Prepare data for model
    X = data.drop('spam',axis=1)
    y = data['spam']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)

    # Choose parameters for training model 
    randomstate = 0
    number_clusters = 2

    # Using K-Mean clustering
    accuracy = kMeanCluster(number_clusters, X_train, X_test, y_test, randomstate)
    print(f'Accuracy using KMeans Clustering: {accuracy}')

    # Using Agglomerative Clustering 
    accuracy = agglomerativeClustering(number_clusters, X_train, X_test, y_test)
    print(f'Accuracy using Agglomerative Clustering: {accuracy}')
    
    # Using BIRCH Clustering
    accuracy = BIRCHClustering(number_clusters, X_train, X_test, y_test)
    print(f'Accuracy using BIRCH Clustering: {accuracy}')

    '''
    With random_state == 0 and number of clusters == 2:
        Accuracy using KMeans Clustering: 0.896551724137931
        Accuracy using Agglomerative Clustering: 0.896551724137931
        Accuracy using BIRCH Clustering: 0.896551724137931
    '''
if __name__ == "__main__":
    main()
