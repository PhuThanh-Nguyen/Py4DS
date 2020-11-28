
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest, chi2, SelectFromModel, RFE
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
def ConstantFeatures(X_train, X_test, threshold = 0):
    '''
        Removing Constant Features using Variance Threshold
        ------------------
        Parameters:
            threshold: float, default = 0
                threshold parameter to identify the variable as constant
            X_train, X_test: DataFrame object
                Training and testing data to evaluate 
        ------------------
        Returns: 
            train data, test data after applying filter methods
    '''
    # Create the VarianceThreshold object.
    vs_constant = VarianceThreshold(threshold)

    # Select the numerical columns only.
    numerical_x_train = X_train[X_train.select_dtypes([np.number]).columns]

    # Fit the object to our data.
    vs_constant.fit(numerical_x_train)

    # Get the constant colum names.
    constant_columns = [column for column in numerical_x_train.columns
                        if column not in numerical_x_train.columns[vs_constant.get_support()]]

    # Detect constant categorical variables.
    constant_cat_columns = [column for column in X_train.columns \
                            if (X_train[column].dtype == "O" and len(X_train[column].unique())  == 1 )]

    # Concatenating the two lists.
    all_constant_columns = constant_cat_columns + constant_columns

    # Drop the constant columns
    X_train = X_train.drop(labels = all_constant_columns, axis = 1)
    X_test = X_test.drop(labels = all_constant_columns, axis = 1)
    
    return X_train, X_test

def correlationFilterMethod(X_train, X_test, thres = 0.8):
    '''
        Perform Correlation filter method
        ------------------
        Parameters:
            X_train, X_test: DataFrame objects
                Training and testing data
            thres: float, default: 0.8
                Threshold of absolute value of correlation, must between 0 and 1
        ------------------
        Returns:
            X_train, X_test after applying correlation filter method
    '''
    # Creating set to hold the correlated features
    corr_features = set()

    # Create the correlation matrix (default to pearson)
    corr_matrix = X_train.corr()

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > thres:
                colname = corr_matrix.columns[i]
                corr_features.add(colname)
                
    X_train = X_train.drop(labels = corr_features, axis = 1)
    X_test = X_test.drop(labels = corr_features, axis = 1)
    return X_train, X_test

def duplicateFeatures(X_train, X_test):
    '''
        Drop duplicated columns from traning, testing data
        ------------------
        Parameters:
            X_train, X_test: DataFrame objects
                Training and testing data
        ------------------
        Returns:
            X_train, X_test after dropping duplicated columns
    '''
    # Transpose the feature matrice
    train_features_T = X_train.T

    # Select the duplicated features columns names
    duplicated_columns = train_features_T[train_features_T.duplicated()].index.values

    # Drop those columns
    X_train = X_train.drop(labels = duplicated_columns, axis = 1)
    X_test = X_test.drop(labels = duplicated_columns, axis = 1)
    return X_train, X_test

def PCAMethod(num_components, X_train):
    '''
        Perform Principle Component Analysis (PCA) Method
        ------------------
        Parameters:
            num_components:	int
                Number of components to keep
            X_train, X_test: DataFrame objects
        ------------------
        Returns:
            Training and testing data after applying PCA Method
    '''
    pca = PCA(n_components = num_components)
    pca.fit(X_train)
    return pca.transform(X_train)

def quasiConstantFeatures(X_train, X_test, thres = 0.98):
    
    '''
        Perform Quasi-constant feature selection
        ------------------
        Parameters:
            X_train, X_test: DataFrame objects
                Training and testing data
            thres: float, between 0 and 1
                threshold to drop feature
        ------------------
        Returns:
            X_train, X_test after applying Quasi - Constant feature selection
    '''
    
    # Create empty list
    quasi_constant_feature = []

    # Loop over all the columns
    for feature in X_train.columns:
        # Calculate the ratio.
        predominant = (X_train[feature].value_counts() / np.float(len(X_train))).sort_values(ascending=False).values[0]
        
        # Append the column name if it is bigger than the threshold
        if predominant >= thres:
            quasi_constant_feature.append(feature)   

    # Drop the quasi constant columns
    X_train = X_train.drop(labels = quasi_constant_feature, axis = 1)
    X_test = X_test.drop(labels = quasi_constant_feature, axis = 1)
    
    return X_train, X_test

def duplicateFeatures(X_train, X_test):
    '''
        Drop duplicated columns from traning, testing data
        ------------------
        Parameters:
            X_train, X_test: DataFrame objects
                Training and testing data
        ------------------
        Returns:
            X_train, X_test after dropping duplicated columns
    '''
    # Transpose the feature matrice
    train_features_T = X_train.T

    # Select the duplicated features columns names
    duplicated_columns = train_features_T[train_features_T.duplicated()].index.values

    # Drop those columns
    X_train = X_train.drop(labels = duplicated_columns, axis = 1)
    X_test = X_test.drop(labels = duplicated_columns, axis = 1)
    return X_train, X_test

def PCAMethod(num_components, X_train, X_test):
	'''
		Perform Principle Component Analysis (PCA) Method
		------------------
		Parameters:
			num_components:	int
				Number of components to keep
			X_train, X_test: DataFrame objects
		------------------
		Returns:
			Training and testing data after applying PCA Method
	'''
	pca = PCA(n_components = num_components)
	pca.fit(X_train)
	return pca.transform(X_train), pca.transform(X_test)

def randomForestModel(X_train, X_valid, y_train, y_valid):
    '''
    Performs Random Forest Model using sklearn library
    -----------------
    Parameters:
        X_train, X_dev, y_train, y_dev: array-like
            Training and development data
    -----------------
    Returns:
        A tuple (accuracy, score) where `accuracy` is the accuracy of model based on validation data and
        `score` is the score of model based on training data
    '''
    clf = RandomForestClassifier(random_state = 0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    return accuracy_score(y_pred, y_valid), clf.score(X_train, y_train)

def main():
    # ********************************* EDA ************************************
    path_train = '/home/duc-hoang/Documents/Junior-Year/Py4DS/LAB/Lab 06/dataset/train.csv'
    path_test = '/home/duc-hoang/Documents/Junior-Year/Py4DS/LAB/Lab 06/dataset/test.csv'
    data_train = pd.read_csv(path_train)
    data_test = pd.read_csv(path_test)

    print('*'*50 + ' Original Train Data ' + '*'*50)
    print(data_train.head())
    print(f'>> Data shape: {data_train.shape}')
    print('>>' + ' Data info:')
    print(data_train.info())
    print('>>' + ' Data describe:')
    print(data_train.describe())

    print('*'*50 + ' Original Test Data ' + '*'*50)
    print(data_test.head())
    print(f'>> Data shape: {data_test.shape}')
    print('>>' + ' Data info:')
    print(data_test.info())
    print('>>' + ' Data describe:')
    print(data_test.describe())

    # Checking for NaNs or empty cells by features of data_train 
    print('*'*20 + ' Missing values percentage ' + '*'*20)
    percent_missing = data_train.isnull().sum() * 100 / len(data_train)
    missing_value_df_train = pd.DataFrame({'column_name': data_train.columns,
                                           'percent_missing': percent_missing})
    print(missing_value_df_train)

    # Checking for NaNs or empty cells by features of data_test
    print('*'*20 + ' Missing values percentage ' + '*'*20)
    percent_missing = data_test.isnull().sum() * 100 / len(data_test)
    missing_value_df_test = pd.DataFrame({'column_name': data_test.columns,
                                           'percent_missing': percent_missing})
    print(missing_value_df_test)
    # fill missing values
    data_train = data_train.apply(lambda x:x.fillna(x.value_counts().index[0]))
    data_test = data_test.apply(lambda x:x.fillna(x.value_counts().index[0]))

    print('*'*50 + ' Train data after filling missing values ' + '*'*50)
    print(data_train.head())
    print(f'>> Data shape: {data_train.shape}')
    print('>>' + ' Data info:')
    print(data_train.info())
    print('>>' + ' Data describe:')
    print(data_train.describe())

    print('*'*50 + ' Test data after filling missing values ' + '*'*50)
    print(data_test.head())
    print(f'>> Data shape: {data_test.shape}')
    print('>>' + ' Data info:')
    print(data_test.info())
    print('>>' + ' Data describe:')
    print(data_test.describe())

    # Detect outliers and remove
    Q1 = data_train.quantile(0.25)
    Q3 = data_train.quantile(0.75)
    IQR = Q3 - Q1
    data_train = data_train[~((data_train < (Q1 - 1.5 * IQR)) | (data_train > (Q3 + 1.5 * IQR))).any(axis=1)]

    Q1 = data_test.quantile(0.25)
    Q3 = data_test.quantile(0.75) 
    IQR = Q3 - Q1
    data_test = data_test[~((data_test < (Q1 - 1.5 * IQR)) | (data_test > (Q3 + 1.5 * IQR))).any(axis=1)]

    # print('*'*50 + ' Train data after remove outliers ' + '*'*50)
    # print(data_train.head())
    # print(f'>> Data shape: {data_train.shape}')
    # print('>>' + ' Data info:')
    # print(data_train.info())
    # print('>>' + ' Data describe:')
    # print(data_train.describe())

    # print('*'*50 + ' Test data after remove outliers ' + '*'*50)
    # print(data_test.head())
    # print(f'>> Data shape: {data_test.shape}')
    # print('>>' + ' Data info:')
    # print(data_test.info())
    # print('>>' + ' Data describe:')
    # print(data_test.describe())

    # ------------------- Costant Features method -------------------
    X = data_train.drop('target',axis=1)
    y = data_train['target']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=0)

    X_test = data_test

    X_train_new, X_valid_new = ConstantFeatures(X_train, X_valid)  
    val_score1, score1 = randomForestModel(X_train_new, X_valid_new, y_train, y_valid)

    # ------------------- Correlation Filter method ------------------

    X_train_new, X_valid_new = correlationFilterMethod(X_train, X_valid)
    val_score2, score2 = randomForestModel(X_train_new, X_valid_new, y_train, y_valid)

    # ------------------- Quasi constant method -------------------

    X_train_new, X_valid_new = quasiConstantFeatures(X_train, X_valid)
    val_score3, score3 = randomForestModel(X_train_new, X_valid_new, y_train, y_valid)

    # ------------------- Duplicate method -------------------

    X_train_new, X_valid_new = duplicateFeatures(X_train, X_valid)
    val_score4, score4 = randomForestModel(X_train_new, X_valid_new, y_train, y_valid)

    # ------------------- PCA ----------------------------------
    X_train_new, X_valid_new = PCAMethod(7, X_train, X_valid)
    val_score5, score5 = randomForestModel(X_train_new, X_valid_new, y_train, y_valid)

    print(f'>> Random Forest Model accuracy using Constant features filter: ({val_score1}, {score1})')
    print(f'>> Random Forest Model accuracy using Correlation filter method: ({val_score2}, {score2})')
    print(f'>> Random Forest Model accuracy using Quasi Constant method: ({val_score3}, {score3})')
    print(f'>> Random Forest Model accuracy using Duplicate method: ({val_score4}, {score4})')
    print(f'>> Random Forest Model accuracy using PCA method: ({val_score5}, {score5})')

    '''
    with random_state=0:
        >> Random Forest Model accuracy using Constant features filter: (0.5, 1.0)
        >> Random Forest Model accuracy using Correlation filter method: (0.75, 1.0)
        >> Random Forest Model accuracy using Quasi Constant method: (0.5, 1.0)
        >> Random Forest Model accuracy using Duplicate method: (0.5, 1.0)
        >> Random Forest Model accuracy using PCA method: (0.75, 1.0)
    '''
if __name__ == "__main__":
    main()
