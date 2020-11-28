# Add code here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest, chi2, SelectFromModel, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

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

def chiSquareMethod(numFeatures, X_train, X_test, y_train, y_test):
	'''
		Perform Chi-Square method
		------------------
		Parameters:
			numFeatures: int
				Number of features to retain after filter
			X_train, X_test, y_train: DataFrame/Series objects
				Training data and testing data
		------------------
		Returns:
			Training data and testing data after applying chisquare method
	'''
	# Apply the chi2 score on the data and target (target should be binary).  
	selection_XTrain = SelectKBest(chi2, k = numFeatures).fit(X_train, y_train)
	selection_XTest = SelectKBest(chi2, k = numFeatures).fit(X_test, y_test)

	return selection_XTrain.transform(X_train), selection_XTest.transform(X_test)

def randomForestReg(X_train, X_test, y_train, y_test):
    reg = RandomForestRegressor(random_state=42).fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    return np.sqrt(mean_squared_error(y_test,y_pred))

def main():
    # ********************************* EDA ************************************
    path = '/home/duc-hoang/Documents/Junior-Year/Py4DS/LAB/Lab 06/dataset/data.csv'
    data = pd.read_csv(path)

    print('*'*50 + ' Original Data ' + '*'*50)
    print(data.head())
    print(f'>> Data shape: {data.shape}')
    print('>>' + ' Data info:')
    print(data.info())
    print('>>' + ' Data describe:')
    print(data.describe())

    # Checking for NaNs or empty cells by features of data train
    print('*'*20 + ' Missing values percentage ' + '*'*20)
    percent_missing = data.isnull().sum() * 100 / len(data)
    missing_value_df_train = pd.DataFrame({'column_name': data.columns,
                                           'percent_missing': percent_missing})
    print(missing_value_df_train)

    # fill missing values
    data = data.apply(lambda x:x.fillna(x.value_counts().index[0]))
    
    print('*'*50 + ' After filling missing values ' + '*'*50)
    print(data.head())
    print(f'>> Data shape: {data.shape}')
    print('>>' + ' Data info:')
    print(data.info())
    print('>>' + ' Data describe:')
    print(data.describe())

    # Checking duplicate values
    print('*'*50 + ' CHECKING DUPLICATE VALUES ON DATA ' + '*'*50)
    duplicateRowsTrain = data[data.duplicated()]
    total_duplicates = len(duplicateRowsTrain)
    print(
        f"Duplicate Rows (Data Train) (total: {total_duplicates}) except first occurrence based on all columns are :")
    print(duplicateRowsTrain)
    print(f'>> Data shape: {data.shape}')
    print('>>' + ' Data info:')
    print(data.info())
    print('>>' + ' Data describe:')
    print(data.describe())

    # Drop duplicate values
    data.drop_duplicates(subset = data.columns.values[:-1], keep='first', inplace=True)
    print('*'*50 + ' After droping duplicate values ' + '*'*50)
    print(f'>> Data shape: {data.shape}')
    print('>>' + ' Data info:')
    print(data.info())
    print('>>' + ' Data describe:')
    print(data.describe())

    # Encode label
    data['Release Clause'] = data['Release Clause'].replace(regex=True,to_replace=r'\D',value=r'').astype(float)

    # Detect and encode categorical features
    categorical_features = data.select_dtypes(include=[np.object]).columns
    label_enc = LabelEncoder()
    for col in categorical_features:
        data[col] = label_enc.fit_transform(data[col])
    
    print('*'*50 + ' After encoding categorical features ' + '*'*50)
    print(data.head())
    print(f'>> Data shape: {data.shape}')
    print('>>' + ' Data info:')
    print(data.info())
    print('>>' + ' Data describe:')
    print(data.describe())
    

    # Detect outliers and remove
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

    print('*'*50 + ' After remove outliers ' + '*'*50)
    print(data.head())
    print(f'>> Data shape: {data.shape}')
    print('>>' + ' Data info:')
    print(data.info())
    print('>>' + ' Data describe:')
    print(data.describe())

    # ------------------- Costant Features method -------------------
    # Split data    
    X, y = data.drop('Release Clause', axis = 1), data['Release Clause']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    print('*'*50 + ' Before apply constant features filter ' + '*'*50)
    print('>> X_train:')
    print(X_train)
    print('>> X_test:')
    print(X_test)
    # X_train_new and X_test_new must drop label
    X_train_new, X_test_new = ConstantFeatures(X_train, X_test)
    print('*'*50 + ' After apply constant features filter ' + '*'*50)
    print('>> X_train:')
    print(X_train_new)
    print('>> X_test:')
    print(X_test_new)

    # Model
    accuracy1 = randomForestReg(X_train_new, X_test_new, y_train, y_test)

    # ------------------- quasiConstantFeatures method -------------------
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    print('*'*50 + ' Before apply quasi - constant features filter ' + '*'*50)
    print('>> X_train:')
    print(X_train)
    print('>> X_test:')
    print(X_test)
    # X_train_new and X_test_new must drop label
    X_train_new, X_test_new = quasiConstantFeatures(X_train, X_test, 0.8)
    print('*'*50 + ' After apply quasi - constant features filter ' + '*'*50)
    print('>> X_train:')
    print(X_train_new)
    print('>> X_test:')
    print(X_test_new)

    # Model
    accuracy2 = randomForestReg(X_train_new, X_test_new, y_train, y_test)

    # ------------------- PCA method -------------------
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    print('*'*50 + ' Before apply PCA ' + '*'*50)
    print('>> X_train:')
    print(X_train)
    print('>> X_test:')
    print(X_test)
    # X_train_new and X_test_new must drop label
    X_train_new, X_test_new = PCAMethod(30, X_train, X_test)
    print('*'*50 + ' After apply PCA ' + '*'*50)
    print('>> X_train:')
    print(X_train)
    print('>> X_test:')
    print(X_test)
    # Model
    accuracy3 = randomForestReg(X_train_new, X_test_new, y_train, y_test)

    # ------------------- correlationFilterMethod method -------------------
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    print('*'*50 + ' Before apply correlation Filter Method ' + '*'*50)
    print('>> X_train:')
    print(X_train)
    print('>> X_test:')
    print(X_test)
    # X_train_new and X_test_new must drop label
    X_train_new, X_test_new = correlationFilterMethod(X_train, X_test)
    print('*'*50 + ' After apply PCA ' + '*'*50)
    print('>> X_train:')
    print(X_train)
    print('>> X_test:')
    print(X_test)
    accuracy4 = randomForestReg(X_train_new, X_test_new, y_train, y_test)

     # ------------------- Chi Square method -------------------
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    print('*'*50 + ' Before apply correlation Filter Method ' + '*'*50)
    print('>> X_train:')
    print(X_train)
    print('>> X_test:')
    print(X_test)
    # X_train_new and X_test_new must drop label
    X_train_new, X_test_new = chiSquareMethod(88, X_train, X_test, y_train, y_test)
    
    accuracy5 = randomForestReg(X_train_new, X_test_new, y_train, y_test)


    # Result
    print(f'>> Random Forest Regressor Model accuracy using Constant features filter: {accuracy1}')
    print(f'>> Random Forest Regressor Model accuracy using Quasi Constant filter: {accuracy2}')
    print(f'>> Random Forest Regressor Model accuracy using PCA: {accuracy3}')
    print(f'>> Random Forest Regressor Model accuracy using correlation Filter Method: {accuracy4}')
    print(f'>> Random Forest Regressor Model accuracy using chi-square Method: {accuracy5}')

    '''
    with random_state = 42:
        >> Random Forest Regressor Model accuracy using Constant features filter: 181.3938561495124
        >> Random Forest Regressor Model accuracy using Quasi Constant filter: 181.3938561495124
        >> Random Forest Regressor Model accuracy using PCA: 194.56772947667358
        >> Random Forest Regressor Model accuracy using correlation Filter Method: 176.3197263958726
        >> Random Forest Regressor Model accuracy using chi-square Filter Method: 181.6364678948609
    '''
if __name__ == "__main__":
    main()