# Add code here
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split

def decisionTreeModel(X_train,X_test,y_train,y_test, maxdepth):
    mdl = DecisionTreeClassifier(max_depth=maxdepth)
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)
    return metrics.accuracy_score(y_pred,y_test)
def randomForestModel(X_train,X_test,y_train,y_test):
    mdl = RandomForestClassifier()
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)
    return metrics.accuracy_score(y_pred,y_test)
def supportVectorMachine(X_train,X_test,y_train,y_test):
    mdl = SVC(kernel='poly')
    mdl.fit(X_train,y_train)
    y_pred = mdl.predict(X_test)
    return metrics.accuracy_score(y_pred,y_test)
def sgdModel(X_train,X_test,y_train,y_test):
    mdl = SGDClassifier()
    mdl.fit(X_train,y_train)
    y_pred = mdl.predict(X_test)
    return metrics.accuracy_score(y_pred,y_test)
def LR_Model(X_train,X_test,y_train,y_test):
    mdl = LogisticRegression()
    mdl.fit(X_train,y_train)
    y_pred = mdl.predict(X_test)
    return metrics.accuracy_score(y_pred,y_test)
def main():

    data_path = '../dataset/xAPI-Edu-Data.csv'
    data = pd.read_csv(data_path)

    # Feature selection
    features = ['raisedhands', 'VisITedResources','AnnouncementsView','Discussion']
    X = data[features]
    y = data['Class']

    # Label Encoder
    enc = preprocessing.LabelEncoder()
    label_encoded = enc.fit_transform(y)
    y = label_encoded

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Build model

    # Decision Tree model

    # compare accuracy with differing values of max_depth
    # for max_depth in [50, 500, 5000, 50000]:
    #     my_mae = decisionTreeModel(X_train, X_test, y_train, y_test,max_depth)
    #     print("Max depth: %d  \t\t Decision Tree accuracy:  %f" %(max_depth, my_mae))
    decisionTreeAccuracy = decisionTreeModel(X_train,X_test,y_train,y_test,500) # take max_depth = 500
    print(f'Decision Tree prediction Accuracy = {decisionTreeAccuracy}')

    # Random Forest model
    randomForestAccuracy = randomForestModel(X_train,X_test,y_train,y_test)
    print(f'Random Forest prediction Accuracy = {randomForestAccuracy}')

    # SVM model
    svcAccuracy = supportVectorMachine(X_train,X_test,y_train,y_test)
    print(f'SVM prediction Accuracy = {svcAccuracy}')
    
    # SGD model
    sgdAccuracy = SGDClassifier(X_train,X_test,y_train,y_test)
    print(f'SGD prediction Accuracy = {sgdAccuracy}')

    # Linear Regression model
    lrAccuracy = LR_Model(X_train,X_test,y_train,y_test)
    print(f'SGD prediction Accuracy = {lrAccuracy}')
if __name__ == '__main__':
    main()