# Add code here
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def decisionTreeModel(X_train,X_test,y_train,y_test):
    mdl = DecisionTreeClassifier()
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
def main():

    data_path = '../xAPI-Edu-Data.csv'
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
    decisionTreeAccuracy = decisionTreeModel(X_train,X_test,y_train,y_test)
    print(f'CART (Tree prediction) Accuracy = {decisionTreeAccuracy}')

    randomForestAccuracy = randomForestModel(X_train,X_test,y_train,y_test)
    print(f'Random Forest prediction Accuracy = {randomForestAccuracy}')

    svcAccuracy = supportVectorMachine(X_train,X_test,y_train,y_test)
    print(f'SVM prediction Accuracy = {svcAccuracy}')

if __name__ == '__main__':
    main()