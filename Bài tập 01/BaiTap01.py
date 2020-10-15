import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import BernoulliNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics

def decisionTreeModel(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)

def ridgeClassifierModel(X_train, X_test, y_train, y_test):
    clf = RidgeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)

def stochasticGradientDescent(X_train, X_test, y_train, y_test):
    clf = SGDClassifier(loss = 'log', penalty="elasticnet", random_state = 1)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)

def naiveBayesBernoulli(X_train, X_test, y_train, y_test):
    clf = BernoulliNB()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)

def KNNModel(X_train, X_test, y_train, y_test):
    neigh = KNeighborsClassifier(n_neighbors = 3, weights = 'distance')
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)

def main():
    path = '../dataset/spam.csv'
    data = np.genfromtxt(path, delimiter = ',')
    X, y = data[:, :-1], data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 29)
    # Decision Tree Classifier
    decisionTreeAccuracy = decisionTreeModel(X_train, X_test, y_train, y_test)
    print(f'CART (Tree prediction) Accuracy = {decisionTreeAccuracy}')
    # Support Vector Machines
    ridgeAccuracy = ridgeClassifierModel(X_train, X_test, y_train, y_test)
    print(f'Ridge Classifier Accuracy = {ridgeAccuracy}')
    # Stochastic Gradient Descent
    sgdAccuracy = stochasticGradientDescent(X_train, X_test, y_train, y_test)
    print(f'Stochastic Gradient Descent Accuracy = {sgdAccuracy}')
    # Bernoulli Naive Bayes
    bernoulliAccuracy = naiveBayesBernoulli(X_train, X_test, y_train, y_test)
    print(f'Bernoulli Naive Bayes Accuracy = {bernoulliAccuracy}')
    #
    knnAccuracy = KNNModel(X_train, X_test, y_train, y_test)
    print(f'KNN Accuracy = {knnAccuracy}')

if __name__ == '__main__':
    main()