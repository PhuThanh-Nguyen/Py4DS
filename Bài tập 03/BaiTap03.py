import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics

def decisionTreeModel(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)

def supportVectorMachine(X_train, X_test, y_train, y_test):
    clf = SVC(kernel = 'poly')
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)

def stochasticGradientDescent(X_train, X_test, y_train, y_test):
    clf = SGDClassifier(loss = 'log')
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)

def naiveBayesBernoulli(X_train, X_test, y_train, y_test):
    clf = BernoulliNB()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)

def randomForest(X_train, X_test, y_train, y_test):
    rdf = RandomForestClassifier()
    rdf.fit(X_train, y_train)
    y_pred = rdf.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)

def main():
    data = pd.read_csv('../dataset/mushrooms.csv')
    label_encoder = LabelEncoder()
    for column in data.columns:
        data[column] = label_encoder.fit_transform(data[column])
    y = data['class']
    X = data.drop(['class'], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    # Decision Tree Classifier
    decisionTreeAccuracy = decisionTreeModel(X_train, X_test, y_train, y_test)
    print(f'CART (Tree prediction) Accuracy = {decisionTreeAccuracy}')
    # Support Vector Machines
    svcAccuracy = supportVectorMachine(X_train, X_test, y_train, y_test)
    print(f'Support Vector Machine Accuracy = {svcAccuracy}')
    # Stochastic Gradient Descent
    sgdAccuracy = stochasticGradientDescent(X_train, X_test, y_train, y_test)
    print(f'Stochastic Gradient Descent Accuracy = {sgdAccuracy}')
    # Bernoulli Naive Bayes
    bernoulliAccuracy = naiveBayesBernoulli(X_train, X_test, y_train, y_test)
    print(f'Bernoulli Naive Bayes Accuracy = {bernoulliAccuracy}')
    # Random Forest
    randomForestAccuracy = randomForest(X_train, X_test, y_train, y_test)
    print(f'Random Forest Accuracy = {randomForestAccuracy}')

if __name__ == '__main__':
    main()