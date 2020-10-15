import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# import eli5
# from eli5.sklearn import PermutationImportance

def decisionTreeModel(max_depth, X_train,X_test,y_train,y_test):
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    return metrics.accuracy_score(y_pred,y_test)
def supportVectorMachine(X_train,X_test,y_train,y_test):
    clf = SVC(kernel='poly')
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    return metrics.accuracy_score(y_pred,y_test)
def randomForestModel(X_train,X_test,y_train,y_test):
    clf = RandomForestClassifier()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    return metrics.accuracy_score(y_pred,y_test)
def main():
    data_path = '/home/duc-hoang/Documents/Junior-Year/Python-For-Data-Science/Lab-01/Py4DS-Lab01/dataset/diabetes.csv'
    col_names = ['pregnant', 'glucose','bp','skin','isulin','bmi','pedigree','age','label']
    df = pd.read_csv(data_path,header=0, names=col_names)
    # Feature selection
    feature_cols = ['pregnant','isulin','bmi','age','glucose','bp','pedigree']
    X = df[feature_cols]
    y = df.label
    # Split data 
    X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)
    # Build model

    # Decision Tree model
    # compare error with differing values of max_depth
    # for max_depth in [5, 10, 15, 20]:
    #     my_error = decisionTreeModel(max_depth, X_train, X_test, y_train, y_test)
    #     print("Max depth: %d  \t\t Decision Tree prediction Accuracy:  %f" %(max_depth, my_error))
    
    decisionTreeAccuracy = decisionTreeModel(10,X_train,X_test,y_train,y_test)
    print(f'Decision Tree prediction Accuracy = {decisionTreeAccuracy}')
    
    # Random Forest model
    randomForestAccuracy = randomForestModel(X_train,X_test,y_train,y_test)
    print(f'Random Forest prediction Accuracy = {randomForestAccuracy}')

    svcAccuracy = supportVectorMachine(X_train,X_test,y_train,y_test)
    print(f'SVM prediction Accuracy = {svcAccuracy}')
    
if __name__ == '__main__':
    main()


