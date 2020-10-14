# Add code here
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import eli5
from eli5.sklearn import PermutationImportance

def decisionTreeModel(X_train,X_test,y_train,y_test):
    clf = DecisionTreeClassifier()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    perm = PermutationImportance(clf,random_state=1).fit(X_test,y_test)
    eli5.show_weights(perm,feature_names=X_test.columns.tolist())
    return metrics.accuracy_score(y_pred,y_test)

def main():
    data_path = '../diabetes.csv'
    col_names = ['pregnant', 'glucose','bp','skin','isulin','bmi','pedigree','age','label']
    df = pd.read_csv(data_path,header=0, names=col_names)
    # Feature selection
    feature_cols = ['pregnant','isulin','bmi','age','glucose','bp','pedigree']
    X = df[feature_cols]
    y = df.label
    # Split data 
    X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)
    # Build model
    decisionTreeAccuracy = decisionTreeModel(X_train,X_test,y_train,y_test)
    print(f'CART (Tree prediction) Accuracy = {decisionTreeAccuracy}')

if __name__ == '__main__':
    main()


