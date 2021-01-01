import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

def sigmoid(X):
    """
    X: an numpy array
    return sigmoid function of each element of X
    """
    return 1/(1 + np.exp(-X))

class LogisticRegressionModel:
    
    def __init__(self, bias = True, random_state = 0, lambd = 0, learning_rate = 0.01, max_iter = 100, thresh = 0.5):
        '''
            Create LogisticRegressionModel instance
            ------------------------
            Parameters:
                
                bias: bool, default True
                    If True, when call .fit, bias term will be added
                
                random_state: int, default 0
                    Random State to initial parameters
                
                lambd: float, default 0
                    Regulization coefficent on l2 norm , if set to default, 
                    a normal Logistic Regression without regulization will be created
                
                learning_rate: float, default 0.01
                    Learning rate to update parameters
                
                max_iter: int, default 100
                    Maximum number of iterations
                
                thresh: float, default 0.5
                    Threshold to classify, if predicted value less than thresh then it will classify as 0, otherwise 1
                    Thresh must between 0 and 1
            ------------------------
            Returns: None
                 
        '''
        
        self.bias, self.rnd_state, self.rate, self.maxIter, self.lambd, self.thresh = bias, random_state, learning_rate, max_iter,\
                                                                                        lambd, thresh
    
    def fit(self, X_train, y_train):
        
        '''
            Fit model to training dataset
            -----------------------
            Parameters:
                X_train, y_train: numpy.ndarray
                    Training dataset with corresponding label
            -----------------------
            Returns: LogisticRegressionModel instance
                self
        '''
        
        if len(X_train.shape) == 1:
            X = X_train.reshape((X_train.shape[0], 1))
        else:
            X = X_train.copy()

        rnd = np.random.RandomState(self.rnd_state)
        
        if self.bias is True:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        y_train = y_train.reshape((y_train.shape[0], 1))

        self.w = rnd.normal(size = X.shape[1]).reshape((X.shape[1], 1))
        
        N, d = X.shape[0], X.shape[1]
        w_old = self.w
        # store history of loss in loss_hist
        # loss_hist = [loss(self.w, X, y_train, self.lambd)]
        ep = 0
        while ep < self.maxIter:
            ep += 1
            mix_ids = rnd.permutation(N)
            for i in mix_ids:
                xi = X[i].reshape((X.shape[1],1))
                yi = y_train[i]
                yHat_i = sigmoid(xi.T.dot(self.w))
                # update
                self.w = self.w - self.rate*((yHat_i - yi)*xi + self.lambd*self.w)
                # loss_hist.append(loss(self.w, X, y_train, self.lambd))
                if np.linalg.norm(self.w - w_old) < 1e-6:
                    break
                w_old = self.w
        return self
    def predict(self, X_test):
        
        '''
            Get predicted values on testing dataset
            -----------------------
            Parameters:
                X_test: numpy.ndarray
                    Testing dataset
            -----------------------
            Returns: numpy.ndarray
                1D label array predicted from testing dataset
        '''
        
        if len(X_test.shape) == 1:
            X = X_test.reshape((X_test.shape[0], 1))
        else:
            X = X_test.copy()
        
        if self.bias is True:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        y_predict = sigmoid(X.dot(self.w))
        
        y_predict = np.where(y_predict < self.thresh, 0, 1)
        
        return y_predict.flatten()
    def predict_proba(self, X_test):
        
        '''
            Get probability predicted from testing dataset
            -----------------------
            Parameters:
                X_test: numpy.ndarray
                    Testing dataset
            -----------------------
            Returns: numpy.ndarray
                1D probability array predicted from testing dataset
        '''
        
        if len(X_test.shape) == 1:
            X = X_test.reshape((X_test.shape[0], 1))
        else:
            X = X_test.copy()
        
        if self.bias is True:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        y_predict = sigmoid(X.dot(self.w))
        
        return y_predict.flatten()

def main():
    path = '/home/duc-hoang/Downloads/data.csv'
    data = pd.read_csv(path)
    X = data['x'].to_numpy()
    y = data['y'].to_numpy()

    clf = LogisticRegressionModel(random_state = 0, learning_rate = 0.1, lambd = 0.01, max_iter = 50)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)
    a, b = clf.w
    print(f'>> Coefficent and intercept produced by LogisticRegressionModel: {a},{b}')
    
    # So sánh nghiệm tìm được với nghiệm của thư viện sklearn
    clf_sklearn = LogisticRegression(random_state = 0, solver = 'sag', C = 2, max_iter = 50).fit(X.reshape((X.shape[0], 1)), y)
    y_proba_sklearn = clf_sklearn.predict_proba(X.reshape((X.shape[0], 1)))
    
    print(f'>> Coefficent and intercept produced by sklearn: {clf_sklearn.intercept_}, {clf_sklearn.coef_}')
    
    fig, ax = plt.subplots()
    ax.scatter(X[y == 0], np.zeros_like(X[y == 0]), color = 'red', marker = 'x')
    ax.scatter(X[y == 1], np.ones_like(X[y == 1]), color = 'blue', marker = 'o')
    ax.plot(X, y_proba, color = 'magenta', label = 'SGD solution')
    ax.plot(X, y_proba_sklearn[:, 1], color = 'green', label = 'sklearn solution')
    ax.axvline(-a/b, 0, 1, color = 'k')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()
