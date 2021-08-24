#Name: Min-Hsueh Chiu
#USCID: 9656645860

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

class Logistic_Regression():
    def __init__(self, x, y, max_iter = 7000, alpha = 1, tol = 0.001):
        #y.shape = (N,)
        #x.shape = (N, d+1)
        #w.shape = (d+1,)

        if not isinstance(x, np.ndarray) or not isinstance(x, np.ndarray):
            raise Exception('x and y should by numpy')
        self.alpha = alpha
        self.x = np.concatenate((np.ones(shape = (x.shape[0], 1)), x), axis = 1)
        self.w = np.random.uniform(size = (self.x.shape[1], 1))
        self.y = y.reshape(len(y), -1)
        self.N = x.shape[0]
        self.tol = tol
        self.max_iter = max_iter

    def sigmoid(self, s):
        return(np.exp(s)/(1 + np.exp(s)))

    def fit(self):
        A = np.dot(np.dot(self.x.T, self.y), self.w.T)
        E = 1/self.N*np.log(1+np.exp(-A))
        counter = 0
        while(E.all() > self.tol):
            if counter == self.max_iter:
                return()
            #A = np.dot(np.dot(self.x.T, self.y), self.w.T)
            #deltaE = -1/self.N*np.dot(self.sigmoid(A), np.dot(self.x.T, self.y))
            deltaE = -1/self.N*np.dot(1/(1+np.exp(A)), np.dot(self.x.T, self.y))
            self.w -= self.alpha * deltaE
            A = np.dot(np.dot(self.x.T, self.y), self.w.T)
            E = 1/self.N*np.log(1+np.exp(-A))
            counter += 1




    def predict(self, x_test):
        x_test = np.concatenate((np.ones(shape = (x_test.shape[0], 1)), x_test), axis = 1)
        #return(1/(1 + np.exp(-np.dot(x_test, self.w))))
        return(self.sigmoid(np.dot(x_test, self.w)))




def main():
    df = pd.read_csv('classification.txt', header = None)
    y = df[4].to_numpy()
    x = df.drop([3, 4], axis = 1).to_numpy()   

    lr =  Logistic_Regression(x, y, 1)
    lr.fit()
    predict_y_raw = lr.predict(x)
    #report:
    predict_y = []
    for i in predict_y_raw:
        if i >= 0.5:
            predict_y.append(+1)
        else:
            predict_y.append(-1)
    correct = 0
    for i in range(len(predict_y)):
        if predict_y[i] == y[i]:
            correct += 1

    print('accuracy: ' + str(correct/len(y)))
    print(lr.w)



if __name__ == '__main__':
    main()