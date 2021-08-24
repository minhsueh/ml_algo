import numpy as np
import pandas as pd
from qpsolvers import solve_qp

class SVM():
    def __init__(self):
        self.alpha = None

    def rbf(self, x, y):
        #gamma = 1
        return np.exp(-1.0*np.dot(np.subtract(x,y).T,np.subtract(x,y)))

    def transform(self,X, x2):
        K = np.zeros([X.shape[0],x2.shape[0]])
        for i in range(X.shape[0]):
            for j in range(x2.shape[0]):
                K[i,j] = self.rbf(X[i],x2[j])
        return(K)

    def fit(self, x_train, y_train):
        N = x_train.shape[0]


        P = np.outer(y_train,y_train) * self.transform(x_train, x_train)

        q = -np.ones((N, 1))
        G = -np.eye(N)
        h = np.zeros((N))
        A = y_train.reshape((1, -1))
        b = np.array([0.])
        self.alpha = solve_qp(P, q, A = A, b = b, G = G, h = h, solver = 'cvxopt')

        self.S = (self.alpha > 1e-3).flatten()
        self.x_s = x_train[self.S]
        self.y_s = y_train[self.S]
        print("=======")
        print("support vector:")
        print(self.x_s)


        self.b = np.mean(self.y_s - np.dot((self.y_s * self.alpha[self.S]).T, self.transform(self.x_s, self.x_s)))
        print("=======")
        print("b:")
        print(self.b)

    def predict(self, x_test):
        return(np.sign(np.dot((self.y_s * self.alpha[self.S]).T, self.transform(self.x_s, x_test)) + self.b))



def main():
    
    df = np.genfromtxt('linsep.txt', delimiter=',')
    x_train = df[:, 0:2]
    y_train = df[:, 2]
    y_train = y_train

    svm = SVM()
    svm.fit(x_train, y_train)
    #print(svm.predict(x_train))

if __name__ == '__main__':
    main()