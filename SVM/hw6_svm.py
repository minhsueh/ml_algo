import numpy as np
import pandas as pd
from qpsolvers import solve_qp

class SVM():
    def __init__(self):
        self.alpha = None

    def fit(self, x_train, y_train):
        N = x_train.shape[0]
        P = np.dot(y_train.reshape((len(y_train),-1))*x_train, (y_train.reshape((len(y_train),-1))*x_train).T)
        q = -np.ones((N, 1))
        G = -np.eye(N)
        h = np.zeros((N))
        A = y_train.reshape((1, -1))
        b = np.array([0.])
        self.alpha = solve_qp(P, q, A = A, b = b, G = G, h = h, solver = 'cvxopt')


        S = (self.alpha > 1e-3).flatten()
        self.w = np.sum(self.alpha[S].reshape(-1,1) * y_train[S].reshape(-1,1) * x_train[S], axis = 0)
        print("=======")
        print("w:")
        print(self.w)


    
        print("=======")
        print("support vector:")
        print(x_train[S])
        self.b = np.mean(y_train[S] - np.dot(x_train[S], self.w.T))
        print("=======")
        print("b:")
        print(self.b)

    def predict(self, x_test):
        return(np.sign(x_test @ self.w.T + self.b))


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