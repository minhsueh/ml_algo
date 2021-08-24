#Name: Min-Hsueh Chiu
#USCID: 9656645860

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

class Linear_Regression():
    def __init__(self):
        self.w = None

    def fit(self, x, y):
        if not isinstance(x, np.ndarray) or not isinstance(x, np.ndarray):
            raise Exception('x and y should by numpy')
        self.D = np.concatenate((np.ones(shape = (x.shape[0], 1)), x), axis = 1)

        self.w = np.dot(np.dot(np.linalg.inv(np.dot(self.D.T, self.D)), self.D.T), y)

    def predict(self,x_test):
        x_test = np.concatenate((np.ones(shape = (x_test.shape[0], 1)), x_test), axis = 1)
        return(np.dot(x_test, self.w))


def main():
    df = pd.read_csv('linear-regression.txt', header = None)
    y = df[2].to_numpy()
    x = df.drop([2], axis = 1).to_numpy()

    lr = Linear_Regression()
    lr.fit(x, y)
    print(lr.w)
    


if __name__ == '__main__':
    main()