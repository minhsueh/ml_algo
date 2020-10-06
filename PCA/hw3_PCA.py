#Name: Min-Hsueh Chiu
#USCID: 9656645860

import numpy as np
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt

class PCA():
    def __init__(self, k):
        self.k = k



    def fit(self, x):
        if not isinstance(x, np.ndarray):
            raise Exception("Type of x should be numpy") 

        '''
        #covariance method 1
        mu = x.mean(axis = 0)
        mu_bar = x - mu
        covariance = np.dot(mu_bar.T, mu_bar)/len(x)
        '''
        
        #covariance method 2
        covariance = np.cov(x.T, bias = True)

        lamda, v = LA.eig(covariance)

        lamda_sort = lamda.argsort()[::-1] #[::-1] giving the order decreasing


        self.v_truncate = v[:,lamda_sort[0:self.k]]

        


    def transform(self, x):

        return(np.dot(self.v_truncate.T, x.T).T)






def main():
    x = pd.read_csv('pca-data.txt', sep = '\t', header = None).to_numpy()
    pca = PCA(2)
    pca.fit(x)
    output = pca.transform(x)
    print(pca.v_truncate)


    #plot
    plt.scatter(output[:,0], output[:,1], s = 5)
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.grid()
    plt.savefig('hw3_PCA.png')



if __name__ == '__main__':
    main()