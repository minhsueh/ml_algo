#Name: Min-Hsueh Chiu
#USCID: 9656645860

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class kmean():
    def __init__(self, K = 3, tol = 0.01, max_iter = 300):
        self.K = K #cluster number
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, x):
        self.x = x
        '''
        #first version, the initial centroid random sample within the range of data. Having problem that centroid may not be assign points
        c = None
        for i in range(self.K):
            if c is not None:
                c = np.vstack((c, [np.random.random()*(x[:,0].max()-x[:,0].min()), np.random.random()*(x[:,1].max()-x[:,1].min())]))
            else:
                c = [np.random.random()*(x[:,0].max()-x[:,0].min()), np.random.random()*(x[:,1].max()-x[:,1].min())]
        '''
        #second version, the initial centroid random sample within the data
        c = x[np.random.randint(len(x), size=3), :]
        #
        dist = None
        for i in range(self.K):
            if dist is not None:
                dist = np.hstack((dist, np.linalg.norm(x - c[i], axis = 1).reshape(len(x), -1)))
            else:
                dist = np.linalg.norm(x - c[i], axis = 1).reshape(len(x), -1)
        y_pre = np.argmin(dist, axis = 1)

        counter = 0
        epsilon = 999
        while(epsilon > self.tol):
            #exceed maximum iteration
            if counter == self.max_iter:
                self.c = c
                return()
            #
            y_pre_hist = y_pre
            for i in np.unique(y_pre):
                c[i] = x[y_pre == i].sum(axis = 0)/len(x[y_pre == i])

            dist = None
            for i in range(self.K):
                if dist is not None:
                    dist = np.hstack((dist, np.linalg.norm(x - c[i], axis = 1).reshape(len(x), -1)))
                else:
                    dist = np.linalg.norm(x - c[i], axis = 1).reshape(len(x), -1)
            y_pre = np.argmin(dist, axis = 1)

            epsilon = np.count_nonzero(y_pre - y_pre_hist)/len(y_pre)
            counter += 1

        self.c = c
        self.y_pre = y_pre

    def plot(self, filename):
        for i in range(self.K):
            plt.scatter(self.x[self.y_pre == i,0], self.x[self.y_pre == i, 1], label = i)
        #plt.scatter(self.x[:,0], self.x[:,1])
        plt.scatter(self.c[:,0], self.c[:,1], c = 'red', marker = 'x', s = 50)
        plt.grid()
        plt.legend()
        plt.savefig(filename)
        plt.clf()


def main():
    x = pd.read_csv('clusters.txt', header = None).to_numpy()

    for i in range(3):
        print(i)
        clf = kmean()
        clf.fit(x)
        print('c:')
        print(clf.c)
        clf.plot('Kmean_'+str(i+1)+'.png')
    

if __name__ == '__main__':
    main()