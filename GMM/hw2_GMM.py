#Name: Min-Hsueh Chiu
#USCID: 9656645860

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
class gmm:
    def __init__(self, K = 3, tol = 0.00001, max_iter = 300):
        self.K = K #cluster number
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, x):
        self.x = x
        ##initialize r
        r = np.random.rand(len(x), self.K)
        #normalize
        for i in range(len(r)):
            r[i,:] /= r[i,:].sum()
        ##
        epison = 999
        counter = 0
        while(epison > self.tol):
            if counter == self.max_iter:
                print('reach max_iter')
                self.r = r
                self.u = u
                self.covariance = co_list
                self.pi_c = pi_c    
                return()
            ## M step
            #cal mean
            u = (np.dot(r.T,x).T/r.sum(axis = 0)).T
            #cal covariance matrix (co_list) and amplitude (pi_c)
            co_list = []
            pi_c = []
            for c in range(self.K):
                co = np.zeros(shape = (len(x[0]), len(x[0])))
                for i in range(len(x)):
                    #co += r[i, c]*np.dot((x[i,:] - u[c]).T, (x[i,:] - u[c]))
                    co += r[i, c]*np.dot((x[i,:] - u[c]).reshape(len(x[i,:]),-1), (x[i,:] - u[c]).reshape(len(x[i,:]),-1).T)
                co /= r[:,c].sum()
                co_list.append(co)
                pi_c.append(r[:,c].sum())
            co_list = np.array(co_list)
            pi_c = np.array(pi_c)
            ## E step
            r_new = np.zeros(shape = r.shape)
            for i in range(len(x)):
                for c in range(self.K):
                    r_new[i,c] = pi_c[c]*stats.multivariate_normal.pdf(x[i,:], u[c,:], co_list[c,:], allow_singular=True)

                r_new[i,:] /= r_new[i,:].sum()
            #epison = np.linalg.norm(r - r_new)
            epison = abs(r - r_new).max()
            r = r_new
        self.y_pre = np.argmax(r, axis = 1)
        self.r = r
        self.u = u
        self.covariance = co_list
        self.pi_c = pi_c

    def plot(self, filename):
        m1 = stats.multivariate_normal(self.u[0], self.covariance[0])
        m2 = stats.multivariate_normal(self.u[1], self.covariance[1])
        m3 = stats.multivariate_normal(self.u[2], self.covariance[2])
        x = np.arange(self.x[:,0].min()-1, self.x[:,0].max()+2, 0.5)
        y = np.arange(self.x[:,1].min()-1, self.x[:,1].max()+2, 0.5)
        X, Y = np.meshgrid(x, y)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X; pos[:, :, 1] = Y


        for i in range(self.K):
            plt.scatter(self.x[self.y_pre == i,0], self.x[self.y_pre == i, 1], label = i)
        #plt.scatter(self.x[:,0], self.x[:,1], c = self.y_pre, label = self.y_pre)
        plt.contour(X, Y, m1.pdf(pos))
        plt.contour(X, Y, m2.pdf(pos))
        plt.contour(X, Y, m3.pdf(pos))
        plt.legend()
        plt.grid()
        plt.savefig(filename)
        plt.clf()



def main():
    x = pd.read_csv('clusters.txt', header = None).to_numpy()

    for i in range(3):
        print(i)
        clf = gmm()
        clf.fit(x)
        print('u:')
        print(clf.u)
        print('--------')
        print('covariance:')
        print(clf.covariance)
        clf.plot('GMM_'+str(i+1)+'.png')
    
    

if __name__ == '__main__':
    main()




