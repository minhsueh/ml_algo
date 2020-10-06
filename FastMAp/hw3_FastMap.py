#Name: Min-Hsueh Chiu
#USCID: 9656645860

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

class FastMap():
    def __init__(self, N, k, dis_raw, dis_func_opt = 1, *args):
        ##
        #N: number of data 
        #k: desired dimension 
        #In this project, predefined distance is given, so dis_func_opt should be 1.
        ##

        self.N = N
        self.k = k
        self.k_record = 0
        self.z = np.zeros([N, k]) #output
        self.PA = np.zeros([2, k])
        self.col = 0
        self.dis_raw = dis_raw  
        ##
        self.dis_func_opt = dis_func_opt 
        #If dis_func_opt == 1, the predefined distance should be given in dis_raw.
        #If dis_func_opt == 2, the defined distance function should be given in dis_raw.
        self.args = args
        ##


    def dis_cal(self, Oi_index, Oj_index):
        #in this distance project, Oi_index < Oj_index:
        if Oi_index > Oj_index:
            tem = Oi_index
            Oi_index = Oj_index
            Oj_index = tem
        #
        if self.dis_func_opt == 1:
            #Not going to calculate the new distance of all combination of objects, because it is the complexity O(N^2)
            if self.k_record == 0:
                return(float(self.dis_raw[(self.dis_raw[:,0] == Oi_index) & (self.dis_raw[:,1] == Oj_index)][0,2]))
            else:
                d_cal = self.dis_raw[(self.dis_raw[:,0] == Oi_index) & (self.dis_raw[:,1] == Oj_index)][0,2]
                for i in range(self.k_record):
                    xi = self.z[Oi_index-1, i]
                    xj = self.z[Oj_index-1, i]
                    d_cal = d_cal*d_cal - (xi-xj)*(xi-xj)
                return(float(np.sqrt(d_cal)))
        if self.dis_func_opt == 2:
            #if users define customize distrance function
            if self.k_record == 0:
                return(float(self.dis_raw(*self.args)))
            else:
                d_cal = self.dis_raw(*self.args)
                for i in range(self.k_record):
                    xi = self.z[Oi_index-1, i]
                    xj = self.z[Oj_index-1, i]
                    d_cal = d_cal*d_cal - (xi-xj)*(xi-xj)
                return(float(np.sqrt(d_cal)))

    def project_dis_cal(self, Oa, Ob, Oi):
        d_ab = self.dis_cal(Oa, Ob)
        d_ai = self.dis_cal(Oa, Oi)
        d_bi = self.dis_cal(Ob, Oi)
        return((d_ai*d_ai + d_ab*d_ab - d_bi*d_bi)/2/d_ab)




    def Farest_pair(self):
        Object_len = self.N
        #Object_len: The length of objects. The object is represented as its index
        #
        #The distance will be calculate by using dis_cal function
        Oa_index = random.randint(1, Object_len)
        pre = None
        pre_dist = None
        while(True):
            dis_farest = -1
            for i in range(1, Object_len+1):
                if i != Oa_index:
                    dis_tem = self.dis_cal(Oa_index, i)
                    if dis_tem > dis_farest:
                        Ob = i
                        dis_farest = dis_tem

            if dis_farest == pre_dist and Ob == pre:
                return(Oa_index, Ob)
            pre = Oa_index
            pre_dist = dis_farest
            Oa_index = Ob



    def fastmap(self, k_cal):
        ##
        #k_cal: calculating dimension
        ##

        Object_len = self.N
        #Object_len: The length of objects. The object is represented as its index

        if(k_cal <= 0):
            return()

        Oa, Ob = self.Farest_pair()


        self.PA[0, self.col] = Oa
        self.PA[1, self.col] = Ob


        if self.dis_cal(Oa, Ob) == 0:
            for i in range(Object_len):
                self.z[i, self.col] = 0
            return()


        for i in range(1, Object_len+1):
            if i == Oa:
                self.z[i-1, self.col] = 0
            elif i == Ob:
                self.z[i-1, self.col] = self.dis_cal(Oa, Ob)
            else:
                self.z[i-1, self.col] = self.project_dis_cal(Oa, Ob, i)

        self.col += 1   
        self.k_record += 1     
        self.fastmap(k_cal - 1)

    def plot_2d(self, obj_name):
        #only use for k = 2 case

        fig, ax = plt.subplots()

        plt.scatter(self.z[:,0], self.z[:,1])

        for i in range(self.N):
            ax.annotate(str(obj_name[i][0]), (self.z[i,0], self.z[i,1]))
        ax.grid()
        plt.savefig('hw3_FastMap.png')



def main():
    dis_raw = pd.read_csv('fastmap-data.txt', sep = '\t', header = None).to_numpy()
    obj_name = pd.read_csv('fastmap-wordlist.txt', header = None).to_numpy()
    fm = FastMap(len(obj_name), 2, dis_raw)
    fm.fastmap(2)
    print("output:")
    print(fm.z)
    fm.plot_2d(obj_name)




if __name__ == '__main__':
    main()

