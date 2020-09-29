#Min-Hsueh Chiu
#USCID: 9656645860


import numpy as np
import pandas as pd 
import sys
class Tree(object):

    def __init__(self):

        self.children = None
        self.name = None 
        self.x_data = None
        self.y_data = None
        self.layer_number = None
        self.predict_y = None
        self.predict_y_confidence = None
        self.is_end = False

class dt():
    def __init__(self, height = 3):
        self.tree = None
        self.height = height
        self.tree_height = None 

    def split_tree(self, pre_tree):
        x = pre_tree.x_data
        y = pre_tree.y_data

        if len(y.unique()) == 1:
            output = pre_tree
            output.is_end = True
            output.predict_y = y.unique()[0]
            output.predict_y_confidence = 1.0
            return(output)
        
        tem_layer_number = pre_tree.layer_number
        if not pre_tree.layer_number == self.height:
            tem_node = {}
            lowest_information = 2.0
            for i in x.columns:
                entropy_list = []
                probability_list = []
                tem_information = 0.0
                for j in x[i].unique():
                    entropy = 0.0
                    sub_class = y[x[i] == j]
                    for k in sub_class.unique():
                        p = len(sub_class[sub_class == k])/len(sub_class)
                        entropy += p*np.log(1/p)
                    entropy_list.append(entropy) 
                    probability_list.append(len(y[x[i] == j])/len(y))
                for e_n in range(len(entropy_list)):
                    tem_information += entropy_list[e_n]*probability_list[e_n]
                if(float(tem_information) < float(lowest_information)):
                    lowest_information = float(tem_information)
                    tem_key = i
                    

            output = pre_tree
            output.children = {i: None for i in x[tem_key].unique()}
            output.name = tem_key

            for index, k_n in enumerate(x[tem_key].unique()):
                if len(y[x[tem_key] == k_n]) != 0:
                    sub_tree = Tree()
                    sub_tree.x_data = x[x[tem_key] == k_n]
                    sub_tree.y_data = y[x[tem_key] == k_n]
                    sub_tree.layer_number = tem_layer_number + 1

                    if lowest_information == 0:
                        sub_tree.is_end = True
                        sub_tree.predict_y = y[x[tem_key] == k_n].unique()[0]
                        sub_tree.predict_y_confidence = 1.0
                output.children[k_n] = sub_tree

            return(output)

        else:
            output = pre_tree
            output.name = 'End node'
            output.predict_y = y.mode()[0]
            output.predict_y_confidence = len(y[y == y.mode()[0]])/len(y)
            output.is_end = True
            return(output)

    def fit(self, x, y):
        #check input is pandas.DataFrame
        if not isinstance(x, pd.DataFrame):
            raise Exception("input x's type should be pandas.DataFrame")
        root = Tree()
        root.x_data = x
        root.y_data = y
        root.layer_number = 1

        tree_list = [root]
        while(len(tree_list) != 0):
            
            tem = tree_list.pop(0)
            if tem.layer_number > self.height:
                tem = self.split_tree(tem)
                self.tree_height = tem.layer_number
                self.tree = root
                return()
            else:
                tem = self.split_tree(tem)
                

                if tem.children:
                    for i in tem.children:
                        tree_list.append(tem.children[i])


        self.tree_height = tem.layer_number
        self.tree = root

    def predict(self, x_test):
        tem = self.tree
        while(True):
            tem = tem.children[x_test[tem.name][0]]
                
            if tem.is_end:
                return(tem.predict_y)

def main1():
    #the easier example for implement decision tree
    df = pd.DataFrame([['yes', 'yes', 'yes' ], ['yes', 'no', 'yes'], ['no', 'yes', 'no'], ['no', 'no', 'no']], columns = ['A', 'B', 'C'])
    y = df['C']
    x = df.drop(['C'], axis = 1)
    d_tree = dt(height = 5)
    d_tree.fit(x,y)

def main():
    #data collecting
    df = pd.read_csv('dt_data.txt', sep=",")
    df.columns = ['Occupied', 'Price', 'Music', 'Location', 'VIP', 'Favorite Beer', 'Enjoy']
    #remove space and semicolon
    df['Occupied'] = df['Occupied'].str.rsplit(': ', expand=True)[1]
    df['Price'] = df['Price'].str.rsplit(' ', expand=True)[1]
    df['Music'] = df['Music'].str.rsplit(' ', expand=True)[1]
    df['Location'] = df['Location'].str.rsplit(' ', expand=True)[1]
    df['VIP'] = df['VIP'].str.rsplit(' ', expand=True)[1]
    df['Favorite Beer'] = df['Favorite Beer'].str.rsplit(' ', expand=True)[1]
    df['Enjoy'] = df['Enjoy'].str.rsplit(';', expand=True)[0].str.rsplit(' ', expand=True)[1]

    #
    y = df['Enjoy']
    x = df.drop(['Enjoy'], axis = 1)
    #train
    d_tree = dt(height = 5)
    d_tree.fit(x,y)

    #prediction
    ##make x_test
    x_test = pd.DataFrame(['Moderate', 'Cheap', 'Loud', 'City-Center', 'No', 'No']).T
    x_test.columns = ['Occupied', 'Price', 'Music', 'Location', 'VIP', 'Favorite Beer']
    #
    print('-----')
    print('Query:')
    print(x_test)
    print('-----')
    print('Result:')
    print(d_tree.predict(x_test))


if __name__ == '__main__':
    main()