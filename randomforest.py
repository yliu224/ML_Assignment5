from classifier import classifier
from decision_tree import decision_tree
import random
import numpy as np
from collections import Counter

class randomforest(classifier):
    def __init__(self, tree_size=10, max_depth=-1):
        self.tree_size = tree_size
        self.max_depth = max_depth
        self.trees = []
        self.features = []

    # random select sample x and y
    def random_data(self, X, Y):
        selectedX = []
        selectedY = []
        #random data
        size = len(X)
        for i in range(size):
            index = random.randint(0,size-1)
            selectedX.append(X[index])
            selectedY.append(Y[index])
        #random feature
        feature_size = int(np.sqrt(len(X[0])))
        featuredX = np.array([[]]*len(selectedX))
        selectedX = np.array(selectedX)
        feature = random.sample(range(len(X[0])),feature_size)
        for i in feature:
            featuredX=np.concatenate((featuredX,selectedX[:,i:i+1]), axis = 1)
        return feature,featuredX.tolist(),selectedY


    def fit(self, X, Y):
        for i in range(self.tree_size):
            feature,selectedX, selectedY = self.random_data(X, Y)
            tree = decision_tree()
            print("Fitting "+str(i))
            tree.fit(selectedX, selectedY)
            self.trees.append(tree)
            self.features.append(feature)

    def predict(self, X):
        hypothesis_list=[]
        for i in range(self.tree_size):
            hypothesis_list.append(self.trees[i].predict(self.get_data_set(i,X)))
        ret = []
        for i in range(len(hypothesis_list[0])):
            temp = []
            for j in hypothesis_list:
                if j[i] != None:
                    temp.append(j[i])
            ret.append(temp)
        predicts=[]
        for row in ret:
            if not row:
                predicts.append("None")
            predicts.append(Counter(row).most_common()[0][0])
        return predicts
        return ret

    def get_data_set(self,index,X):
        objX = np.array([[]]*len(X))
        npX = np.array(X)
        for i in self.features[index]:
            objX=np.concatenate((objX,npX[:,i:i+1]), axis = 1)
        return objX.tolist()
