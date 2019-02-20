import  neededFuncs as funcs
import  numpy
import  math

class Neural_Network :
    def __init__(self, layersDims):
        if len(layersDims) < 2:
            return
        self.layersDims = layersDims
        # theta mats is an array where any element in index i is the matrix of theta in layer i
        self.thetaMats = []
        self.a = []
        for i in range(0, len(layersDims)-1):
            self.thetaMats.append(funcs.getRandomMatrix(layersDims[i+1], layersDims[i]))

    # output must be of the same number as layersDims[last]
    def move_forward(self, features, output):
        self.a.clear()
        # features is assumed to be list of features and equal layersDim[0] passed in init
        if len(features) != self.layersDims[0] :
            return
        self.a.append(features)
        for i in range(1, len(self.layersDims)): # in each layer
            self.a.append([])
            for j in range(0,self.layersDims[i]):
                self.a[j] = sum(numpy.dot(self.a[i-1], self.thetaMats[i-1]))



