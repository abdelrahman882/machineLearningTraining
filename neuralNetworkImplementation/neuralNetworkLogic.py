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
    def train(self, features, output):
        delta = []
        D =[]
        for i in range(0,len(self.layersDims)):
            delta.append([])
            D.append([])
            for j in range(0, self.layersDims[i]):
                D[i].append(0)
        # features is assumed to be 2d array of features and its
        #           second dimention equal layersDim[0] passed in init
        if len(features) != self.layersDims[0] :
            return
        for row in range(0,len(features)):
            self.a.clear()
            self.a.append(features[row])
            for i in range(1, len(self.layersDims)): # in each layer
                self.a.append([])
                for j in range(0, self.layersDims[i]):
                    self.a[j] = funcs.sigmoid(sum(numpy.dot(self.a[i-1], self.thetaMats[i-1])))
            # At this point i have done the forward part now i will do the backpropagation
            for j in range(0, self.layersDims[len(self.layersDims)-1]):
                delta[len(delta)-1].append(self.a[len(self.a) - 1][j] - output[j])
            for k in range(1, len(self.layersDims)):  # in each layer
                i = len(self.layersDims) - k - 1
                delta[i] = elementWP(numpy.dot(numpy.transpose(self.thetaMats[i]),delta[i+1]),elementWP(self.a[i],1-self.a[i]))
                D[i] += delta[i+1] * numpy.transpose(self.a[i])
            # to be continued .. get the derevative for each unit using D then preform gradient decent thenupdate theta

