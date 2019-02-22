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
            self.thetaMats.append(funcs.getRandomMatrix(layersDims[i+1], layersDims[i]+1))

    # output must be of the same number as layersDims[last]
    # features is assumed to be 2d array of features and its
    #           second dimention equal layersDim[0] passed in init
    def train(self, features, output , alpha , lamda, iterations):



        if len(features[0]) != self.layersDims[0] :
            return
        for it in range(0,iterations):

            for row in range(0,len(features)):
                D = []
                for i in range(0, len(self.layersDims) - 1):
                    D.append(funcs.getZeroMatrix(self.layersDims[i + 1], self.layersDims[i] + 1))
                self.a=[]
                delta = []
                for i in range(0, len(self.layersDims)):
                    delta.append([])
                if it  ==0 :
                    features[row].insert(0,1.0)
                self.a.append(features[row])
                for i in range(1, len(self.layersDims)): # in each layer
                    self.a.append([])
                    temp = numpy.dot( self.thetaMats[i-1],(self.a[i - 1]))
                    self.a[i] = (funcs.sigmoid(temp))
                    self.a[i].insert(0,1.0)
                # At this point i have done the forward part now i will do the backpropagation
                for j in range(1,len(self.a[len(self.a) - 1])):
                    delta[len(delta)-1].append(self.a[len(self.a) - 1][j] - output[row] )
                    print(self.a[len(self.a) - 1][j] )
                    #                    delta[len(delta)-1].append((1-self.a[len(self.a) - 1][j] )* (j-1) * output[row]+
                                              # (self.a[len(self.a) - 1][j]) * ( j-2)* output[row])
                for k in range(1, len(self.layersDims)):  # in each layer
                    i = len(self.layersDims) - k - 1 # from n-2 to 0
                    #print(numpy.delete(numpy.multiply(self.a[i],numpy.subtract([1]*len(self.a[i]),self.a[i])), -1, axis=0))
                    delta[i] = numpy.multiply(
                        numpy.dot(numpy.transpose( numpy.delete(self.thetaMats[i], -1, axis=1)), delta[i + 1])
                        , numpy.delete(numpy.multiply(self.a[i],numpy.subtract([1]*len(self.a[i]),self.a[i])), -1, axis=0)
                    )
                    toAdd =numpy.dot(numpy.transpose((numpy.asmatrix(delta[i+1]))), (numpy.asmatrix(self.a[i])))
                    D[i] += toAdd
                # to be continued .. get the derevative for each unit using D then preform gradient decent then update theta

                for l in range(0,len(self.layersDims)-1):
                    for i in range(0,len(D[l])) :
                        for j in range(0,len(D[l][i])):
                            if j == 0:
                                D[l][i][j] = ((1.0 /( len(features))) * D[l][i][j])
                            else:
                                D[l][i][j] =( (1.0/(len(features))) * D[l][i][j] + lamda * self.thetaMats[l][i][j])

                self.thetaMats = funcs.gradient_decent(self.thetaMats, D, alpha)


    def solve(self, input):
        #print(self.thetaMats)
        self.a = []
        input.insert(0,1.0)
        self.a.append(input)
        for i in range(1, len(self.layersDims)):  # in each layer after features layer
            self.a.append([])
            temp = numpy.dot(self.thetaMats[i - 1],(self.a[i - 1]))
            self.a[i]=(funcs.sigmoid(temp))
            self.a[i].insert(0,1.0)


            if i == len(self.layersDims)-1:
                print(self.a[i])
                print('-------res------')
        return self.a[len(self.layersDims)-1]


    def test(self,input, output):
        aHat  = self.solve(input)
        for i in range(0, len(output)):
            print(aHat[i])
            print(output[i])
            print('------')
