import numpy
import matplotlib as plt
import math


def sigmoid(z):
    l =[]
    for i in range(0,len(z)):
        l.append(1/(1+numpy.exp(-z[i])))
    return l


def getRandomMatrix(x,y):
    min = -5
    max = 5
    return ((numpy.random.rand(x,y)* (max - min) ) + min)


def getZeroMatrix(x,y):
    return numpy.zeros([x,y],dtype=float)


def gradient_decent(theta, dJdT, alpha):

    return numpy.subtract(theta ,numpy.multiply( [alpha]*len(dJdT) , dJdT))
