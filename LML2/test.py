import matplotlib.pyplot as plt
import pandas as pd
import numpy
file = pd.read_csv('Mall_Customers.csv')
alpha = 0.0001
theta = [-67.1,200.2,21.2,-34.1,7.33,-9.54]
costA = []
i=0
for i in range(0,101):
    cost=0
    index=-1
    thetaT = [0.0,0.0,0.0,0.0,0.0,0.0]

    for thetaN in theta:
        index +=1
        sum = 0
        cost=0
        r=-1
        for data in file.iterrows():

            r +=1
            currentF=0
            if index >2 :
                currentF =numpy.square( data[1][index+1-2])/100.2
            else :
                currentF = data[1][index + 1]
            d1 = 0.5
            if file['f1'][r] == 'Male':
                d1 = 0.7
            if index==0 :
                currentF = d1
            hyp = theta[0] +\
                  theta[1] * d1 +\
                  theta[2] * int (file['f2'][r]) +\
                  theta[3] * int(file['f3'][r]) +\
                  theta[4] * numpy.square(int(file['f2'][r]))/100.2 +\
                  theta[5] * numpy.square(int (file['f3'][r]) )/100.2
            sum += (hyp - file['op'][3]) * int(currentF)
            cost += numpy.abs(hyp - file['op'][r])/1000.0
        f = (1.0/(file['f1'].count()))
        thetaT[index] -= alpha * f * float(sum)
    theta = thetaT
    costA.append(cost)

plt.scatter(range(0,101),numpy.asarray(costA))
plt.show()