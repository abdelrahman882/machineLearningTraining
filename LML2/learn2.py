import matplotlib.pyplot as plt
import pandas as pd
import numpy
import  math

class regression :
    def __init__ (self,pramsNum, file , IterationsNum , alpha,lfn,modFile,ax):
        self.ax =ax
        self.modFile = modFile
        self.linearFeaturesNum = lfn
        self.alpha = alpha
        self.file = file
        self.pramsNum = pramsNum
        self.iterationsNum = IterationsNum
    def operate (self):
        linearFeaturesNum = self.linearFeaturesNum
        file = self.file
        if self.modFile:
            # -1 the output
            for i in range(len(file.columns.values)-1):
                file[file.columns.values[i]] = file[file.columns.values[i]].astype(float)
                max=file[file.columns.values[i]].max()
                for j in range(len(file[file.columns.values[i]])):
                    file[file.columns.values[i]][j]=float(file[file.columns.values[i]][j]*10/ float(max))
        alpha = self.alpha
        theta = []
        for i in range(0,self.pramsNum) :
            theta.append(0)
        costA = []
        for i in range(0, self.iterationsNum):
            cost = 0
            thetaT = []
            for k in range(0, self.pramsNum):
                thetaT.append(0)
            for inTheta in range(len((theta))):
                sum = 0
                cost = 0
                r=-1
                for data in file.iterrows():
                    currentFeature = 0

                    r+=1
                    if inTheta >= linearFeaturesNum:
                        currentFeature =10 * (data[1][inTheta % linearFeaturesNum])**(inTheta / linearFeaturesNum+2)/\
                                        (file[file.columns.values[inTheta% linearFeaturesNum]].max()**(inTheta / linearFeaturesNum+2))
                    else:
                        currentFeature = data[1][inTheta]

                    hyp = theta[0]
                    for num in range(1,len(theta)):
                        if num >= linearFeaturesNum:
                            hyp += theta[num] * int(file[file.columns.values[num%linearFeaturesNum]][r])**(num / linearFeaturesNum+2)/\
                                   (file[file.columns.values[num% linearFeaturesNum]].max()**(num / linearFeaturesNum+2))

                        else:
                            hyp += theta[num] * int(file[file.columns.values[num]][r])
                    sum += (hyp - file[file.columns.values[linearFeaturesNum]][r]) * int(currentFeature)
                    cost += numpy.abs(hyp - file[file.columns.values[linearFeaturesNum]][r])
                f = (1.0 / (file[file.columns.values[0]].count()))
                thetaT[inTheta] -= alpha * f * float(sum)
            theta = thetaT
            costA.append(cost)

        self.ax.scatter(range(0, self.iterationsNum), numpy.asarray(costA), s=10, c=col[self.pramsNum-4 ], marker="p",
                        label = 'prams ='+ str(self.pramsNum ))


fig= plt.figure()
ax1 = fig.add_subplot(111)
col = ['r','b','g','y','k']

for pramNN in range(0,4):
    myfile = pd.read_csv('Mall_Customers.csv')
    myfile2 = pd.read_csv('heart.csv')
    myfile.insert(2,'Gender',0.3)
    for i in range(0,myfile[myfile.columns.values[0]].count()):
        if myfile[myfile.columns.values[1]][i] == 'Male':
            myfile[myfile.columns.values[2]][i] = 0.2
        else :
            myfile[myfile.columns.values[2]][i] = 0.3
    myfile.drop('f1',axis=1,inplace=True)
    myfile.drop('CustomerID',axis=1,inplace=True)
    # number of parameters , file , iterations , alpha , linear fatures
    r = regression(pramNN , myfile,10,0.0001,3,True,ax1)
    r.operate()
 #   c = classification(13+pramNN , myfile2,10,0.0001,13,True,ax1)
  #  c.operate()
plt.legend(loc='upper center');
plt.show()



class classification :
    def __init__ (self,pramsNum, file , IterationsNum , alpha,lfn,modFile,ax):
        self.ax =ax
        self.modFile = modFile
        self.linearFeaturesNum = lfn
        self.alpha = alpha
        self.file = file
        self.pramsNum = pramsNum
        self.iterationsNum = IterationsNum
    def operate (self):
        linearFeaturesNum = self.linearFeaturesNum
        file = self.file
        if self.modFile:
            # -1 the output
            for i in range(len(file.columns.values)-1):
                file[file.columns.values[i]] = file[file.columns.values[i]].astype(float)
                max=file[file.columns.values[i]].max()
                for j in range(len(file[file.columns.values[i]])):
                    file[file.columns.values[i]][j]=float(file[file.columns.values[i]][j]*10/ float(max))
        alpha = self.alpha
        theta = []
        for i in range(0,self.pramsNum) :
            theta.append(0)
        costA = []
        for i in range(0, self.iterationsNum):
            cost = 0
            thetaT = []
            for k in range(0, self.pramsNum):
                thetaT.append(0)
            for inTheta in range(len((theta))):
                sum = 0
                cost = 0
                r=-1
                for data in file.iterrows():
                    currentFeature = 0

                    r+=1
                    if inTheta >= linearFeaturesNum:
                        currentFeature =10 * (data[1][inTheta % linearFeaturesNum])**(inTheta / linearFeaturesNum+2)/\
                                        (file[file.columns.values[inTheta% linearFeaturesNum]].max()**(inTheta / linearFeaturesNum+2))
                    else:
                        currentFeature = data[1][inTheta ]

                    hyp = theta[0]
                    for num in range(1,len(theta)):
                        if num >= linearFeaturesNum:
                            hyp += theta[num] * int(file[file.columns.values[num%linearFeaturesNum]][r])**(num / linearFeaturesNum+2)/\
                                   (file[file.columns.values[num% linearFeaturesNum]].max()**(num / linearFeaturesNum+2))

                        else:
                            hyp += theta[num] * int(file[file.columns.values[num]][r])
                    newHyp = 1/(1+ math.exp(-hyp)) # classification
                    sum += (newHyp - file[file.columns.values[linearFeaturesNum]][r]) * int(currentFeature)
                    y=file[file.columns.values[linearFeaturesNum]][r]
                    cost -= math.log(newHyp)* y + (1-y) * math.log(1-newHyp)
                f = (1.0 / (file[file.columns.values[0]].count()))
                thetaT[inTheta] -= alpha * f * float(sum)
            theta = thetaT
            costA.append(cost)

        self.ax.scatter(range(0, self.iterationsNum), numpy.asarray(costA), s=10, c=col[(self.pramsNum-4) %5], marker="p",
                        label = 'prams ='+ str(self.pramsNum ))

