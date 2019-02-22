import  neuralNetworkLogic as nn

neural = nn.Neural_Network([2,8,20,4,4])
neural.train([
    [1,1],
    [1, 1],

    [1,0],
    [0,1],
    [0,0]],
    [1,1,0,0,0],
    0.001,100,1000)
neural.solve([0,0])
