import numpy as np
import math
import matplotlib as plt

def EuclideanProjectionHyperCube(u, dimention = 4):
    x = []
    for i in range(dimention):
        if u[i] > 1:
            x.append(1)
        elif u[i]<-1:
            x.append(-1)
        else:
            x.append(u[i])
    return x
def EuclideanProjectionHyperBall(u, dimention = 4):
    x = []
    norm = 0
    for i in range(dimention):
        norm += u[i]*u[i]
    norm = math.sqrt(norm)
    if norm > 1:
        for i in range(dimention):
           x.append(u[i]/norm * 1.0)
    else:
        x = u

    return x
def extendX(x):
    x = np.append(x, 1)
    return x


def generateExampleSet(num, sigma, scenario, dimention = 4):
    ## training set to return
    set = []
    ## generate the label of the examples by the uniform distribution
    label = np.random.randint(2, size = num)
    ## change the 0 in the array to be -1 to match the scenario
    for j in range(len(label)):
        if label[j] == 0:
            label[j] = -1
    
    for i in range(num):
        mu = 0.25
        if label[i] == -1:
            mu = -0.25
        u = np.random.normal(mu, sigma, 4)
        ## each training example in the set is a list in the form [x, y] where x is the Euclidean Projection based on the scenario specified
        if scenario == 1: 
            set.append([EuclideanProjectionHyperCube(u), label[i]])
        elif scenario == 2:
            set.append([EuclideanProjectionHyperBall(u), label[i]])
    
    return set


##set = generateExampleSet(30,0.3,2, dimention = 1)
##print(set)
