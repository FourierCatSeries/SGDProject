import numpy as np
import math
import dataGenerator as d
class logSGD:

    def __init__(self, train_set, test_set, learn_rate, scenario):
        self.train_set = train_set
        self.test_set = test_set
        self.learn_rate = learn_rate
        self.scenario = scenario
        self.w = [[0,0,0,0,0]]
        self.gradient = []
    
    def computeLearnRate(self, rho, M):
        learn_rate = []
        for i in range(len(self.train_set)):
            learn_rate.append(M/(rho*math.sqrt(len(self.train_set))))
        self.learn_rate = learn_rate
    
    def learn(self):
        iteration = 0
        for example in self.train_set:
            x_extend = d.extendX(example[0])
            y = example[1]
            inner_product_w_x_ex = np.inner(self.w[iteration], x_extend)
            scalar = (-1.0 * y) * np.exp(-1.0 * y * inner_product_w_x_ex) / (1 + np.exp(-1.0 * y * inner_product_w_x_ex))
            gradient = np.multiply(scalar, x_extend)
            self.gradient.append(gradient)
            raw_w_new = np.subtract(self.w[iteration], np.multiply(self.learn_rate[iteration], gradient))
            if(self.scenario == 1):
                w_new = d.EuclideanProjectionHyperCube(raw_w_new, 5)
            elif (self.scenario == 2):
                w_new = d.EuclideanProjectionHyperBall(raw_w_new, 5)
            self.w.append(w_new)
            iteration += 1
        

    def output(self):
        w_head = np.array([0,0,0,0,0])
        for w in self.w:
            w_head += np.array(w)
        w_head = np.divide(w_head, len(self.w))
        return w_head
    
    def getW(self):
        return self.w
