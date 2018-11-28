import numpy as np
import math
import dataGenerator as d
class logSGD:

    def __init__(self, train_set, test_set, scenario):
        self.train_set = train_set
        self.test_set = test_set
        self.learn_rate = []
        self.scenario = scenario
        self.w = [[0,0,0,0,0]]
        self.w_head = []
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
        self.w_head = w_head
    
    def getW(self):
        return self.w

    def loss_logistic(w, example):
        y = example[1]
        x_extend = example[0].append(1)
        ywx = (-1.0) * y * np.multiply(w, x_extend)
        loss = np.log(1+np.exp(ywx))
        return loss

    def log_risk_average(self):
        sum = 0
        for example in self.test_set:
            loss = self.loss_logistic(self.w_head, example)
            sum += loss
        average = sum/len(self.test_set)
        return average
    
    def class_error(w, example):
        y = example[1]
        x_extend = example[0].append(1)
        product = np.multiply(w, x_extend)
        error = 0
        if product * y < 0: 
            error = 1
        return error
    
    def class_error_average(self):
        sum = 0
        for example in self.test_set:
            sum += self.class_error(self.w_head, example)
        average = sum/len(self.test_set)
        return average

train = d.generateExampleSet(20, 0.05, 2)
test = d.generateExampleSet(20, 0.05, 2)

sgd = logSGD(train, test, 2)
sgd.computeLearnRate(math.sqrt(2), 1)
sgd.learn()
sgd.output
print(sgd.getW())