import numpy as np
import math as m
import dataGenerator as d
import SGD as s
import matplotlib as plt


## in the experiment we are going to run the SGD on two sigma which are 0.05 and 0.3 for 4 training set respectively for each senario, thus we need 8 differnet training 
## set and two test set for both sigma for each scenario

## constant set
N = 400
sigma1= 0.05
sigma2 = 0.3
n = [50,100,500,1000]
## the setting of rho and M please refer to the section of the report addressing the analysis of the rho-lipschitzness of the 
## loss function.
rho_1 = m.sqrt(5)
M_1 = m.sqrt(5)

##---------------------------------------Scenario 1 ----------------------------------------##
## generate test set size of 400
test_set1_sigma1 = d.generateExampleSet(N, sigma1, 1)
test_set1_sigma2 = d.generateExampleSet(N, sigma2, 1)

train_sets_sigma1 = []
train_sets_sigma2 = []
## generate the training set that will be used in the training of the SGD by 30 times for each setting of (sigma, n)
for num in n:
    train_sets_1 = []
    train_sets_2 = []
    for i in range(30):
        train_sets_1.append(d.generateExampleSet(num, sigma1, 1))
        train_sets_2.append(d.generateExampleSet(num, sigma2, 1))

    train_sets_sigma1.append(train_sets_1)
    train_sets_sigma2.append(train_sets_2)
## declare the variable to store the result

##result w_head
w_head = []
## result for logistic loss
mean_log = []
std_dev_log= []
min_log = []
excess_risk_log = []
## result for Classification error
mean_class = []
std_dev_class = []

## train SGD and gather the result for sigma1 = 0.05
for sets in train_sets_sigma1:
    w_head_current = []
    for train_set in sets:
        sgd = s.logSGD(train_set, test_set1_sigma1,1)
        sgd.computeLearnRate(rho_1, M_1)
        sgd.learn()
        w_head_current.append(sgd.output)



