import numpy as np
import math as m
import dataGenerator as d
import SGD as s
import matplotlib as plt


## in the experiment we are going to run the SGD on two sigma which are 0.05 and 0.3 for 4 training set respectively for each senario, thus we need 8 differnet training 
## set and two test set for both sigma for each scenario

## generate test sets 
N = 400
sigma1= 0.05
sigma2 = 0.3
n = [50,100,500,1000]

##---------------------------------------Scenario 1 ----------------------------------------##
test_set1_sigma1 = d.generateExampleSet(N, sigma1, 1)
test_set1_sigma2 = d.generateExampleSet(N, sigma2, 1)

train_set = []

