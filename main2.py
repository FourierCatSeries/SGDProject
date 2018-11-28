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
rho_2 = m.sqrt(2)
M_2 = 1

##---------------------------------------Scenario 2 ----------------------------------------##
## generate test set size of 400
test_set2_sigma1 = d.generateExampleSet(N, sigma1, 2)
test_set2_sigma2 = d.generateExampleSet(N, sigma2, 2)

train_sets_sigma1 = []
train_sets_sigma2 = []
## generate the training set that will be used in the training of the SGD by 30 times for each setting of (sigma, n)
for num in n:
    train_sets_1 = []
    train_sets_2 = []
    for i in range(30):
        train_sets_1.append(d.generateExampleSet(num, sigma1, 2))
        train_sets_2.append(d.generateExampleSet(num, sigma2, 2))

    train_sets_sigma1.append(train_sets_1)
    train_sets_sigma2.append(train_sets_2)
## declare the variable to store the result

## result for logistic loss
mean_log = [[], []]
std_dev_log= [[], []]
min_log = [[], []]
excess_risk_log = [[], []]
## result for Classification error
mean_class = [[], []]
std_dev_class = [[], []]

## train SGD and gather the result for sigma1 = 0.05
for sets in train_sets_sigma1:
    w_head_current = []
    log_loss = []
    log_loss_mean = 0
    log_loss_std  = 0
    log_loss_min = 300
    log_excess = 0
    class_error = []
    class_error_mean = 0
    class_std = 0
    for train_set in sets:
        sgd = s.logSGD(train_set, test_set2_sigma1,2)
        sgd.computeLearnRate(rho_2, M_2)
        sgd.learn()
        w_head_current.append(sgd.output)
        ##calculate error and loss
        loss = sgd.log_risk_average()
        log_loss.append(loss)
        error = sgd.class_error_average()
        class_error.append(error)
        if(loss<log_loss_min):
        	log_loss_min = loss
    ##compute the estimate
    log_loss_mean = np.mean(log_loss)
    class_error_mean = np.mean(class_error)
    log_loss_std = np.std(log_loss)
    class_std = np.std(class_error)
    log_excess = log_loss_mean - log_loss_min
    ## store the estimate
    min_log[0].append(log_loss_mean)
    std_dev_log[0].append(log_loss_std)
    min_log[0].append(log_loss_min)
    excess_risk_log[0].append(log_excess)
    mean_class[0].append(class_error_mean)
    std_dev_class[0].append(class_std)

## train SGD and gather the result for sigma2 = 0.3

for sets in train_sets_sigma2:
    w_head_current = []
    log_loss = []
    log_loss_mean = 0
    log_loss_std  = 0
    log_loss_min = 300
    log_excess = 0
    class_error = []
    class_error_mean = 0
    class_std = 0
    for train_set in sets:
        sgd = s.logSGD(train_set, test_set2_sigma2,2)
        sgd.computeLearnRate(rho_2, M_2)
        sgd.learn()
        w_head_current.append(sgd.output)
        ##calculate error and loss
        loss = sgd.log_risk_average()
        log_loss.append(loss)
        error = sgd.class_error_average()
        class_error.append(error)
        if(loss<log_loss_min):
        	log_loss_min = loss
    ##compute the estimate
    log_loss_mean = np.mean(log_loss)
    class_error_mean = np.mean(class_error)
    log_loss_std = np.std(log_loss)
    class_std = np.std(class_error)
    log_excess = log_loss_mean - log_loss_min
    ## store the estimate
    min_log[1].append(log_loss_mean)
    std_dev_log[1].append(log_loss_std)
    min_log[1].append(log_loss_min)
    excess_risk_log[1].append(log_excess)
    mean_class[1].append(class_error_mean)
    std_dev_class[1].append(class_std)





    
    

	





