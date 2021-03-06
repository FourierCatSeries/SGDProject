import numpy as np
import math as m
import dataGenerator as d
import SGD as s
import matplotlib.pyplot as plt


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
    log_loss = []
    log_loss_mean = 0
    log_loss_std  = 0
    log_loss_min = 300
    log_excess = 0
    class_error = []
    class_error_mean = 0
    class_std = 0
    for train_set in sets:
        sgd = s.logSGD(train_set, test_set1_sigma1,1)
        sgd.computeLearnRate(rho_1, M_1)
        sgd.learn()
        sgd.output()
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
    mean_log[0].append(log_loss_mean)
    std_dev_log[0].append(log_loss_std)
    min_log[0].append(log_loss_min)
    excess_risk_log[0].append(log_excess)
    mean_class[0].append(class_error_mean)
    std_dev_class[0].append(class_std)

## train SGD and gather the result for sigma2 = 0.3

for sets in train_sets_sigma2:
    log_loss = []
    log_loss_mean = 0
    log_loss_std  = 0
    log_loss_min = 300
    log_excess = 0
    class_error = []
    class_error_mean = 0
    class_std = 0
    for train_set in sets:
        sgd = s.logSGD(train_set, test_set1_sigma2,1)
        sgd.computeLearnRate(rho_1, M_1)
        sgd.learn()
        sgd.output()
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
    mean_log[1].append(log_loss_mean)
    std_dev_log[1].append(log_loss_std)
    min_log[1].append(log_loss_min)
    excess_risk_log[1].append(log_excess)
    mean_class[1].append(class_error_mean)
    std_dev_class[1].append(class_std)

## output result and plot the numbers
file = open('scenario1.txt', 'w')
file.write("sigma = 0.05\n")
for i in range(4):
    file.write("n = " + str(n[i]) + ": ")
    file.write("log: ")
    file.write("Mean: ")
    file.write(str(mean_log[0][i]) + ", ")
    file.write("Std Dev: ")
    file.write(str(std_dev_log[0][i]) + ", ")
    file.write("Min : ")
    file.write(str(min_log[0][i]) + ", ")
    file.write("Excess: ")
    file.write(str(excess_risk_log[0][i]) + ". ")
    file.write("Class: ")
    file.write("Mean: ")
    file.write(str(mean_class[0][i]) + ", ")
    file.write("Std Dev: ")
    file.write(str(std_dev_class[0][i]) + ". \n")

file.write("sigma = 0.3\n")
for i in range(4):
    file.write("n = " + str(n[i]) + ": ")
    file.write("log: ")
    file.write("Mean: ")
    file.write(str(mean_log[1][i]) + ", ")
    file.write("Std Dev: ")
    file.write(str(std_dev_log[1][i]) + ", ")
    file.write("Min : ")
    file.write(str(min_log[1][i]) + ", ")
    file.write("Excess: ")
    file.write(str(excess_risk_log[1][i]) + ". ")
    file.write("Class: ")
    file.write("Mean: ")
    file.write(str(mean_class[1][i]) + ", ")
    file.write("Std Dev: ")
    file.write(str(std_dev_class[1][i]) + ". \n")



fig = plt.figure(1)
x = n
y1 = excess_risk_log[0]
line1 = plt.errorbar(x, y1, yerr = std_dev_log[0], linestyle = '-', linewidth = 1)
y2 = excess_risk_log[1]
line2 = plt.errorbar(x, y2, yerr = std_dev_log[1], linestyle = '--', linewidth = 1)
y3 = mean_class[0]
line3 = plt.errorbar(x, y3, yerr = std_dev_class[0], linestyle = '-.', linewidth = 1)
y4 = mean_class[1]
line4 = plt.errorbar(x, y4, yerr = std_dev_class[1], linestyle = '-.', linewidth = 1)

plt.legend(('logistic excess risk: sigma = 0.05', 'logistic excess risk: sigma = 0.3', 'classification error sigma = 0.05', 'classification error sigma = 0.3'), loc = 'upper right')
plt.xlabel('n')
plt.title('Scenario 1')
plt.show()

	





