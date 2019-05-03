#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# imports needed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
# setting seed, DON'T modify
np.random.seed(10)
from pylab import rcParams
rcParams['figure.figsize'] = 10, 7

Data = load_boston()
data = Data.data
column_1 = data[:,0]
median= np.ma.median(column_1)
y = []
for i in range(0, np.size(column_1)):
    if(median > column_1[i]):
        y.append(0)
    else:
        y.append(1)
        
y = np.reshape(y, (506,1))
X = data[:, 1:-1]

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

Xtrain = X[0:401,:]
ytrain = y[0:401]
Xtest = X[401:507, :]
ytest = y[401:507]

LogRegModel = LogisticRegression().fit(Xtrain, ytrain)
LogisticPrediction = LogRegModel.predict(Xtest)
LogisticPrediction = np.reshape(LogisticPrediction, np.shape(ytest))
logistic_error = np.linalg.norm(LogisticPrediction - ytest)

LinDiscModel = LinearDiscriminantAnalysis().fit(Xtrain, ytrain)
LinDiscPrediction = LinDiscModel.predict(Xtest)
LinDiscPrediction = np.reshape(LinDiscPrediction, np.shape(ytest))
lindisc_error = np.linalg.norm(LinDiscPrediction - ytest)

plt.figure(0, figsize=(12, 8))
plt.title('Errors')
l = np.arange(2)
errors = [logistic_error, lindisc_error]
fig, ax = plt.subplots()
plt.bar(l, errors)
plt.xticks(l, ('LogisticRegresion', 'LinearDiscriminantAnalysis'))
plt.show()

sampleinterval = [100,200,300,400]

def train_and_fit_sample(X_train, Randomsample):
    LogRegModel = LogisticRegression().fit(X_train, ytrain[Randomsample])
    LogisticPrediction = LogRegModel.predict(Xtest)
    LogisticPrediction = np.reshape(LogisticPrediction, np.shape(ytest))
    logistic_error = np.linalg.norm(LogisticPrediction - ytest)

    LinDiscModel = LinearDiscriminantAnalysis().fit(X_train, ytrain[Randomsample])
    LinDiscPrediction = LinDiscModel.predict(Xtest)
    LinDiscPrediction = np.reshape(LinDiscPrediction, np.shape(ytest))
    lindisc_error = np.linalg.norm(LinDiscPrediction - ytest)
    
    return [logistic_error, lindisc_error]

logistic_mean = []
lindisc_mean = []
logistic_std = []
lindisc_std = []

for n in sampleinterval:
    logistic_error = []
    lindisc_error = []
    for i in range(0, 10):
        x = int(n *0.5)
        Randomsample = np.random.choice(n+1,x)
        Randomsample = np.sort(Randomsample)
        Xtrain_sample = Xtrain[Randomsample,:]
        errors = train_and_fit_sample(Xtrain_sample, Randomsample)
        logistic_error.append(errors[0])
        lindisc_error.append(errors[1])
        
    log_mean = np.mean(logistic_error)
    log_std = np.std(logistic_error)
    lin_mean = np.mean(lindisc_error)
    lin_std = np.std(lindisc_error)
    logistic_mean.append(log_mean)
    lindisc_mean.append(lin_mean)
    logistic_std.append(log_std)
    lindisc_std.append(lin_std)

N = 4
ind = np.arange(N)  
width = 0.27       
fig = plt.figure()
ax = fig.add_subplot(111)

rects1 = ax.bar(ind+width/2, logistic_mean, width, color='r')
rects3 = ax.bar(ind+3*width/2, lindisc_mean, width, color='b')

ax.set_ylabel('Errormean')
ax.set_xlabel('Samplesize')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('100', '200', '300', '400') )
ax.legend( (rects1[0], rects3[0]), ('Logistic Regression', 'Linear Discriminant Analysis'))


plt.show()

