#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# imports needed
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import plotly.plotly as py
import plotly.tools as tls

# setting seed, DON'T modify
np.random.seed(10)
from pylab import rcParams
rcParams['figure.figsize'] = 10, 7

# no need to modify this
def poly_feature(X,poly = 1):
    # expects an array (X) of shape (n,1)
    newX = []
    for i in range(poly+1):
        newX.append(X**i)
    return np.concatenate(newX, axis=1)

data = np.load('ps01.data').item()
X_train = data['Xtrain']
X_test = data['Xtest']
Y_train = data['Ytrain']
Y_test = data['Ytest']
plt.figure(0)
plt.scatter(X_train, Y_train, c='red', label='Trainingdata')
plt.scatter(X_test, Y_test, c='blue', label='Testdata')
plt.title('Data')
plt.legend(loc=2)
plt.show()

X_train_new = poly_feature(X_train)
X_test_new = poly_feature(X_test)

beta = (np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X_train_new), X_train_new)), np.transpose(X_train_new)), Y_train))
print('beta = ', beta)
Y_train_hat = np.matmul(X_train_new, beta)
Y_test_hat = np.matmul(X_test_new, beta)
plt.figure(0, figsize=(12, 8))
plt.plot(X_train, Y_train_hat,'-ok', c= "black", label='Y_hat')
plt.scatter(X_train, Y_train, c='red', label='Trainingdata')
plt.scatter(X_test, Y_test, c='blue', label='Testdata')
plt.title('Data')
plt.legend(loc=2)
plt.show()

Y_mean_train = np.average(Y_train_hat)
Y_mean_test = np.average(Y_test_hat)

Diff = np.subtract(Y_test, Y_test_hat)
sum_nominator = 0
sum_denominator = 0
for i in range(len(Y_test_hat)):
    sum_nominator += (Diff[i])**(2)
    sum_denominator += (Y_test_hat[i] - Y_mean_test)**2
E_test_sum = sum_nominator
E_test = E_test_sum/len(Y_test_hat)
print('Test error = ', E_test)
R_square_test = 1 - sum_nominator/sum_denominator
print('R_square_test = ', R_square_test)

Diff = np.subtract(Y_train, Y_train_hat)
sum_nominator = 0
sum_denominator = 0
for i in range(len(Y_train_hat)):
    sum_nominator += (Diff[i])**(2)
    sum_denominator += (Y_train_hat[i] - Y_mean_train)**2
E_train_sum = sum_nominator
E_train = E_train_sum/len(Y_train_hat)
print('Train error = ', E_train)
R_square_train = 1 - sum_nominator/sum_denominator
print('R_square_train = ', R_square_train)

print('\nR_square_test < 0 => If we take just the average of the Test data results,') 
print('we do better than if we take the computed Yhat')

get_ipython().run_line_magic('matplotlib', 'inline')

d =  [2,3,4,5,6,7]
colors = ['green', 'purple', 'pink', 'yellow', 'brown', 'black']
plt.figure(0, figsize=(12, 8))
plt.scatter(X_train, Y_train, c='red', label='Trainingdata')
plt.scatter(X_test, Y_test, c='blue', label='Testdata')
data = np.load('ps01.data').item()
X_train = data['Xtrain']
X_test = data['Xtest']
Y_train = data['Ytrain']
Y_test = data['Ytest']
def plot_beta(d, color, label):
    X_train_new = poly_feature(X_train, d)
    X_test_new = poly_feature(X_test, d)
    beta = (np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X_train_new), X_train_new)), np.transpose(X_train_new)), Y_train))
    Y_train_hat = np.matmul(X_train_new, beta)
    Y_test_hat = np.matmul(X_test_new, beta)
    Y_train_hat_1 = Y_train_hat[:,0]
    X_train_1 = X_train_new[:,1]
    sorted_order = np.argsort(X_train_1)
    X_train_1 = np.sort(X_train_1)
    Y_train_hat_new = []
    for index in sorted_order:
        Y_train_hat_new.append(Y_train_hat_1[index])
    plt.plot(X_train_1, Y_train_hat_new,c= color, label = label, ms =3)
    return Error_func(X_train_new, X_test_new, Y_train_hat, Y_test_hat)

def Error_func(X_train, X_test, Y_train_hat, Y_test_hat):
    Y_mean_train = np.average(Y_train_hat)
    Y_mean_test = np.average(Y_test_hat)
    Diff = np.subtract(Y_test, Y_test_hat)
    sum_nominator = 0
    sum_denominator = 0
    for i in range(len(Y_test_hat)):
        sum_nominator += (Diff[i])**(2)
        sum_denominator += (Y_test_hat[i] - Y_mean_test)**2
    E_test = sum_nominator/len(Y_test_hat)
    R_square_test = 1 - sum_nominator/sum_denominator
    
    Diff = np.subtract(Y_train, Y_train_hat)
    sum_nominator = 0
    sum_denominator = 0
    for i in range(len(Y_train_hat)):
        sum_nominator += (Diff[i])**(2)
        sum_denominator += (Y_train_hat[i] - Y_mean_train)**2
    R_square_train = 1 - sum_nominator/sum_denominator
    return [R_square_test, R_square_train, E_test] 

Error_list = []
R_square_train_list = []
R_square_test_list = []

for degree, color in zip(d, colors):
    label='yhat{0}'.format(degree)
    result = plot_beta(degree, color, label)
    R_square_test = result[0]
    R_square_train = result[1]
    error = result[2]
    Error_list.append(error)
    R_square_train_list.append(R_square_train)
    R_square_test_list.append(R_square_test)


plt.title('Data')
plt.legend(loc=2)
plt.show()

plt.figure(1, figsize=(12, 8))
plt.title('Error')
plt.plot(d, Error_list, label = 'test-error')
plt.legend(loc=2)
plt.show()
plt.figure(2, figsize=(12, 8))
plt.title('R_square in comparison')
plt.plot(d, R_square_train_list, label = 'train')
plt.plot(d, R_square_test_list, label = 'test')
plt.legend(loc=2)
plt.show()

