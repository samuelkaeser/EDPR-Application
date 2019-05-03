#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# imports needed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression

# setting seed, DON'T modify
np.random.seed(10)
from pylab import rcParams
rcParams['figure.figsize'] = 10, 7

# do not change
def cache_data(f):
    x = {}
    def cached_f(rho):
        if rho not in x:
            x[rho] = f(rho)
        return x[rho]
    return cached_f

# code here
def generate_dataset(rhos = [0.1]):
    n_features = 100
    n_samples = 1000
    X = []
    rho_dict = {}
    for i in range(n_features-1):
        # 1. Add in code to append a random normal sample of size dim
        #     assume mean = 0, variance = 1
        #     make sure feat has shape (n_samples,1)
        feat = np.random.normal( loc = 0, scale = 1, size = (n_samples,1))
        X.append(feat)
        
    # 2. Convert X into a numpy array of shape (n_samples,n_features)
    X = np.asarray(X)
    X = np.reshape(X,(99, 1000))
    X = X.T
    # Generates the true beta
    true_beta = np.random.uniform(1,2,size=n_features)/np.sqrt(n_features)
    
    # constants, do not change
    a = np.ones((n_features-1,1))/np.sqrt(99)
    rho_noise = np.random.normal(size=(n_samples,1))
    
    for rho in rhos:
        # 3. Create the final feature for each sample by using a and rho
        #    i.e. final_column = a_1 x_1 + a_2 x_2 + ... + a_99 x_99 + rho * rho_noise
        final_column = np.zeros(shape = (n_samples, 1))
        for row_index in range(0, len(X[:,0])):
            feature = 0
            for c_index in range(0, len(X[row_index, :])):
                feature += a[c_index] * X[row_index, c_index]
            feature += rho*rho_noise[row_index]
            final_column[row_index] = feature
        rho_dict[rho] = final_column
    
    
    noise = np.random.normal(scale=0.1,size=n_samples)
    # closure, no need to complete on your own
    # also includes decorator so that your dataset will be the same even with repeated rhos.
    @cache_data
    def create_x_y(rho, X=X, beta=true_beta, noise=noise, rho_dict = rho_dict):
        final_feature = rho_dict[rho]
        X = np.append(X,final_feature,axis=1)
        return X, np.dot(X,beta) + noise
        
    return true_beta,create_x_y

# starter code -- you should only run this once
rhos = [0.02,0.05,0.1,0.2,0.5,1.0]
beta_star,create_x_y = generate_dataset(rhos=rhos)

plt.figure(0, figsize=(12, 8))
Error = []
for rho in rhos:
    X,Y = create_x_y(rho)
    X_train = X[0:800,:]
    X_test = X[800:1000,:]
    Y_train = Y[0:800]
    Y_test = Y[800:1000]
    beta = (np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X_train), X_train)), np.transpose(X_train)), Y_train))
    Y_train_hat = np.matmul(X_train, beta)
    Y_test_hat = np.matmul(X_test, beta)
    sum = 0
    Error.append((1/len(beta_star)) * np.sum(np.square(np.subtract(beta, beta_star))))
plt.plot(rhos, Error, label = 'Error')
plt.legend(loc=2)
plt.title('Error vs rho')
plt.show()

# code here
def create_and_train_model(X, Y, model_name = "Ridge", alphas = [1.0]):
    # This assumes the following parameters:
    #      X           : an (n_samples,n_features) array that contains the input features
    #      Y           : an (n_samples,1) array that contains the values you want to regress to
    #      model_name  : a string containing either "Ridge" or "Lasso"
    #      alphas      : list of possible regularization paremeters for the corresponding term in the model.
    
    if model_name == "Ridge":
        # initialize a RidgeCV regression model
        model = RidgeCV(alphas=alphas)
    elif model_name == "Lasso":
        # initialize a LassoCV regression model
        model = LassoCV(alphas=alphas)
    else:
        raise Exception("Invalid model name provided")
    
    # train/fit model
    model = model.fit(X,Y)
    
    # find the best alpha found
    best_alpha = model.alpha_
    
    return model, best_alpha

def beta_error(model, beta_star):
    # This assumes the following parameters:
    #      model     : Scikit learn linear_model that has ALREADY been trained
    #      beta_star : the beta_star computed in part 1 for some rho
    
    # extract the beta from the model
    beta = model.coef_
    # find the MSE between beta and beta_star
    error = (np.linalg.norm(beta-beta_star))

    return error

alphas=[10,1.0,0.1,0.001,0.0001]
rhos = [0.02,0.05,0.1,0.2,0.5,1.0]
Error_lasso = []
Error_ridge = []
Error_linear = []
for rho in rhos:
    X,Y = create_x_y(rho)
    
    X = X[0:800,:]
    Y = Y[0:800]
    
    
    model, best_alpha = create_and_train_model(X,Y,'Ridge', alphas)
    error = beta_error(model, beta_star)
    Error_ridge.append(error)
    
    model, best_alpha = create_and_train_model(X,Y,'Lasso', alphas)
    error = beta_error(model, beta_star)
    Error_lasso.append(error)
    
    beta = (np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X)),Y))
    error = (np.linalg.norm(beta-beta_star))
    Error_linear.append(error)

    

    
plt.figure(0, figsize=(12, 8))
#plt.ylim(0, 0.004)
plt.plot(rhos, Error_lasso, c= 'red', label = 'Error Lasso', ms =3)
plt.plot(rhos, Error_ridge, c= 'blue', label = 'Error Ridge', ms =3)
plt.plot(rhos, Error_linear, c= 'yellow', label = 'Error Linear', ms =3)
plt.title('Errors')
plt.legend(loc=2)
plt.show()

