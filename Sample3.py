#!/usr/bin/env python
# coding: utf-8

# ### 1.

# In[1]:


import pandas as pd
import matplotlib

train = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
print(train.shape)
test = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')
print(test.shape)
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

test.head()


# ### 2.

# In[2]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook")
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np

matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
prices.hist()


# In[3]:


import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])


# In[4]:


all_data = pd.get_dummies(all_data)


# In[5]:


#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())


# In[6]:


#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

#test.head()
#X_test.shape
#X_train.shap


# ### 3.

# In[7]:


from sklearn.linear_model import Ridge, Lasso

def get_price_est(model, X_train, X_test, y_train, alpha=0.1):
    price_model = model(alpha=alpha)
    price_model.fit(X_train, y_train)
    y_hat = np.expm1(price_model.predict(X_test))
    return pd.DataFrame(y_hat, range(1461, 1461+1459), ['SalePrice'])
    
y_hat_lasso = get_price_est(Lasso, X_train, X_test, y)
y_hat_ridge = get_price_est(Ridge, X_train, X_test, y)

print(y_hat_lasso.shape)
print(y_hat_lasso)
y_hat_lasso.to_csv('lasso_res.csv', index_label='Id')
y_hat_ridge.to_csv('ridge_res.csv', index_label='Id')


# Lasso MSE = 0.21524
# Ridge MSE = 0.13029

# In[8]:


from sklearn.model_selection import GridSearchCV

alphas = np.linspace(-1, 1, 200)

lasso_grid = GridSearchCV(Lasso(), param_grid={'alpha': alphas}, cv=3)
lasso_grid.fit(X_train, y)
best_lasso = lasso_grid.best_estimator_
print(lasso_grid.best_params_)


# In[9]:


ridge_alphas = np.linspace(1, 10, 200)
ridge_grid = GridSearchCV(Ridge(), param_grid={'alpha': alphas}, cv=3)
ridge_grid.fit(X_train, y)

best_ridge = ridge_grid.best_estimator_
print(ridge_grid.best_params_)


# In[10]:


def get_price_est_grid(model, X_train, X_test, y_train, alpha=0.1):
    price_model = model
    price_model.fit(X_train, y_train)
    y_hat = np.expm1(price_model.predict(X_test))
    return pd.DataFrame(y_hat, range(1461, 1461+1459), ['SalePrice'])

y_hat_lasso = get_price_est_grid(lasso_grid.best_estimator_, X_train, X_test, y)
y_hat_ridge = get_price_est_grid(ridge_grid.best_estimator_, X_train, X_test, y)

y_hat_lasso.to_csv('lasso_res_grid.csv', index_label='Id')
y_hat_ridge.to_csv('ridge_res_grid.csv', index_label='Id')


# The Best lasso alpha is 0.0 and gives an RMSE=0.14135. The best ridge alpha we found was 1.0 and gives RMSE=0.12661.

# ### 4.

# In[11]:


import matplotlib.pyplot as plt

def get_l0_norm(estimator):
    non_zero_b =[x for x in estimator.coef_ if x != 0]
    return len(non_zero_b)

ys = []
alphas = np.linspace(10**-4, 1, 100)
for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y)
    ys.append(get_l0_norm(lasso))
    
plt.title('Alphas vs L0 norm')
plt.xlabel('Alpha')
plt.ylabel('L0 norm')
plt.plot(alphas, ys)
plt.show()


# ### 5.

# In[12]:


X_train_output = X_train

lasso_output = lasso_grid.best_estimator_.predict(X_train)
lasso_output_df = pd.DataFrame(data=lasso_output.tolist(), columns=['lasso_output'])

lasso_test_output = lasso_grid.best_estimator_.predict(X_test)
lasso_test_output_df = pd.DataFrame(data=lasso_test_output.tolist(), columns=['lasso_output'])

ridge_output = ridge_grid.best_estimator_.predict(X_train)
ridge_output_df = pd.DataFrame(data=ridge_output.tolist(), columns=['ridge_output'])

ridge_test_output = ridge_grid.best_estimator_.predict(X_test)
ridge_test_output_df = pd.DataFrame(data=ridge_test_output.tolist(), columns=['ridge_output'])

X_train_with_outputs = X_train_output.join(lasso_output_df).join(ridge_output_df)
X_test_with_outputs = X_test.join(lasso_test_output_df).join(ridge_test_output_df)


# In[13]:


alphas= np.linspace(10**-3, 2, 100)
ridge_est_output = GridSearchCV(Ridge(), param_grid={'alpha': alphas}, cv=3)
ridge_est_output.fit(X_train_with_outputs, y)
y_hat_output = np.expm1(ridge_est_output.best_estimator_.predict(X_test_with_outputs))

res = pd.DataFrame(y_hat_output, range(1461, 1461+1459), ['SalePrice'])
res.to_csv('ridge_with_outputs.csv', index_label='Id')


# In[14]:


# print(ridge_est_output.best_estimator_.coef_)


# The RMSE returned on our ensemble predictor is 0.12687, which is very close to the original ridge regression result. This is expected because we are adding two columns which are linear combinations of the original features.

# ### 6.

# In[16]:


from xgboost import XGBRegressor

# xgbreg = XGBRegressor()
# xgbreg.fit(X_train, y)
# xgb_y_hat = np.expm1(xgbreg.predict(X_test))
xgb_params = {
    'max_depth':[1,2, 3],
    'n_estimators':[100, 200, 300, 400, 500],
    'n_jobs': [8],
    'learning_rate': [0.2, 0.25, 0.3],
    'objective': ['reg:linear', 'reg:tweedie']
}

xgb_grid = GridSearchCV(XGBRegressor(), param_grid=xgb_params, cv=4, n_jobs=-1)
xgb_grid.fit(X_train, y)
xgb_y_hat = np.expm1(xgb_grid.best_estimator_.predict(X_test))

xgb_res = pd.DataFrame(xgb_y_hat, range(1461, 1461+1459), ['SalePrice'])
xgb_res.to_csv('xgb_out.csv', index_label='Id')


# In[17]:


xgb_grid.best_params_


# In[27]:


from sklearn.preprocessing import PolynomialFeatures

xgb_poly = PolynomialFeatures(degree=2)
X_train_poly = xgb_poly.fit_transform(X_train)
X_test_poly = xgb_poly.fit_transform(X_test)


# In[25]:


xgb_params = {
    'max_depth':[2,3,4,5],
    'n_estimators':[100, 300, 500],
    'n_jobs': [-1],
    'learning_rate': [0.1, 0.2]
}

xgb_grid = GridSearchCV(XGBRegressor(), param_grid=xgb_params, cv=4, n_jobs=-1, verbose=5)
xgb_grid.fit(X_train_poly, y)
xgb_y_hat = np.expm1(xgb_grid.best_estimator_.predict(X_test_poly))
xgb_res = pd.DataFrame(xgb_y_hat, range(1461, 1461+1459), ['SalePrice'])
xgb_res.to_csv('xgb_out_poly.csv', index_label='Id')


# In[26]:


xgb_grid.best_params_


# In[29]:


xgb_reg = XGBRegressor(max_depth=2, n_estimators=500, n_jobs=-1, learning_rate=0.1)
xgb_reg.fit(X_train_poly, y)
y_hat_boost = np.expm1(xgb_reg.predict(X_test_poly))
xgb_res = pd.DataFrame(y_hat_boost, range(1461, 1461+1459), ['SalePrice'])
xgb_res.to_csv('xgb_out_boost.csv', index_label='Id')


# In[ ]:




