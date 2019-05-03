#!/usr/bin/env python
# coding: utf-8

# # Lab 1

# ### 1.

# Create 1000 samples from a Gaussian distribution with mean -10 and standard deviation 5. Create another 1000 samples from another independent Gaussian with mean 10 and standard deviation 5.
# 
# 
# 1. Take the sum of 2 these Gaussians by adding the two sets of 1000 points, point by point, and plot the histogram of the resulting 1000 points. What do you observe?
# 2. Estimate the mean and the variance of the sum.

# ### Answer

# #### 1.1 

# In[1]:


import numpy as np

mu1, sigma1 = -10, 5
mu2, sigma2 = 10, 5
samples = 1000

dis1 = np.random.normal(mu1, sigma1, samples)
dis2 = np.random.normal(mu2, sigma2, samples)


# In[2]:


import matplotlib.pyplot as plt

dis3 = dis1 + dis2
fig = plt.figure()
plt.hist(dis3, bins=16)


# We observe that the sum of the original sets is roughly centered around a mean of zero. The values also lie mostly between extremes of -20 and 20, indicating that the variance has increased.

# #### 1.2

# The estimated mean is 0 and the estimated variance is 50. We can observe the estimated mean in the plot above. The estimated variance is taken by summing the individual variances of each random variable, since they are independent and uncorrelated.

# ### 2.

# *Central Limit Theorem*. Let $X_i$ be an iid Bernoulli random variable with value $\{-1,1\}$.
# Look at the random variable $Z_n = \frac{1}{n} \sum X_i$. By taking 1000 draws from $Z_n$, plot its histogram. Check that for small $n$ (say, $5-10$) $Z_n$ does not look that much like a Gaussian, but when $n$ is bigger (already by the time $n = 30$ or $50$) it looks much more like a Gaussian. Check also for much bigger $n: n = 250$, to see that at this point, one can really see the bell curve.

# ### Answer

# In[3]:


draws = 1000
ns = [5, 10, 30, 50, 250]
p = 0.5

def clt(n, p, draws, bins=None):
    res = np.random.binomial(n, p, draws)
    #print(res)
    actual = []
    for ans in res:
        actual.append(ans - (n - ans))
    #print(actual)
    plt.hist(actual, bins=bins)
    
clt(ns[0], p, draws)


# In[4]:


clt(ns[1], p, draws)


# In[5]:


clt(ns[2], p, draws, bins=10)


# In[6]:


clt(ns[3], p, draws, bins=12)


# In[7]:


clt(ns[4], p, draws, bins=16)


# ### 3.

# Estimate the mean and standard deviation from $1$ dimensional data: generate $25,000$ samples from a Gaussian distribution with mean $0$ and standard deviation $5$. Then estimate the mean and standard deviation of this gaussian using elementary numpy commands, i.e., addition, multiplication, division (do not use a command that takes data and returns the mean or standard deviation).

# ### Answer

# In[8]:


mu, sigma, samples = 0, 5, 25000

dist = np.random.normal(mu, sigma, samples)
expectation = sum(dist)/samples
print("Estimated Expectation: {0}".format(expectation))
variance = sum([(x - expectation)**2 for x in dist]) / (samples - 1)
print("Estimated Variance: {0}".format(variance))


# ### 4.

# ### Answer

# In[9]:


mu = [-5, 5]
cov = [[20, 0.8], [0.8, 30]]
samples = 10000
dist = np.random.multivariate_normal(mu, cov, samples)
print(dist)


# In[10]:


X = dist[:, 0]
Y = dist[:, 1]

X_mean = sum(X)/samples
Y_mean = sum(Y)/samples

X_var = sum([(x - X_mean)**2 for x in X])/(samples - 1)
Y_var = sum([(y - Y_mean)**2 for y in Y])/(samples - 1)

X_sub_mean = [x - X_mean for x in X]
Y_sub_mean = [y - Y_mean for y in Y]

cov = sum([(x)*(y) for x, y in zip(X_sub_mean, Y_sub_mean)])/(samples-1)
print("X mean: {0}".format(X_mean))
print("Y mean: {0}".format(Y_mean))
print("Var(X) = {0}".format(X_var))
print("Var(y) = {0}".format(Y_var))
print("Covariance: [[{0}, {1}], [{1}, {2}]]".format(X_var, cov, Y_var))


# ### 5.

# Each row is a patient and the last column is the condition that the patient has. Do data exploration using Pandas and other visualization tools to understand what you can about the dataset. For example:
# 1. How many patients and how many features are there?
# 2. What is the meaning of the first 4 features? See if you can understand what they mean.
# 3. Are there missing values? Replace them with the average of the corresponding feature column
# 4. How could you test which features strongly influence the patient condition and which do not?
# 
# List what you think are the three most important features.

# In[42]:


import pandas as pd

patient_data = pd.read_csv('PatientData.csv', dtype="object")
patient_data.head()


# In[43]:


patient_data.shape


# ### Answer 

# #### 5.1

# There are 451 patients and 280 features

# #### 5.2

# Feature 0: Age in years
# 
# Feature 1: Gender
# 
# Feature 2: Height in centimeters
# 
# Feature 3: Weight in kilograms

# #### 5.3

# Yes, some columns, like column 13, contain missing values.

# In[55]:


for i in range (patient_data.shape[0]):
    col_sum = 0
    count = 0
    for j in range(patient_data.shape[1]):
        if (patient_data.values[i][j] == '?'):
            continue
        col_sum += float(patient_data.values[i][j])
        count += 1
        
    col_avg = col_sum/count
    for j in range(patient_data.shape[1]):
        if (patient_data.values[i][j] == '?'):
            patient_data.values[i][j] = str(col_avg)
            
patient_data = patient_data.astype("float")
print(patient_data.values)


# #### 5.4

# If we run linear regression on the data set, the magnitude of the coefficients $\beta_1,...,\beta_n$ will tell us which features have the biggest impact on the label. The larger the magnitude of the coefficient, the larger its influence.

# In[70]:


from sklearn import linear_model as lm
model = lm.LinearRegression()
X = [row[0:len(row) - 1] for row in patient_data.values]
y = [row[len(row) - 1] for row in patient_data.values]
model.fit(X,y)
beta = model.coef_
rank = np.argsort([abs(b) for b in beta])
for i in range(3):
    index = rank[-1 * i - 1]
    print("Index: " + str(index) + "   Beta coefficient value: " + str(beta[index]))


# The 3 most important features are displayed above in order of the magnitude of their coefficients. These come from the solution to the emperical risk minimization problem under the assumption that the patients' conditions is a numerical quantity.

# ### Answer 

# In[35]:


import numpy as np

v1 = np.asarray([1, 1, 1])
v2 = np.asarray([1, 0 ,0])

# Create the orthonormal basis
u1 = v2
u2 = v1 - (v1.dot(u1) * u1)/(np.linalg.norm(u1)**2)

p1 = np.asarray([3, 3, 3])
p2 = np.asarray([1, 2, 3])
p3 = np.asarray([0, 0, 1])

def project(u, v):
    u1 = u[0]
    u2 = u[1]
    
    return (u1.dot(v) * u1)/(np.linalg.norm(u1)**2) + (u2.dot(v) * u2) / (np.linalg.norm(u2)**2)

proj_p1 = project([u1, u2], p1)
proj_p2 = project([u1, u2], p2)
proj_p3 = project([u1, u2], p3)

print("The point {0} turns into {1} when projected onto the subspace spanned by v1 and v2".format(p1, project([u1, u2], p1)))
print("The point {0} turns into {1} when projected onto the subspace spanned by v1 and v2".format(p2, project([u1, u2], p2)))
print("The point {0} turns into {1} when projected onto the subspace spanned by v1 and v2".format(p3, project([u1, u2], p3)))

A = np.asarray([[1,1],
                [1,0],
                [1,0]])
def coordinates(A, p):
    mul = np.matmul
    inv = np.linalg.inv
    return mul(inv(mul(A.T,A)),A.T.dot(p))

c1 = coordinates(A, p1)
c2 = coordinates(A, p2)
c3 = coordinates(A, p3)

print("\nThe coordinates of p1 with respect to the basis {{v1, v2}} is ({0:.2},{1:.2})".format(c1[0], c1[1]))
print("The coordinates of p2 with respect to the basis {{v1, v2}} is ({0:.2},{1:.2})".format(c2[0], c2[1]))
print("The coordinates of p3 with respect to the basis {{v1, v2}} is ({0:.2},{1:.2})".format(c3[0], c3[1]))

