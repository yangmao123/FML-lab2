# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 07:29:35 2020

@author: 75965
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path
from sklearn import datasets
from sklearn.linear_model import Lasso
import pandas as pd
from sklearn.model_selection import train_test_split

# Synthetic data:
# Problem taken from Hastie, et al., Statistical Learning with Sparsity
# Z1, Z2 ~ N(0,1)
# Y = 3*Z1 -1.5*Z2 + 10*N(0,1) Noisy response
# Noisy inputs (the six are in two groups of three each)
# Xj= Z1 + 0.2*N(0,1) for j = 1,2,3, and
# Xj= Z2 + 0.2*N(0,1) for j = 4,5,6.
N = 100
y = np.empty(0)
X = np.empty([0,6])
for i in range(N):
    Z1= np.random.randn()
    Z2= np.random.randn()
    y = np.append(y, 3*Z1 - 1.5*Z2 + 2*np.random.randn())
    Xarr = np.array([Z1,Z1,Z1,Z2,Z2,Z2])+ np.random.randn(6)/5
    X = np.vstack ((X, Xarr.tolist()))
# Compute regressions with Lasso and return paths
#
alphas_lasso, coefs_lasso, _ = lasso_path(X, y, fit_intercept=False)
# Plot each coefficient
#
fig, ax = plt.subplots(figsize = (8,4))
for i in range(6):
    ax.plot(alphas_lasso, coefs_lasso[i,:])
ax.grid(True)
ax.set_xlabel("Regularization")
ax.set_ylabel("Regression Coefficients")
sol = pd.read_excel("Husskonen_Solubility_Features.xlsx", verbose=False)

colnames = sol.columns.tolist()
print(sol.shape)
print(sol.columns)
t = sol["LogS.M."].values
fig, ax = plt.subplots(figsize=(4,4))
ax.hist(t, bins=40, facecolor='m')
ax.set_title("Histogram of Log Solubility", fontsize=14)
X = sol[colnames[5:len(colnames)]]
N, p = X.shape
print(X.shape)
print(t.shape)
# Split data into training and test sets
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.3)
# Regularized regression
gamma = 2.3
w = np.linalg.inv(X_train.T @ X_train + gamma*np.identity(p)) @ X_train.T @ t_train
ll = Lasso(alpha=0.2)
ll.fit(X, t)
th_lasso = ll.predict(X)

th_train = X_train @ w.to_numpy()
th_test = X_test @ w.to_numpy()
# Plot training and test predictions
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,4))
ax[0].scatter(t_train, th_train, c='m', s=3)
ax[1].scatter(t_test, th_test, c='m', s=3)
ax[2].scatter(t_train, th_train, c='b', s=3)
ax[2].scatter(t_test, th_test, c='r', s=3)

# Over to you for implementing Lasso
plt.show()