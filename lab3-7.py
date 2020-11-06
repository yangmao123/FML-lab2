# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 10:47:49 2020

@author: 75965
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path
from sklearn import datasets
from sklearn.linear_model import Lasso
import pandas as pd
from sklearn.model_selection import train_test_split
import os
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