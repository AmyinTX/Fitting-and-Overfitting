# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 16:38:36 2017

@author: abrown09
"""

#%% Import packages
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import matplotlib.pyplot as plt
#%% Set seed for reproducible results
np.random.seed(414)

#%% Gen toy data
X = np.linspace(0, 15, 1000)
y = 3 * np.sin(X) + np.random.normal(1 + X, .2, 1000)

#%% split into training and testing datasets
train_X, train_y = X[:700], y[:700]
test_X, test_y = X[700:], y[700:]

train_df = pd.DataFrame({'X': train_X, 'y': train_y})
test_df = pd.DataFrame({'X': test_X, 'y': test_y})

#%% create quadratic and cubic terms
train_df['X-squared'] = (train_df['X'])**2
train_df['X-cubed'] = (train_df['X'])**3

test_df['X-squared'] = (test_df['X'])**2
test_df['X-cubed'] = (test_df['X'])**3
#%% reshape data
testX = test_df['X'].reshape((300,1))
testy = test_df['y'].reshape((300,1))

X = train_df['X'].reshape((700,1))
y = train_df['y'].reshape((700,1))

#%% modeling

#%% model1
m1_train_X = train_df['X'].reshape((700,1))
m1_test_X = test_df['X'].reshape((300,1))

m1 = linear_model.LinearRegression()
r1 = m1.fit(m1_train_X, y)
print(r1.intercept_, r1.coef_)

m1_pred_train = m1.predict(m1_train_X)
m1_pred_test = m1.predict(m1_test_X)

train_m1_mse = print(round(float(mean_squared_error(y, m1_pred_train)), 2))
test_m1_mse = print(round(float(mean_squared_error(testy, m1_pred_test)), 2))
#%% model2
m2_train_X = train_df[['X', 'X-squared']]
m2_test_X = test_df[['X', 'X-squared']]

m2 = linear_model.LinearRegression()
r2 = m2.fit(m2_train_X, y)
print(r2.intercept_, r2.coef_)

m2_pred_train = m2.predict(m2_train_X)
m2_pred_test = m2.predict(m2_test_X)

train_m2_mse = print(round(float(mean_squared_error(y, m2_pred_train)), 2))
test_m2_mse = print(round(float(mean_squared_error(testy, m2_pred_test)), 2))

#%% model 3
m3_train_X = train_df[['X', 'X-squared', 'X-cubed']]
m3_test_X = test_df[['X', 'X-squared', 'X-cubed']]

m3 = linear_model.LinearRegression()
r3 = m3.fit(m3_train_X, y)
print(r3.intercept_, r3.coef_)

m3_pred_train = m3.predict(m3_train_X)
m3_pred_test = m3.predict(m3_test_X)

train_m3_mse = print(round(float(mean_squared_error(y, m3_pred_train)), 2))
test_m3_mse = print(round(float(mean_squared_error(testy, m3_pred_test)), 2))

#%%
# plot MSEs--cannot seem to change the datatype from nonetype. will have to mantually enter into a df

mse_data = {
    'model': ['1', '2', '3'],
    'mse_train': [4.06, 3.79, 3.05],
    'mse_test': [6.55, 7.99, 199.65]}

mse_df = pd.DataFrame(mse_data, columns = ['model', 'mse_train', 'mse_test'])

n_groups = 3
mse_train = (4.06, 3.79, 3.05)
mse_test = (6.55, 7.99, 199.65)
bar_width = 0.35
opacity = 0.8

fig, ax = plt.subplots()
index = np.arange(n_groups)

rects1 = plt.bar(index, mse_train, bar_width, alpha=opacity, color='b', label='Training MSE')

rects2 = plt.bar(index + bar_width, mse_test, bar_width, alpha=opacity, color='g', label='Testing MSE')

plt.xlabel('Model')
plt.ylabel('Mean Standard Errors')
plt.title('Mean Standard Error by Model, for Testing and Training Data')
plt.xticks(index + bar_width, ('Model 1', 'Model 2', 'Model 3'))
plt.legend()

plt.show()
