#!/usr/bin/env python
# coding: utf-8

#


import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import mean_squared_error as mse
from pandas import DataFrame

# Read data
x1 = np.loadtxt('data/1/x.txt', dtype=str).astype('float32')
y1 = np.loadtxt('data/1/y.txt', dtype=str).astype('float32')
x2 = np.loadtxt('data/2/x.txt', dtype=str).astype('float32')
y2 = np.loadtxt('data/2/y.txt', dtype=str).astype('float32')
x3 = np.loadtxt('data/3/x.txt', dtype=str).astype('float32')
y3 = np.loadtxt('data/3/y.txt', dtype=str).astype('float32')
x4 = np.loadtxt('data/4/x.txt', dtype=str).astype('float32')
y4 = np.loadtxt('data/4/y.txt', dtype=str).astype('float32')

# Plot data
fig = plt.figure(figsize=(15, 15))
plt.suptitle('Plot data', fontsize=18, fontweight='bold')
for (x, y, i) in zip((x1, x2, x3, x4), (y1, y2, y3, y4), (1, 2, 3, 4)):
    ax = plt.subplot(2, 2, i)
    ax.set_xlim(-9, 9)
    ax.set_ylim(-31000, 31000)
    plt.xlabel('x', fontsize=14, fontweight='bold')
    plt.ylabel('y', fontsize=14, fontweight='bold', rotation='horizontal')
    plt.title('dataset ' + str(i), fontsize=14, fontweight='bold')
    plt.scatter(x, y)
plt.show()


# ## Two-fold cross validation


RMSE = np.zeros((4, 11), dtype='float32')
kf = KFold(n_splits=2, shuffle=True)
for x, y, i in zip((x1, x2, x3, x4), (y1, y2, y3, y4), range(4)):
    N = len(x)
    for degree in range(11):
        rmse = 0
        for train_index, test_index in kf.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            p = np.polyfit(x_train, y_train, degree)
            y_predict = np.polyval(p, x_test)
            rmse += mse(y_test, y_predict, squared=False)
        rmse /= kf.get_n_splits()
        RMSE[i][degree] = rmse




df = DataFrame({'Dataset 1': RMSE[0], 'Dataset 2': RMSE[1], 'Dataset 3': RMSE[2], 'Dataset 4': RMSE[3]})
df.index.name = 'Degree'
df['Average'] = df.mean(numeric_only=True, axis=1)
df.loc['Average'] = df.mean()
df


# ## Five-fold cross validation



RMSE = np.zeros((4, 11), dtype='float32')
kf = KFold(n_splits=5, shuffle=True)
for x, y, i in zip((x1, x2, x3, x4), (y1, y2, y3, y4), range(4)):
    N = len(x)
    for degree in range(11):
        rmse = 0
        for train_index, test_index in kf.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            p = np.polyfit(x_train, y_train, degree)
            y_predict = np.polyval(p, x_test)
            rmse += mse(y_test, y_predict, squared=False)
        rmse /= kf.get_n_splits()
        RMSE[i][degree] = rmse





df = DataFrame({'Dataset 1': RMSE[0], 'Dataset 2': RMSE[1], 'Dataset 3': RMSE[2], 'Dataset 4': RMSE[3]})
df.index.name = 'Degree'
df['Average'] = df.mean(numeric_only=True, axis=1)
df.loc['Average'] = df.mean()
df


# ## Ten-fold cross validation




RMSE = np.zeros((4, 11), dtype='float32')
kf = KFold(n_splits=10, shuffle=True)
for x, y, i in zip((x1, x2, x3, x4), (y1, y2, y3, y4), range(4)):
    N = len(x)
    for degree in range(11):
        rmse = 0
        for train_index, test_index in kf.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            p = np.polyfit(x_train, y_train, degree)
            y_predict = np.polyval(p, x_test)
            rmse += mse(y_test, y_predict, squared=False)
        rmse /= kf.get_n_splits()
        RMSE[i][degree] = rmse





df = DataFrame({'Dataset 1': RMSE[0], 'Dataset 2': RMSE[1], 'Dataset 3': RMSE[2], 'Dataset 4': RMSE[3]})
df.index.name = 'Degree'
df['Average'] = df.mean(numeric_only=True, axis=1)
df.loc['Average'] = df.mean()
df


# ## N-fold cross validation




RMSE = np.zeros((4, 11), dtype='float32')
loo = LeaveOneOut()
for x, y, i in zip((x1, x2, x3, x4), (y1, y2, y3, y4), range(4)):
    N = len(x)
    for degree in range(11):
        rmse = 0
        for train_index, test_index in loo.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            p = np.polyfit(x_train, y_train, degree)
            y_predict = np.polyval(p, x_test)
            rmse += mse(y_test, y_predict, squared=False)
        rmse /= loo.get_n_splits(x)
        RMSE[i][degree] = rmse




df = DataFrame({'Dataset 1': RMSE[0], 'Dataset 2': RMSE[1], 'Dataset 3': RMSE[2], 'Dataset 4': RMSE[3]})
df.index.name = 'Degree'
df['Average'] = df.mean(numeric_only=True, axis=1)
df.loc['Average'] = df.mean()
df

