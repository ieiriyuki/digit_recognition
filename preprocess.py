#!/usr/local/bin/python
import csv
import numpy as np
from sklearn.linear_model import LogisticRegression as LR

data = np.loadtxt('./data/train.csv',delimiter=',', skiprows=1)
print("data shape is {0}".format(data.shape))

y_data, x_data = np.split(data, [1],axis=1)

print("x_data shape is {0}".format(x_data.shape))
print("y_data shape is {0}".format(y_data.shape))


# end of file
