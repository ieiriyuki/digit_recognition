#!/usr/local/bin/python
import os
import pickle
import numpy as np
import pandas as pd
from xy_extract import proc_extract
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

pfile = './pickles/proced_data.pickle'
try:
    with open(pfile, 'rb') as lp:
        proced_data = pickle.load(lp)
except Exception as e:
    print('{0} does not exist'.format(pfile))

x_train_scale = proced_data['x_train_scale']
x_test_scale = proced_data['x_test_scale']
xt_mean = proced_data['xt_mean']
xt_var = proced_data['xt_var']
y_train = proced_data['y_train']
y_test = proced_data['y_test']
pca = proced_data['pca']

model = './pickles/lf_model.pickle'
try:
    with open(model, 'rb') as lp:
        clf = pickle.load(lp)
except Exception as e:
    print('Unable to load {0} : {1}'.format(model, e))

try:
    data = np.loadtxt('./data/test.csv',delimiter=',', skiprows=1)
except Exception as e:
    print('Unable to read test.csv: ', e)

print("data shape is {0}".format(data.shape))
data_scale = (data - xt_mean) / xt_var
data_pca = pca.transform(data_scale)

data_pred = clf.predict(data_pca).astype(int)

submit = pd.DataFrame({'ImageId': [ i + 1 for i in range(len(data_pred))],
                        'Label': data_pred})

print(submit.head(3))

submit.to_csv('./data/pc20_lr_submission.csv', index=False)

# end of file
