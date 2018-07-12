#!/usr/local/bin/python
import os
import pickle
import numpy as np
from xy_extract import proc_extract
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as mtr

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

clf = LogisticRegression(multi_class='multinomial',\
    solver='saga', max_iter=100)

x_train_pca = pca.transform(x_train_scale)
x_test_pca = pca.transform(x_test_scale)
clf.fit(x_train_pca, y_train)

train_pred = clf.predict(x_train_pca)
print('train accuracy: {0}'.format(mtr.accuracy_score(y_train, train_pred)))

test_pred = clf.predict(x_test_pca)
print('test accuracy: {0}'.format(mtr.accuracy_score(y_test, test_pred)))

model = './pickles/lf_model.pickle'
try:
    with open(model, 'wb') as sp:
        pickle.dump(clf,sp)
except Exception as e:
    print('Unable to save {0} : {1}'.format(model, e))


# end of file
