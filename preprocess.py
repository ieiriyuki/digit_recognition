#!/usr/local/bin/python
import os
import pickle
import numpy as np
from xy_extract import xy_extract
from sklearn.model_selection import train_test_split as tts
from sklearn.decomposition import PCA

pfile = './pickles/stored_data.pickle'
# a file storing x_train, test, y_train, test

if os.path.exists(pfile):
    print("%s exists - load pickle" % pfile)
    with open(pfile, 'rb') as lp:
        stored_data = pickle.load(lp)
    x_train, x_test, y_train, y_test = xy_extract(stored_data)
else:
    data = np.loadtxt('./data/train.csv',delimiter=',', skiprows=1)
    print("data shape is {0}".format(data.shape))
    y_data, x_data = np.split(data, [1], axis=1)
    print("x_data shape is {0}".format(x_data.shape))
    print("y_data shape is {0}".format(y_data.shape))

    x_train, x_test, y_train, y_test = tts(\
        x_data, y_data, test_size=0.3)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    stored_data = {
        'x_train' : x_train,
        'x_test' : x_test,
        'y_train' : y_train,
        'y_test' : y_test
    }
    try:
        with open(pfile, 'wb') as sp:
            pickle.dump(stored_data, sp)
    except Exception as e:
        print('Unable to save data to %s' % pfile, ': ', e)

proc_file = './pickles/proced_data.pickle'
if os.path.exists(proc_file):
    print('{0} exists - Load pickle'.format(proc_file))
    with open(proc_file,'rb') as lp:
        proced_data = pickle.load(lp)

else:
    xt_mean = np.mean(x_train, axis=0)
    xt_var = np.var(x_train, axis=0)
    xt_var = np.where(xt_var == 0., 1., xt_var)

    x_train_scale = (x_train - xt_mean) / xt_var
    x_test_scale = (x_test - xt_mean) / xt_var
    pca = PCA(n_components=20)
    pca.fit(x_train_scale)

    proced_data = {
        'x_train_scale': x_train_scale,
        'x_test_scale': x_test_scale,
        'xt_mean': xt_mean,
        'xt_var': xt_var,
        'y_train': y_train,
        'y_test': y_test,
        'pca': pca
    }
    try:
        with open(proc_file,'wb') as sp:
            pickle.dump(proced_data, sp)
    except Exception as e:
        print('Unable to save data to {0}'.format(proc_file))

'''
pc 20 is enough
pc: var
0: 0.4228613818946964
1: 0.845491567870124
2: 0.9018912897159757
3: 0.9312168227179378
4: 0.9520890036533969
5: 0.9605108547484609
6: 0.9688366518115029
7: 0.9722400234212724
8: 0.975301137788885
9: 0.9779009078137434
10: 0.9803081708061244
11: 0.9824630813853049
12: 0.984405663561418
13: 0.9862738013013584
14: 0.987702593141063
15: 0.9890563210767669
16: 0.9899487002203168
17: 0.9905563265915218
18: 0.9911300398829489
19: 0.9916408050061025
20: 0.9921339583513314
'''

# end of file
