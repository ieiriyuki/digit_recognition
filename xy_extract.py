#!/usr/local/bin/python

def xy_extract(stored_data):
    x_train = stored_data['x_train']
    x_test = stored_data['x_test']
    y_train = stored_data['y_train']
    y_test = stored_data['y_test']

    return x_train, x_test, y_train, y_test

def proc_extract(proced_data):
    x_train_scale = proced_data['x_train_scale']
    x_test_scale = proced_data['x_test_scale']
    xt_mean = proced_data['xt_mean']
    xt_var = proced_data['xt_var']
    y_train = proced_data['y_train']
    y_test = proc_extract['y_test']
    pca = proced_data['pca']

    return True

# end of file
