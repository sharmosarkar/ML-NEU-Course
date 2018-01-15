'''
    This file implements SVM using various kernels from the sklearn package 
'''

import numpy as np
import scipy.io
from numpy import loadtxt, where
from sklearn import svm

DATA_LOCATION = 'Z:/ML/HW-2/code/data.mat'

def get_input_data():
    data = scipy.io.loadmat(DATA_LOCATION)
    return data['X_trn'], data['Y_trn'], data['X_tst'], data['Y_tst']

def run_SVM(X_train_data,Y_train_data,X_test_data,kernel_type='linear'):
    clf = svm.SVC(kernel=kernel_type, C=1.0)
    clf.fit(X_train_data, Y_train_data.ravel())
    predictions = clf.predict(X_test_data)
    return predictions

def calc_accuracy(pred_lst,gold_lst):
    hits = 0.0
    for pred,gold in zip(pred_lst,gold_lst):
        if gold == pred:
            hits += 1
    return (hits*1.0 / len(pred_lst))*100

def main():
    X_train_data, Y_train_data, X_test_data, Y_test_data = get_input_data()
    kernel_type_lst = ['linear','poly','sigmoid','rbf']
    for kernel_type in kernel_type_lst:
        print('Accuracy using ',kernel_type,' kernel = ',\
            calc_accuracy(run_SVM(X_train_data,Y_train_data,X_test_data,kernel_type),Y_test_data) , '%')

if __name__ == "__main__":
    main()