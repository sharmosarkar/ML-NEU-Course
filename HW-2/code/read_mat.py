from numpy import loadtxt, zeros, ones, array, linspace, logspace, mean, std, arange
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import scipy.io as sio
import numpy as np

#Load the dataset

mat_contents = sio.loadmat('Z:/ML/HW-2/code/data.mat', squeeze_me=True)
# mat_contents = sio.loadmat('linear_regression.mat')
X_trn = mat_contents['X_trn']
Y_trn = mat_contents['Y_trn']
X_tst = mat_contents['X_tst']
Y_tst = mat_contents['Y_tst']
xTrans = X_trn.shape
print ("X_trn",X_trn)
#print ("X_tst",X_tst)
# print ("Y_tst",Y_tst)
# print ("Y_trn",Y_trn)