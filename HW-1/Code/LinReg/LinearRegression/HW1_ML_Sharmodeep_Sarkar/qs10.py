## Author :: Sharmodeep Sarkar
## Reference: http://aimotion.blogspot.com/2011/11/machine-learning-with-python-logistic.html
## The approach has been discussed with other ML Classmates 

import scipy.io as sio
from matplotlib.pyplot import scatter, xlabel, ylabel, legend, show
from numpy import loadtxt, where
import numpy as np
from scipy.optimize import fmin_bfgs

data_File_Path = 'Z:\ML\HW-1\Code\LinReg\LinearRegression\logistic_regression.mat'

## Calculates the sigmoid function
def sigmoid(theta, it, m):
   theta_t = np.transpose(theta)
   z = np.dot(it, theta_t)
   h = 1.0 / (1.0 + np.exp(-1.0 * z))
   return h

## Calculates the log(h) and also log(1-h) used for computing the cost
def logsig(theta, it, yb):
   theta_t = np.transpose(theta)
   z = np.dot(it, theta_t)
   logh = -np.log(1.0 + np.exp(-1.0 * z))
   loghc = (-1.0 * z) - np.log(1.0 + np.exp(-1.0 * z))
   return logh, loghc

## Calculates the cost function
def cost(theta, it, yb, m, n, penalty):
   [lh, lhc] = logsig(theta, it, yb)
   J = ((np.dot(-yb.T, lh)) - (np.dot((1.0 - yb.T), lhc)))
   J = J.sum() / m
   Th = theta[1:n] ** 2
   R = Th.sum() * (penalty / (2*m))
   J = J + R
   return J

## Calculates the gradient (grad) of the cost function
def comp_grad(theta, it, yb, m, n, penalty):
   h = sigmoid(theta, it, m)
   grad = np.dot((h.T-yb), it)
   R = np.zeros(shape=(1, n))
   R[0, 1:n] = penalty * theta[1:n].T
   G = (grad + R) / m
   return G[0]

## Calculates the prediction class depending on the learned parameters
def predict(all_3_theta, it):
   m, n = it.shape
   h = sigmoid(all_3_theta, it, m)
   p = np.argmax(h, axis=1)+1
   return p

# Calculates accuracy of the prediction against the original (gold) values
def get_accuracy(prediction, gold):
   hit = 0
   for q in range(0, prediction.size):
       if prediction[q] == gold[q]:
           hit += 1
   accu = hit / prediction.size
   return accu

## reshapes the input X to add a column of 1s
def reshape (X):
  m, n = X.shape
  it = np.ones(shape=(m, n+1))
  it[:, 1:n+1] = X  #Added the first column of 1s
  return it 

def main():
  ## loading raw data
  raw_data = sio.loadmat(data_File_Path)
  X_trn = raw_data['X_trn']
  Y_trn = raw_data['Y_trn']
  X_test = raw_data['X_tst']
  Y_test = raw_data['Y_tst']

  ## defining the classes
  class0 = where(Y_trn == 0)
  class1 = where(Y_trn == 1)
  class2 = where(Y_trn == 2)

  ## Draw Scatter Plot
  ##Plot for 3 categories
  # scatter(X_trn[class0, 0], X_trn[class0, 1], marker='o', c='b')
  # scatter(X_trn[class1, 0], X_trn[class1, 1], marker='x', c='r')
  # scatter(X_trn[class2, 0], X_trn[class2, 1], marker='+', c='r')
  # xlabel('X1')
  # ylabel('X2')
  # legend(['0', '1','2'])
  # show()

  ## Total classes considered (label count)
  lbl_cnt = 3  

  ## Reshaping training data
  reshaped_X_trn = reshape(X_trn)
  m, n = reshaped_X_trn.shape
  Y_trn = Y_trn[:, 0]
  ## initializing matrix to hold a single W
  W = np.zeros(shape=(1, n))
  ## Small penalty factor
  penalty = 0.1  
  ## Holding all 3 Ws for the 3 classes
  W_all = np.zeros(shape=(lbl_cnt, n))  # 3X3 matrix 
  

  for label in range(0, lbl_cnt):
    # These function is called by the fmin_bfgs (optimization function) and provides J and grad
    def f(W):
       bb = np.where(Y_trn == label)
       yb = np.zeros(shape=(m, ))
       yb[bb] = 1
       return cost(W, reshaped_X_trn, yb, m, n, penalty)

    def fprime(W):
       bb = np.where(Y_trn == label)
       yb = np.zeros(shape=(m, ))
       yb[bb] = 1
       return comp_grad(W, reshaped_X_trn, yb, m, n, penalty)

    print ('\nClass -> ', label)
    W_all[label-1, :] = fmin_bfgs(f, W, fprime, disp=True, maxiter=400)
  print ('\n\nOptimal W ')
  print(W_all)


  ## Running prediction on Train data and measuring accuracy
  prediction_trn = predict(W_all, reshaped_X_trn)
  print ('Training Data Classification Accuracy = ',get_accuracy(prediction_trn, Y_trn))

  ## Reshaping test data
  reshaped_X_test = reshape(X_test)
  Y_test = Y_test[:, 0]
  ## Running prediction on Test data and measuring accuracy
  prediction_test = predict(W_all, reshaped_X_test)
  print ('Test Data Classification Accuracy = ',get_accuracy(prediction_test, Y_test))

if __name__ == "__main__":
    main()