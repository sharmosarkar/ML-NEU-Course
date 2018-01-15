'''
    This file contains an implementation of multi-class SVM with the SMO algorithm. 
    It reports the accuracy of classification for differnt values of C
    Ref :: http://cs229.stanford.edu/materials/smo.pdf
'''

import numpy as np
import scipy.io
from numpy import loadtxt, where
import matplotlib.pyplot as plt

DATA_LOCATION = 'Z:/ML/HW-2/code/data.mat'

class SVM (object):

    def __init__(self): 
        self.max_iteration = 1000
        self. tolerance = 0.01
        self.class_cnt = 3

    def select_random_J(self,i, m):
        j = i  # we want to select any J not equal to i
        while (j == i):
            j = int(np.random.uniform(0, m))
        return j

    def clipAlpha(self, aj, H, L):
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj    

    def train_SVM(self,dataIn, classLabels, C, tolerance, maxIter):
        dataMatrix = np.mat(dataIn)
        labelMat = classLabels
        m, n = np.shape(dataMatrix)
        alphas = np.mat(np.zeros((m, 1)))
        bias = 0
        iter = 0
        while (iter < maxIter):
            alphaPairsChanged = 0
            for i in range(m):
                # Evaluate the model i
                fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + bias
                Ei = fXi - float(labelMat[i])  # if checks if an example violates KKT conditions
                # Check if we can optimize (alphas always between 0 and C)
                if ((labelMat[i] * Ei < -tolerance) and (alphas[i] < C)) or \
                        ((labelMat[i] * Ei > tolerance) and (alphas[i] > 0)):
                    # Select a random J
                    j = self.select_random_J(i, m)
                    # Evaluate the mode j
                    #                fXj = evaluate(j, alphas, labelMat, dataMatrix, bias)
                    #                Ej = fXj - float(labelMat[j])
                    fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + bias
                    Ej = fXj - float(labelMat[j])
                    # Copy alphas
                    alpha_old_i = alphas[i].copy()
                    alpha_old_j = alphas[j].copy()
                    # Check how much we can change the alphas
                    # L = Lower bound
                    # H = Higher bound
                    if labelMat[i] != labelMat[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(C, C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[j] + alphas[i] - C)
                        H = min(C, alphas[j] + alphas[i])
                    # If the two correspond
                    if L == H:
                        #print("L is H")
                        continue
                    # Calculate ETA
                    eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                          dataMatrix[i, :] * dataMatrix[i, :].T - \
                          dataMatrix[j, :] * dataMatrix[j, :].T
                    if eta >= 0:
                        #print("eta is bigger than 0")
                        continue
                    # Update J and I alphas
                    alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                    alphas[j] = self.clipAlpha(alphas[j], H, L)
                    # If alpha is not moving enough, continue..
                    if abs(alphas[j] - alpha_old_j) < 0.00001:
                        #print("delta alpha is too big")
                        continue
                    # Change alpha I for the exact value, in the opposite
                    # direction
                    alphas[i] += labelMat[j] * labelMat[i] * \
                                 (alpha_old_j - alphas[j])
                    # Update bias
                    b1 = bias - Ei - labelMat[i] * (alphas[i] - alpha_old_i) * \
                                     dataMatrix[i, :] * dataMatrix[i, :].T - \
                         labelMat[j] * (alphas[j] - alpha_old_j) * \
                         dataMatrix[i, :] * dataMatrix[j, :].T
                    b2 = bias - Ej - labelMat[i] * (alphas[i] - alpha_old_i) * \
                                     dataMatrix[i, :] * dataMatrix[i, :].T - \
                         labelMat[j] * (alphas[j] - alpha_old_j) * \
                         dataMatrix[j, :] * dataMatrix[j, :].T
                    # Choose bias to set
                    if 0 < alphas[i] and C > alphas[i]:
                        bias = b1
                    elif 0 < alphas[j] and C > alphas[j]:
                        bias = b2
                    else:
                        bias = (b1 + b2) / 2.0
                    # Increment counter and log
                    alphaPairsChanged += 1
                if alphaPairsChanged == 0:
                    iter += 1
                else:
                    iter = 0
                
            return alphas, bias

    ## Predicts the clss of an unknown data point using the SVM algorithm
    def run_SVM(self,alpha, b, X, Y, data_point):
        val = 0
        i = 0
        data_point = data_point.T
        for point in X:
            part1 = alpha[i] * Y[i]
            part2 = X[i] * data_point
            out = part1 * part2
            val += out
            i += 1
        return (val + b)

    ## prediction wrapper
    def prediction (self, xTest,alpha,b,X, Y):
        prediction_list = []
        for x in xTest:
            probable_cls_lst = []   ## this list has the probability of the datapoint belonging to each class
            for cls in range(self.class_cnt):
                pred = self.run_SVM(alpha[cls],b[cls], X, Y, np.matrix(x))
                probable_cls_lst.append(pred)
            ## the highest probability is the class that the datapoint belongs to
            final_cls = probable_cls_lst.index(max(probable_cls_lst))
            prediction_list.append(final_cls)
        return prediction_list



def plot_graph(x,y):
    print(x)
    print(y)
    plt.plot(x,y)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Values of C')
    plt.title ('Variation of accuracy wrt to Values of C')
    plt.show()

def get_input_data():
    data = scipy.io.loadmat(DATA_LOCATION)
    return data['X_trn'], data['Y_trn'], data['X_tst'], data['Y_tst']

def calc_accuracy(pred_lst,gold_lst):
    hits = 0.0
    for pred,gold in zip(pred_lst,gold_lst):
        if gold == pred:
            hits += 1
    return (hits*1.0 / len(pred_lst))*100

## This function gets the alphas and bias(es) for different C(s) in the c_list and plots 
##  a graph for the C(s) vs the accuracy of predicting the test samples using the corresponding
##  alpha(s) and bias(es)  
def control_SVM(c_list,X,Y,X_tst,Y_tst):
    ## converts the class to its one-hot class eg. 0 is [1,-1,-1].
    ## returns a list of class labels.
    ## eg : Y = [0,1,2]
    ## label_lst = [[1,-1,-1],
    ##              [-1,1,-1],
    ##              [1,-1,1]]
    def make_one_hot_labels(Y,class_cnt):
        label_lst = []
        for cls in range(class_cnt):
            labels = [1 if i == cls else -1 for i in Y]
            label_lst.append(labels)
        return label_lst


    svm = SVM()
    accuracyList = []
    one_hot_label_lst = make_one_hot_labels(Y,svm.class_cnt)
    # Running for different ranges of C
    for c in c_list:
        alphaList = []
        bList = []
        models = []
        for labels in one_hot_label_lst:            
            alpha, b = svm.train_SVM(X, np.mat(labels).T, c, svm.tolerance, svm.max_iteration )
            alphaList.append(alpha)
            bList.append(b)
            models.append((labels, b, alpha))
        predictions = svm.prediction(X_tst,alphaList,bList,X, Y)
        accuracy = calc_accuracy(predictions,Y_tst)
        accuracyList.append(accuracy)
        del alphaList,bList,models
        print("C : ",c,"\taccuracy : ", accuracy,'%')
    return accuracyList


def main():
    X, Y, X_tst, Y_tst = get_input_data()
    C_values = [0.01,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0]

    accuracyList= control_SVM(C_values,X,Y,X_tst,Y_tst)
    plot_graph(C_values,accuracyList)

if __name__ == "__main__":
    main()
