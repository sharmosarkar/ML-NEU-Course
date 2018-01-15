import scipy.io as sio
from numpy import shape, zeros, sqrt,linalg,take
import numpy as np
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def plotData(X,C,closest):
    plt.scatter(X[:, 0], X[:, 1], c=closest)
    plt.scatter(C[:, 0], C[:, 1], c='r')
    plt.show()

def readMatlabFile(filename):
    '''
    :param filename: The name of the file whose content needs to be returned
    :return:
    '''
    mat_contents = {}
    try:
        mat_contents = sio.loadmat(filename)
    except Exception as e:
        print(('Exception : {0}').format(e.message))
    return mat_contents['X_Question2_3']

def GetAffinityMatrix(X,sig):
    (n, d) = shape(X)

    affinityMatrix = zeros((n, n))

    sigSq = 2*sig*sig

    for i in range(n):
        for j in range(n):
            if i != j:
                diff = sqrt(np.sum((X[i] - X[j]) ** 2))
                diff = (-1) * diff
                score = diff/sigSq
                affinityMatrix[i][j] = math.exp(score)

    return affinityMatrix

def GetDiagonalMatrix(A,n):

    diagonalMatrix = np.zeros((n, n))

    for i in range(n):
        rowSum = np.sum(A[i])
        diagonalMatrix[i][i] = rowSum

    return diagonalMatrix

def GetNormalMatrix(X):

    (n,d) = shape(X)

    y = np.zeros((n,d))

    xSQ = X**2

    for i in range(n):
        rowSum = sqrt(np.sum(xSQ[i]))
        for j in range(d):
            y[i][j] = X[i][j]/rowSum

    return y

def SpectralClustering(X,k,sigma):
    (n, d) = shape(X)

    for sig in sigma:
        A = GetAffinityMatrix(X,sig)

        D = GetDiagonalMatrix(A,n)

        L = D - A

        eigen_values, eigen_vectors = linalg.eigh(L)

        Xk = take(eigen_vectors,[196,197,198,199],0)

        #eigen_vectors = (-1) * eigen_vectors

        Xk = Xk.T

        Y = GetNormalMatrix(Xk)

        newK = KMeans(n_clusters=k).fit(Y)

        plotData(X,newK.cluster_centers_,newK.labels_)



if __name__ == "__main__":

    trainX = readMatlabFile("./input_files/data.mat")
    trainX = np.array(trainX)

    X = trainX.T

    total_observation = X.shape[0]

    k = 4
    sigma1 = [0.001, 0.01, 0.1, 1]
    sigma = [0.01]
    SpectralClustering(X, k, sigma)