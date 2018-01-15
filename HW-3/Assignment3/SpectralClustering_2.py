import scipy.io as sio
from numpy import shape, zeros, sqrt
import numpy as np
import math

from scipy.linalg import eigh
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



def plotData(X,C,closest,title):
    plt.scatter(X[:, 0], X[:, 1], c=closest)
    plt.scatter(C[:, 0], C[:, 1], c='r')
    plt.title(title)
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

    sigSq = sig

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
            if rowSum != 0:
                y[i][j] = X[i][j]/rowSum

    return y

def SpectralClustering_D(X,k,sigma):

    (n, d) = shape(X)

    for sig in sigma:

        A = GetAffinityMatrix(X,sig)

        D = GetDiagonalMatrix(A,n)

        L = D - A

        w, v = eigh(L, eigvals=(0,k-1))

        Y = GetNormalMatrix(v)

        # newK = KMeans(n_clusters=k).fit(Y)
        # print (newK.cluster_centers_)
        import k_means
        obj = k_means.K_means(Y)
        optimul_cluster_indice, distance, optimul_centroid = obj.run_k_means()
        title = 'Clustered data for sigma = '+str(sig)
        plotData(X, optimul_centroid, optimul_cluster_indice,title)

        # plotData(X, newK.cluster_centers_, newK.labels_)
        # break

if __name__ == "__main__":

    trainX = readMatlabFile("./input_files/data.mat")
    trainX = np.array(trainX)

    X = trainX.T

    total_observation = X.shape[0]

    k = 4
    sigma = [0.001, 0.01, 0.1, 1]
    sigma1 = [1]

    SpectralClustering_D(X, k, sigma)