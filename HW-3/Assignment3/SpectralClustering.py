


import math

import matplotlib.pyplot as plt

import numpy as np

import scipy.io as sio

from numpy import shape, zeros, sqrt, linalg, take

from sklearn.cluster import KMeans

from sklearn.cross_validation import train_test_split





def plotData(X, C, closest, title):

    plt.scatter(X[:, 0], X[:, 1], c=closest)

    plt.scatter(C[:, 0], C[:, 1], c='r')
    plt.title (title)
    plt.show()





def readMatlabFile(filename):

    

    # :param filename: The name of the file whose content needs to be returned

    # :return:

    

    mat_contents = {}

    try:

        mat_contents = sio.loadmat(filename)

    except Exception as e:

        print(('Exception : {0}').format(e.message))

    return mat_contents['X_Question2_3']





def GetAffinityMatrix(X, sig):

    (n, d) = shape(X)

    affinityMatrix = zeros((n, n))

    sigma = 2 * sig

    for i in range(n):

        for j in range(n):

            if i != j:

                diff = sqrt(np.sum((X[i] - X[j]) ** 2))

                diff = (-1) * diff

                # piazza post say it should be only sigma

                score = diff / sigma

                temp = math.exp(score)

                affinityMatrix[i][j] = temp

    return affinityMatrix





def GetDiagonalMatrix(A, n):

    diagonalMatrix = np.zeros((n, n))

    for i in range(n):

        rowSum = np.sum(A[i])

        diagonalMatrix[i][i] = rowSum



    return diagonalMatrix





def GetNormalMatrix(X):

    (n, d) = shape(X)

    y = np.zeros((n, d))

    xSQ = X ** 2

    for i in range(n):

        rowSum = sqrt(np.sum(xSQ[i]))

        for j in range(d):

            if rowSum == 0:

                rowSum = 1e10

            y[i][j] = X[i][j] / rowSum

    return y





def SpectralClustering(X, k, sigma):

    (n, d) = shape(X)



    for sig in sigma:
        print(sig)
        A = GetAffinityMatrix(X, sig)

        D = GetDiagonalMatrix(A, n)
        D1 =  np.diag([sum(Wi) for Wi in A]) 
        #print (np.array_equal(D,D1))
        L = D1 - A

        D_sq_inv = np.diag([D1[i][i] ** -0.5 for i in range(n)])
        #D_sq_inv = np.diag([D[i][i] ** -1 for i in range(n)])
        
        Ln = D_sq_inv.dot(L).dot(D_sq_inv)
        #Ln = D_sq_inv.dot(L)
        

        eigen_values, eigen_vectors = linalg.eigh(Ln)    

        # Xk = take(eigen_vectors, [196, 197, 198, 199], 0)

        # Xk = take(eigen_vectors, [n-1, n-2, n-3, n-4], 0)

        Xk = take(eigen_vectors, [n-1, n-2, n-3, n-4], 0)

        # eigen_vectors = (-1) * eigen_vectors

        Xk = Xk.T

        Y = GetNormalMatrix(Xk)

        newK = KMeans(n_clusters=k).fit(Y)

        plotData(X, newK.cluster_centers_, newK.labels_,sig)
        #break





if __name__ == "__main__":

    trainX = readMatlabFile("Z:/ML/HW-3/data/data.mat")



    trainX = np.array(trainX).T



    # trainX, test = train_test_split(trainX, train_size=1)

    X = trainX

    # print "x",X

    total_observation = X.shape[0]

    k = 4

    sigma1 = [0.001, 0.01, 0.1, 1, 4 , 10, 100]

    sigma = [0.01]

    SpectralClustering(X, k, sigma1)
