import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
    return mat_contents['X_Question1']

def PCA_Data(X, d):

    '''
    :param X: Given Input Data
    :param d: Reduced dimension value
    :return: PCA components
    '''

    # region 1. Compute Mean Matrix
    xmean = np.mean(X, axis=0)
    Xdiff = X - xmean

    # endregion

    U_x, s, V = np.linalg.svd(Xdiff.T)

    U = U_x[:, 0:d]
    meanU = np.mean(U, axis=0)
    #Y = np.dot(U.T, Xdiff)
    Y = U.T.dot(Xdiff.T).T
    return U, xmean, Y

if __name__ == "__main__":

    # region 1. Read Matlab File
    trainX = readMatlabFile("./input_files/data.mat")
    trainX = np.array(trainX)
    print(trainX.shape)
    trainX = trainX.T
    # endregion

    U, subspace_mean, Y = PCA_Data(trainX, 2)

    print(Y)

    plt.plot(Y.T[0], Y.T[1], 'o')
    plt.show()

    skPCA = PCA(n_components=2)
    skPCA.fit(trainX)
    y = skPCA.transform(trainX)
    print(y)

    plt.plot(y.T[0],y.T[1], 'o')
    plt.show()