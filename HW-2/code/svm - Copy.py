import numpy as np
import scipy.io
from numpy import loadtxt, where
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1

import sys
def readInput():
    print("Reading input from local file...!")
    data = scipy.io.loadmat('Z:/ML/HW-2/code/data.mat')
    X_train_data = data['X_trn']
    Y_train_data = data['Y_trn']
    X_test_data = data['X_tst']
    Y_test_data = data['Y_tst']
    print("Done reading file and storing data in array!")
    # print(X_train_data)
    # plotGraph(X_train_data,Y_train_data)
    return X_train_data, Y_train_data, X_test_data, Y_test_data

    # print(Y_train_data)
    # print(X_test_data)
    # print(Y_test_data)

def selectJrand(i, m):
    j = i  # we want to select any J not equal to i
    while (j == i):
        j = int(np.random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def plotGraph(x,y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    class0 = where(y == 0)
    class1 = where(y == 1)
    class2 = where(y == 2)
    # print("val1",len(x[class0, 0][0]))
    # print("val2",x[class0, 1][0])
    # print("val3",x[class0, 2][0])
    ax.scatter(x[class0, 0][0], x[class0, 1][0], x[class0, 2][0], marker='o', c='b')
    ax.scatter(x[class1, 0][0], x[class1, 1][0], x[class1, 2][0], marker='x', c='r')
    ax.scatter(x[class2, 0][0], x[class2, 1][0], x[class2, 2][0], marker='+', c='g')
    ax.set_title('3D Scatter Plot with X1,X2,X3')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    # plt.show()

def smoSimple_1(dataIn, classLabels, C, tolerance, maxIter):
    # print("dataIn =",  dataIn)
    # print("classLabels =", classLabels)
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
            #            fXi = evaluate(i, alphas, labelMat, dataMatrix, bias)
            #            Ei = fXi - float(labelMat[i])
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + bias
            Ei = fXi - float(labelMat[i])  # if checks if an example violates KKT conditions

            # Check if we can optimize (alphas always between 0 and C)
            if ((labelMat[i] * Ei < -tolerance) and (alphas[i] < C)) or \
                    ((labelMat[i] * Ei > tolerance) and (alphas[i] > 0)):

                # Select a random J
                j = selectJrand(i, m)

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
                # If the two correspond, then there is nothing
                # we can really do
                if L == H:
                    print
                    "L is H"
                    continue

                # Calculate ETA
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                      dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print
                    "eta is bigger than 0"
                    continue

                # Update J and I alphas
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                # If alpha is not moving enough, continue..
                if abs(alphas[j] - alpha_old_j) < 0.00001:
                    print
                    "Alpha not moving too much.."
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
                print
                "iter: %d i:%d, pairs changed %d" % (
                    iter, i, alphaPairsChanged
                )

            if alphaPairsChanged == 0:
                iter += 1
            else:
                iter = 0
            print
            "Iteration number: %s" % iter

        # print "Alphas:"
        #        print alphas[alphas>0]
        #        print "Bias:"
        #        print bias
        return alphas, bias

# Accuracy
def runSVMFunction(alpha, b, X, Y, dPoint):
    val = 0
    i = 0
    dPoint = dPoint.T

    for point in X:
        part1 = alpha[i] * Y[i]
        part2 = X[i] * dPoint
        out = part1 * part2
        val += out
        i += 1
    return (val + b)


# Predict accuracy on testing data points
def getAcc(alpha, b, X, Y, xTest, yTest):
    acc = 0
    i = 0
    for val in yTest:
        print(val)
        predList = list()
        for label in range(3):
            pred = runSVMFunction(alpha[label],
                                  b[label], X, Y, np.matrix(xTest[i]))

            # print "******"
            # print 'Predicted Value',pred
            predList.append(pred)
        pred = predList.index(max(predList))
        print('Predicted Label', pred, 'Actual Label', val)
        if pred == val:
            acc += 1
        i += 1
    return acc / float(len(yTest))

def main():
    X, Y, X_test, Y_test = readInput()
    alphaList = list()
    bList = list()
    models = list()
    # s = SMO(X,Y,1) #Regularization parameter
    # alpba,b = s.startCalculation()
    # print(alpba,"\n")
    # print(b)
    # alpha, bias = SMO_SVM.smoSimple_1(X,Y,10,0.0001,100)

    def updateLabelList(Y, label):
        # print("label is", label)
        # for i in Y:
        #     if (i==label):
        #        labels = 1
        #     else:
        #         labels = -1
        labels = [1 if i == label else -1 for i in Y]
        # print(labels)
        return labels

    def showFinal(x,y):
        print(x)
        print(y)
        plt1.plot(x,y)
        plt1.show()

    def update_C():
        accuracyList = list()
        C_values = [0.01,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0]
        for c in C_values:
            print("Chosen:",c)
            alphaList = list()
            bList = list()
            models = list()

            for label in range(3):
                # print("labels", np.array(updateLabelList(Y, label)))
                labels = np.array(updateLabelList(Y, label))
                # print ("===============");
                # print("JAI labels", Y)
                print ("===============");
                sys.stdout.flush()
                # Running for different ranges of C
                alpha, b = smoSimple_1(X, np.mat(labels).T, c, 0.001, 10000)
                # print("====Sarita====");
                sys.stdout.flush()

                # print(alpha, b)
                alphaList.append(alpha)
                bList.append(b)
                models.append((label, b, alpha))
            accuracy = getAcc(alphaList, bList, X, Y, X_test, Y_test)
            accuracyList.append(accuracy *100)
            del alphaList,bList,models
            print("Accuracy is: ", accuracy * 100)
        return C_values, accuracyList


    x,y = update_C()
    showFinal(x,y)


if __name__ == "__main__":
    main()
