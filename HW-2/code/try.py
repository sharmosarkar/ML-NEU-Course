import numpy as np
"""
Implement a Feed Forward Neural Network, with an input layer with S1 units, one hidden layer
with S2 units, and an output layer with S3 units using the backpropagation algorithm and the sigmoid
activation function. Run it on the dataset provided. Try different number of hidden nodes in
the hidden layer, S2 = 10, 20, 30, 50, 100, and report the classification results as a function of S2.
"""
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j

print (y)
print (y.shape)
print (type (y))
print (X.shape)
print (type (X))
W = 0.01 * np.random.randn(D,K)
b = np.zeros((1,K))
print (W)


	# def plotGraph(x,y):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     class0 = where(y == 0)
    #     class1 = where(y == 1)
    #     class2 = where(y == 2)
    #     # print("val1",len(x[class0, 0][0]))
    #     # print("val2",x[class0, 1][0])
    #     # print("val3",x[class0, 2][0])
    #     ax.scatter(x[class0, 0][0], x[class0, 1][0], x[class0, 2][0], marker='o', c='b')
    #     ax.scatter(x[class1, 0][0], x[class1, 1][0], x[class1, 2][0], marker='x', c='r')
    #     ax.scatter(x[class2, 0][0], x[class2, 1][0], x[class2, 2][0], marker='+', c='g')
    #     ax.set_title('3D Scatter Plot with X1,X2,X3')
    #     ax.set_xlabel('X1')
    #     ax.set_ylabel('X2')
    #     ax.set_zlabel('X3')
    #     # plt.show()