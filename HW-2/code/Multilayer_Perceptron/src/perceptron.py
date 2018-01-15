"""
Implement a Feed Forward Neural Network, with an input layer with S1 units, one hidden layer
with S2 units, and an output layer with S3 units using the backpropagation algorithm and the sigmoid
activation function. Run it on the dataset provided. Try different number of hidden nodes in
the hidden layer, S2 = 10, 20, 30, 50, 100, and report the classification results as a function of S2.
"""
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = sigmoid(z1)#np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

def ReadData(file_name):
    mat_contents = {}
    try:
        mat_contents = sio.loadmat(file_name)
    except Exception as e:
        print(('Exception : {0}').format(e))

    return mat_contents['X_trn'], mat_contents['Y_trn'], mat_contents['X_tst'], mat_contents['Y_tst']

if __name__ == "__main__":
    x_trn, y_trn, x_tst, y_tst = ReadData("Z:/ML/HW-2/code/Multilayer_Perceptron/src/data.mat")

    # input dataset
    X = np.array(x_trn)
    x_tst = np.array(x_tst)

    # output dataset
    y = np.array(y_trn).T
    y_tst = np.array(y_tst).T

    #plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    #plt.show()

    total_inputs = X.shape[0]
    input_nodes = X.shape[1]
    output_nodes = 3

    epsilon = 0.01  # learning rate for gradient descent
    reg_lambda = 0.01

    # nh = [10, 20, 30, 50, 100]
    s2_layer_dimension = [10, 20, 30, 50, 100]

    for dimension in s2_layer_dimension:

        model = {}

        np.random.seed(0)

        W1 = np.random.randn(input_nodes, dimension) / np.sqrt(input_nodes)
        b1 = np.zeros((1, dimension))
        W2 = np.random.randn(dimension, output_nodes) / np.sqrt(dimension)
        b2 = np.zeros((1, output_nodes))

        total_iterations = 2000

        # Gradient Descent
        for i in range(total_iterations):
            # Forward Propagation
            z1 = X.dot(W1) + b1
            a1 = sigmoid(z1)
            z2 = a1.dot(W2) + b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Backpropagation
            delta3 = probs
            delta3[range(total_inputs), y] -= 1
            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
            dW1 = np.dot(X.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += reg_lambda * W2
            dW1 += reg_lambda * W1

            # Gradient descent parameter update
            W1 += -epsilon * dW1
            b1 += -epsilon * db1
            W2 += -epsilon * dW2
            b2 += -epsilon * db2

            # Assign new parameters to the model
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        test_prediction = predict(model, x_tst)

        accuracy = 0.0

        for index in range(len(test_prediction)):
            if(test_prediction[index] == y_tst[0][index]):
                accuracy += 1

        accuracy = accuracy / len(test_prediction)

        print ('dimension = ',dimension,'\t accuracy = ',accuracy)

       # print("For {0} hidden nodes, Accuracy is {1}%").format(dimension,accuracy * 100)