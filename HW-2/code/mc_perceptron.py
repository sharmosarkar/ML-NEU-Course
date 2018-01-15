'''
    Multi class classifier using multi-layer perceptron. 
    In this implementation there is 1 hidden layer with varing number of units in it.
    The _activation_function is sigmoid. In the output layer we use softmax to select the predicted class
'''


import scipy.io
import numpy as np
import matplotlib.pyplot as plt

DATA_LOCATION = 'Z:/ML/HW-2/code/data.mat'

class Neural_Network(object):

    def __init__(self): 
        self.learning_rate = 0.01
        self.regularizing_param = 0.01
        self.max_iterations = 1000
        self.class_cnt = 3

    def _backward_propagation (self,probs,total_inputs,a1,X,Y,W1,W2,b1,b2):
        # layer 3 to 2
        delta3 = probs
        delta3[range(total_inputs), Y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        # layer 2 to 1
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
        # Regularizing
        dW2 += self.regularizing_param * W2
        dW1 += self.regularizing_param * W1
        # update parameters
        W1 += -self.learning_rate * dW1
        b1 += -self.learning_rate * db1
        W2 += -self.learning_rate * dW2
        b2 += -self.learning_rate * db2
        return W1,b1,W2,b2

    def train (self,X,Y,hidden_unit_node_cnt):
        instances, dims = X.shape
        W1 = np.random.randn(dims, hidden_unit_node_cnt) / np.sqrt(dims)
        b1 = np.zeros((1, hidden_unit_node_cnt))
        W2 = np.random.randn(hidden_unit_node_cnt, self.class_cnt) / np.sqrt(hidden_unit_node_cnt)
        b2 = np.zeros((1, self.class_cnt))

        for i in range(self.max_iterations):
            current_model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
            # _forward_propagation propagation
            a1, probs = self._forward_propagation (current_model, X)
            # backpropagation
            W1,b1,W2,b2 = self._backward_propagation (probs,instances,a1,X,Y,W1,W2,b1,b2)        
        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


    ## for the forward propagation
    def _forward_propagation (self,model, x , is_prediction=False):
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        z1 = x.dot(W1) + b1
        a1 = self._activation_function ('sigmoid',z1)
        z2 = a1.dot(W2) + b2
        if is_prediction:
            return self._activation_function('softmax',z2)
        else:
            return a1, self._activation_function('softmax',z2)

    def predict(self,model, x):
        probs = self._forward_propagation (model, x,True)
        return np.argmax(probs, axis=1)   

    def _activation_function (self,type, z):
        if type=='sigmoid':
            return 1 / (1 + np.exp(-z))
        if type=='softmax':
            return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def get_input_data():
    data = scipy.io.loadmat(DATA_LOCATION)
    return data['X_trn'], data['Y_trn'].T, data['X_tst'], data['Y_tst'].T


def calc_accuracy(pred_lst,gold_lst):
    hits = 0.0
    for pred,gold in zip(pred_lst,gold_lst):
        if gold == pred:
            hits += 1
    return (hits*1.0 / len(pred_lst))*100

def plot_graph(x,y):
    print(x)
    print(y)
    plt.plot(x,y)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Number of units in Hidden Layer')
    plt.title ('Variation of accuracy wrt to number of units in hidden layer')
    plt.show()

def control_neural_network (x_trn, y_trn, x_tst, y_tst,hidden_unit_cnt_lst):
    NN = Neural_Network()
    accuracy_list = []
    for hidden_layer_node_cnt in hidden_unit_cnt_lst:
        model = {}
        np.random.seed(0)
        model = NN.train(x_trn, y_trn, hidden_layer_node_cnt)
        predictions = NN.predict(model, x_tst)
        accuracy = calc_accuracy(predictions,y_tst[0])
        accuracy_list.append(accuracy)
        print ('hidden units = ',hidden_layer_node_cnt,'\t accuracy = ',accuracy,'%')
    return accuracy_list


def main (file):
    X_train_data, Y_train_data, X_test_data, Y_test_data = get_input_data()
    hidden_layer_cnt_lst = [10, 20, 30, 50, 100]
    accuracy_list = control_neural_network (X_train_data, Y_train_data, X_test_data, Y_test_data,hidden_layer_cnt_lst)
    plot_graph(hidden_layer_cnt_lst,accuracy_list)


if __name__ == "__main__":
    main("Z:/ML/HW-2/code/data.mat")
