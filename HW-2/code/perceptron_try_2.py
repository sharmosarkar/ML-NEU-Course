
import numpy as np
import math


class Neural_Network(object):
    def __init__(self):        
        #Define Hyperparameters
        self.inputLayerSize = 3 
        self.outputLayerSize = 3
        self.hiddenLayerSize = 10
        self.max_iteration = 2000

        self.training_sample = 189
        self.one_hot_classes = [[1,-1,-1],[-1,1,-1],[-1,-1,1]]
        self.class_count = 3
        
        #Weights (parameters)
        # self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        # self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)

        self.W1 = np.zeros ((self.inputLayerSize+1,self.hiddenLayerSize))
        self.W2 = np.zeros ((self.hiddenLayerSize+1, self.outputLayerSize))

        # self.W1.fill(0)
        # self.W2.fill(0)
        # print('self.W1',type(self.W1) , self.W1)
        # print('self.W2',self.W2)
        
    def train_network (self, X_trn, Y_trn):
        m,n = X_trn.shape
        self.X = np.ones(shape=(m,n+1))
        self.X[:,:-1] = X_trn
        
        m,n = self.W1.shape
        w1_reshaped = np.ones(shape=(m,n+1))
        w1_reshaped[:,:-1] = self.W1
        self.W1 = w1_reshaped

        for k in range (self.max_iteration):
            # print(self.W1.shape)
            # print(self.X.shape)
            print ('iteration ', k)
            self.z2 = np.dot(self.X, self.W1)
            # print (self.z2.shape)
            self.a2 = self.sigmoid(self.z2)
            # print (self.a2)
            # print (self.a2.shape)
            if k==0 :
                f,g = self.a2.shape
                self.a2_with_bias = np.ones(shape=(f,g+1))
                self.a2_with_bias[:,:-1] = self.a2
            
            self.z3 = np.dot(self.a2, self.W2)
            self.a3 = self.soft_max(self.z3)
            # print (self.a3)
            # print (self.a3.shape)
            # print (self.z3.shape)
            self.yhat = np.argmax (self.a3, axis=1)
            # print (self.classes)
            # print (self.classes.shape)
            self.calc_backpropagation(X_trn,Y_trn, self.a3)


    def calc_backpropagation (self,X_trn,Y_trn, probs):
        # some hyperparameters
        step_size = 1e-0
        reg = 0.01#1e-3 # regularization strength
        # sum_val = 0
        # for i in range(self.training_sample):
        #     T =  self.one_hot_classes[Y_trn[i]]
        #     for j in range(self.class_count):
        #         sum_val += (T*math.log( ,10))
        # m,n = self.W1.shape
        # w1_reshaped = np.ones(shape=(m,n+1))
        # w1_reshaped[:,:-1] = self.W1
        # self.W1 = w1_reshaped

    
        #########################################
        # a = self.one_hot_classes
        # b = np.full((len(Y_trn),3),1)
        # for i in range (0,len(Y_trn)):
        #     if Y_trn[i]== 0:
        #         b[i] = a[0]
        #     elif Y_trn[i] == 1:
        #         b[i] = a[1]
        #     else:
        #         b[i] = a[2]
        # T = b
        # # print (type(T))
        # # print (T.shape)
        # log_a3 = np.log10(self.a3)
        # # print (log_a3.shape)
        # entrophy = np.zeros((189,1))
        # for i in range (189):
        #     entrophy[i] = np.dot(log_a3[i],T[i].T)
        # # print (entrophy)
        # # print (entrophy.shape)
        ##############################################################
        
        # compute the loss: average cross-entropy loss and regularization
        corect_logprobs = -np.log(probs[range(self.training_sample ),Y_trn])
        data_loss_2 = -np.sum (corect_logprobs)/self.training_sample
        reg_loss_2 = 0.5*reg*np.dot(self.W2.T,self.W2)
        loss = data_loss_2 + reg_loss_2
        #print ('loss', loss.shape)
        
        dscores = self.a3
        dscores[range(self.training_sample),Y_trn] -= 1
        dscores /= self.training_sample
        #   print (dscores)
        #print (dscores.shape)

        dW_2 = np.dot(self.a2.T, dscores)
        # print (dW_2)
        db_2 = np.sum(dscores, axis=0, keepdims=True)
        # print (db_2)
        # print (self.a2.shape)
        # print (self.W2.shape)
        # print (dscores.shape)
        print (dW_2.shape , 'dW_2')
        dW_2 += reg*self.W2 # don't forget the regularization gradient
        # print (dW_2)
        # print (db_2)
        # perform a parameter update
        self.W2 += -step_size * dW_2


        # next backprop into hidden layer
        dhidden = np.dot(dscores, self.W2.T)

        m,n = self.z2.shape
        z2_reshaped = np.ones(shape=(m,n+1))
        z2_reshaped[:,:-1] = self.z2
        # print (dhidden.shape , 'dhidden')
        # print (loss.shape , 'losss')
        # print (self.W2.shape , 'W2')
        # print (self.sigmoidPrime(z2_reshaped).shape , 'sigmoidPrime(z2_reshaped)')
        delta2 =  self.sigmoidPrime(self.z2)
        # print (self.X.shape,"x")
        # print (delta2.shape,"delta2")
        dW_1 = np.dot(self.X.T, delta2) 
        # print (delta2)
        # print (dW_1.shape)
        # print (self.W1.shape)
        # m,n = self.z2.shape
        # z2_reshaped = np.ones(shape=(m,n+1))
        # z2_reshaped[:,:-1] = self.z2

        self.W1 += -step_size * dW_1
        
        print ('W1\n ',self.W1,'')
        print ('W2\n ', self.W2)

        # print (self.W1, "W1")
        # print (self.W2, "W2")



    def soft_max (self, z):
        #################
        #probs =  np.exp(z) / np.sum(np.exp(z), axis=0)
        #################
        exp_scores = np.exp(z)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

        
    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    
    def predict (self, X_tst):
        m,n = X_tst.shape
        X = np.ones(shape=(m,n+1))
        X[:,:-1] = X_tst
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)

        self.z3 = np.dot(self.a2, self.W2)
        self.a3 = self.soft_max(self.z3)

        self.yhat = np.argmax (self.a3, axis=1)
        # print (self.yhat)
        return self.yhat


    def get_accuracy (self, predictions, Y):
        hits = 0
        if len (predictions) != len (Y):
            print ('Lengths do not match')
            print (' PRedictions length = ',len(predictions), '\tGold Label length =',len(Y))
            return
        for pred, gold in zip(predictions, Y):
            if pred == gold:
                hits += 1
        return (hits*1.0/len(predictions))*100


    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J
        
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        # print (delta3.shape)
        # print(self.W2.T.shape)
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        
        return dJdW1, dJdW2
    
    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)
            
            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0
            
        #Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad 
        
##  Training
from scipy import optimize


class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))   
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        return cost, grad
        
    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res


import scipy.io as sio
def main():
    #Load the dataset
    mat_contents = sio.loadmat('Z:/ML/HW-2/code/data.mat', squeeze_me=True)
    # mat_contents = sio.loadmat('linear_regression.mat')
    X_trn = mat_contents['X_trn']
    Y_trn = mat_contents['Y_trn']
    X_tst = mat_contents['X_tst']
    Y_tst = mat_contents['Y_tst']

    NN = Neural_Network()
    NN. train_network (X_trn , Y_trn)
    predictions = NN. predict(X_tst)
    # print (len(X_tst))
    # print (len(Y_tst))
    print (predictions)
    print (Y_tst)
    accuracy = NN.get_accuracy(predictions.tolist(), Y_tst.tolist())
    print ('Accuracy = ',accuracy,'%')

    # T = trainer(NN)
    # T.train(X_trn,Y_trn)

    # print(NN.forward(X_tst))
    # print(Y_tst)


if __name__ == '__main__':
    main()