## Author :: Sharmodeep Sarkar
## The approach has been discussed with other ML Classmates 

from numpy import loadtxt, zeros, ones, array, linspace, logspace, mean, std, arange
from matplotlib.pyplot import scatter, show, title, xlabel, ylabel, plot, contour
import scipy.io as sio
from scipy import linalg
import numpy as np

data_File_Path = 'Z:\ML\HW-1\Code\LinReg\LinearRegression\linear_regression.mat'

## converting x into phii(x) and stacking them up into an array called X
def reshape (n,X):
    m = X.size
    #Add a column of ones to X (interception data)
    reshaped = ones(shape=(m, n+1))
    reshaped[:, 1] = X 
    ## converting X into phi(x) where phi is of degree n
    for counter in range(2,n+1):
        reshaped[:, counter] = reshaped[:, 1]**counter
    return reshaped

## returns the predictions of Y values corresponding to the X values
def predict (X, W):
    return W.dot(np.transpose(X))

## calculates the squared errors of the predictions wrt the gold (original Y)
def calc_error (gold,prediction):
    error= zeros(shape=gold.size)
    for x in range(0,gold.size):
        error[x] = (gold[x] - prediction[x])**2
    return error

## calculates the mean square error
def mean_square_error (X, W, Y):
    prediction = predict(X, W)
    error_train = calc_error(Y,prediction)
    return (sum(error_train)/Y.size)


## calculating W = inverse(transpose(X).X + lambda.Id).transpose(X).Y
## when linear regression calls this function, the lambda is 0
def calc_W(X,Y,lam=0):
    iD = np.identity(X[0].size)
    X_transpose = np.transpose(X)
    return linalg.inv(np.add(X_transpose.dot(X),lam*(iD))).dot(X_transpose).dot(Y)


## calculation for ridge regression
def ridge_regression_calc(X_trn,Y_trn, K, lambda_list, n, X_tst, Y_tst):
    ## Splitting up the training data for K-fold validation
    X_list = np.array_split(X_trn,K)
    Y_list = np.array_split(Y_trn,K)
    ## Reshaping the training and test data
    reshaped_X_list = []
    for item in X_list:
        reshaped_X_list.append (reshape(n,item))
    reshaped_X_tst = reshape(n,X_tst)
    ## Start of K-fold validation 
    lambda_vs_mse = {}  ## stores lambda vs mse for each of the permutations of the K-fold setting
    for lam in lambda_list:
        mse_list = [] ##the list of mean square errors for each of the permutations of the K-fold setting for current lambda
        ## Starting the K-fold Cross validation
        for i in range(0,K):
            ## reshaping and making training data for the current k-fold iteration
            X_trn_K=zeros(shape=(0,(n+1)))
            Y_trn_K=zeros(shape=(0,))
            X_tst_K = reshaped_X_list[i]
            Y_tst_K = Y_list[i]
            for j in range(0,K):
                if i != j:
                    X_trn_K = np.concatenate((X_trn_K,reshaped_X_list[j]),axis=0)
                    Y_trn_K = np.concatenate((Y_trn_K,Y_list[j]),axis=0)
            ## calculate W with the current k-fold training data
            W = calc_W(X_trn_K,Y_trn_K,lam)
            ## calc mse for the current k-fold test data and add it to the list of mse's
            mse_list.append(mean_square_error(X_tst_K, W, Y_tst_K))
        ## calculating mean hold error for the current lambda and current k-fold value
        mhe = sum(mse_list)/len(mse_list)
        ## storing mse in the dict, to be used later to calculate lambda
        lambda_vs_mse[lam] = mhe
    ## find the best lambda for the current k-fold value and the current degree of polynomial
    lambda_optimal = min(lambda_vs_mse, key=lambda_vs_mse.get)
    ## calculate W using the optimal_lambda for the current setting
    W_optimal = calc_W(reshape(n,X_trn),Y_trn,lambda_optimal)
    ## calculate the mean square erors for the test and the training sets
    mse_trn = mean_square_error(reshape(n,X_trn), W_optimal, Y_trn)
    mse_tst = mean_square_error(reshaped_X_tst, W_optimal, Y_tst)    
    return lambda_optimal, W_optimal, mse_trn, mse_tst


## a wrapper function for the ridge regression
def ridge_regression (X_trn, Y_trn, X_tst, Y_tst,n_lst,lamda_lst,K):
    ## iterating for each degree of polynomial equation in the n_list
    for item in n_lst:
        ## iterating for each of the K values for K-fold cross validation
        for val in K:
            lambda_optimal, W_optimal, mse_trn, mse_tst =\
                 ridge_regression_calc(X_trn,Y_trn, val, lamda_lst ,item,X_tst,Y_tst)
            print('n = ',item,'K = ',val,'Optimal Lambda = ',lambda_optimal,'W = ',W_optimal,\
                'Training MSE = ',mse_trn,'Test MSE = ',mse_tst)


## calculation of linear regression
def linear_regression(X_trn, Y_trn, X_tst, Y_tst,n_lst):
    ## iterating for each degree of polynomial equation in the n_list
    for item in n_lst:
        print ('n = ',item)
        n = item
        ## making the X matrix according to the polynomal phii
        X = reshape (n,X_trn)
        ## calculating the W
        W_linear = calc_W(X,Y_trn)
        print ('W = ',W_linear)
        ## Check Train data Error
        error_train =  mean_square_error(X,W_linear,Y_trn)
        print ('Training Data MSE =',error_train)
        ## Test data calculation
        reshaped_X_tst = reshape (n,X_tst)
        ## Checking error in test data
        error_test = mean_square_error(reshaped_X_tst, W_linear, Y_tst)
        print ('Test Data MSE = ',error_test)


## draw data plot on canvas
def plot_data_points(X_trn,Y_trn):
    scatter(X_trn,Y_trn, marker='x', c='b')
    title('Training Data')
    xlabel('Predictor Variable')
    ylabel('Target Variable')
    #show()
    print('Data points Plotted !!')

def main():
    #Load the dataset
    mat_contents = sio.loadmat(data_File_Path, squeeze_me=True)
    X_trn = mat_contents['X_trn']
    Y_trn = mat_contents['Y_trn']
    X_tst = mat_contents['X_tst']
    Y_tst = mat_contents['Y_tst']

    ## Plotting data points
    plot_data_points(X_trn,Y_trn)

    ## n is the Dimension of Phi which is n-degree polynomial
    n_lst = [2,5,10,20]
    ## Values of K for K_Fold validation (used in Ridge Regression)
    K = [2,5,10,Y_trn.size]

    ## Starting Linear Regression
    print ('\n\nLinear Regression')
    linear_regression(X_trn, Y_trn, X_tst, Y_tst,n_lst)    

    ## making list of lambda (Ridge Regression regularization factor)   
    lamda_lst = []
    for lam in range(1,10):
        lamda_lst.append(lam/100)
    
    ## Starting Ridge Regression
    print ('\n\nRidge Regression')  
    ridge_regression (X_trn, Y_trn, X_tst, Y_tst,n_lst,lamda_lst,K)

    
if __name__ == "__main__":
    main()