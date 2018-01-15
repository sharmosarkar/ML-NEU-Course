from numpy import loadtxt, zeros, ones, array, linspace, logspace, mean, std, arange
#from pylab import scatter, show, title, xlabel, ylabel, plot, contour
from matplotlib.pyplot import scatter, show, title, xlabel, ylabel, plot, contour
import scipy.io as sio
from scipy import linalg
import numpy as np
from collections import OrderedDict

def chk_error (Y_data,y_predicted):
    error_test = zeros(shape=Y_data.size)
    for x in range(0,Y_data.size):
        #print ""
        error_test[x] = (Y_data[x] - y_predicted[x])**2
    return error_test

def predict_all (new_X_tst, final_W):
    final_W_transpose = np.transpose(final_W)
    return final_W.dot(np.transpose(new_X_tst))

def mean_square_error (new_X_tst, final_W, y_tst):
    y_pred = predict_all(new_X_tst, final_W)
    error_train = chk_error(y_tst,y_pred)
    return (sum(error_train)/y_tst.size)

def reshape_X (n,X_tst):
    m = X_tst.size
    #Add a column of ones to X (interception data)
    it = ones(shape=(m, n+1))
    # print ("it shape ",it.shape)
    it[:, 1] = X_tst
    ## the  actual X with the polynomial
    for counter in range(2,n+1):
        it[:, counter] = it[:, 1]**counter
    return it

def calc_W(final_X,y,l=0):
    iD = np.identity(final_X[0].size)
    lamda_cross_iD = l*(iD)
    final_X_transpose = np.transpose(final_X)
    X_trans_dot_X = np.add(final_X_transpose.dot(final_X),lamda_cross_iD)
    # print ("X_trans_dot_X",X_trans_dot_X)
    X_trans_dot_X_inverse = linalg.inv(X_trans_dot_X)
    X_trans_dot_X_inverse_dot_X_transpose = X_trans_dot_X_inverse.dot(final_X_transpose)
    final_W = X_trans_dot_X_inverse_dot_X_transpose.dot(y)
    return final_W

def ridge_calc(X_trn,Y_trn, K_fold, lam,n, X_tst, Y_tst):
    lam_mse_dict = OrderedDict()
    list_fold_X = np.array_split(X_trn,K_fold)
    list_fold_Y = np.array_split(Y_trn,K_fold)
    reshaped_list_X = []
    for item in list_fold_X:
        reshaped_list_X.append (reshape_X(n,item))
    reshaped_X_tst = reshape_X(n,X_tst)
    #print ("list_fold_X",list_fold_X)
    #print (len(reshaped_list_X),'   = len(reshaped_list_X)')
    #print (reshaped_list_X)
    #print("it_list",new_fold_X)
    #print ("list_fold_X",list_fold_X)

    ## starting K-fold 
    test_size = (X_trn.size/K_fold)
    X_trn_list= []
    # K_fold_test_X=zeros(shape=(0,(n+1)))
    # K_fold_trn_X=zeros(shape=(0,(n+1)))
    # K_fold_test_Y=zeros(shape=(0,1))
    # K_fold_trn_Y=zeros(shape=(0,))
    # print("k = ",K_fold)
    for l in lam:
        mean_Hold_Error = 0
        mse = []
        # print("For Lambda ->",l)
        ## Starting the K-fold
        for i in range(0,K_fold):
            # print("Iteration no.:",i+1)
            # K_fold_test_X=zeros(shape=(0,(n+1)))
            K_fold_trn_X=zeros(shape=(0,(n+1)))
            # K_fold_test_Y=zeros(shape=(0,1))
            K_fold_trn_Y=zeros(shape=(0,))
            K_fold_test_X = reshaped_list_X[i]
            K_fold_test_Y = list_fold_Y[i]
            for j in range(0,K_fold):
                if i != j:
                    #X_trn_list.append(reshaped_list_X[j])
                    K_fold_trn_X = np.concatenate((K_fold_trn_X,reshaped_list_X[j]),axis=0)
                    K_fold_trn_Y = np.concatenate((K_fold_trn_Y,list_fold_Y[j]),axis=0)

            final_W = calc_W(K_fold_trn_X,K_fold_trn_Y,l)
            # print ("final_W = ",final_W)
            mse_temp = mean_square_error(K_fold_test_X, final_W, K_fold_test_Y)
            mse.append(mse_temp)
            # print ("mean square error = ",mse)

        mean_Hold_Error = sum(mse)/len(mse)
        # print("mean_Hold_eror",mean_Hold_Error)
        lam_mse_dict[l] = mean_Hold_Error
    # print ("lam_mse_dict",lam_mse_dict)
    # print("best Lambda = ",min(lam_mse_dict, key=lam_mse_dict.get))
        # add mean_Hold_Error to lambda

    '''
    print ("K_fold_test_X",K_fold_test_X)
    print ("K_fold_trn_X",K_fold_trn_X)
    print ("K_fold_trn_X shape", K_fold_trn_X.shape)
    print ("K_fold_test_X shape", K_fold_test_X.shape)
    print ("Split element shape ",reshaped_list_X[2].shape)
    print ("Split element Type ",type(reshaped_list_X[2]))
    print ("K_fold_trn_X Type ",type(K_fold_trn_X))
    '''
    best_lambda = min(lam_mse_dict, key=lam_mse_dict.get)
    optimal_W = calc_W(reshape_X(n,X_trn),Y_trn,best_lambda)
    # final_W = calc_W(final_X,Y_trn,best_lambda)
    mean_square_error_trn = mean_square_error(reshape_X(n,X_trn), optimal_W, Y_trn)
    mean_square_error_tst = mean_square_error(reshaped_X_tst, optimal_W, Y_tst)    
    return best_lambda, optimal_W, mean_square_error_trn, mean_square_error_tst





#Load the dataset
mat_contents = sio.loadmat('Z:\ML\HW-1\Code\LinReg\LinearRegression\linear_regression.mat', squeeze_me=True)
# mat_contents = sio.loadmat('linear_regression.mat')
X_trn = mat_contents['X_trn']
Y_trn = mat_contents['Y_trn']
X_tst = mat_contents['X_tst']
Y_tst = mat_contents['Y_tst']
# print "X_trn",xTrans
# print "X_tst",X_tst
# print "Y_trn",Y_trn
# print "Y_tst",Y_tst
#print "matcontents",mat_contents
#data = loadtxt('ex1data1.txt', delimiter=',')
#print "data",data
#Plot the data
scatter(X_trn,Y_trn, marker='o', c='b')
#scatter(data[:, 0], data[:, 1], marker='o', c='b')
title('Profits distribution')
xlabel('X-Label')
ylabel('Y-label')
#show()

#X = data[:, 0]
#y = data[:, 1]
X = X_trn
# actual_X = ones(shape=())
y = Y_trn
#number of training samples
m = y.size

K = [2,5,10,y.size]
## n is the Dimension of Phi which is n-degree polynomial
n_dim = [2,5,10,20]

print ('Linear Regression')
for item in n_dim:
    print ('n = ',item)
    n = item
    #Add a column of ones to X (interception data)
    it = ones(shape=(m, n+1))
    new_X = ones(shape=(m, n))
    it[:, 1] = X
    ## the  actual X with the polynomial
    new_X[:, 0] = X 
    #print "it-original",it
    for counter in range(2,n+1):
        it[:, counter] = it[:, 1]**counter

    '''
    for counter in range(1,n):
        new_X[:, counter] = new_X[:, 0]**(counter+1)
    '''
    final_W_linear = zeros(shape=(1, n+1))
    final_X = it
    final_X_transpose = np.transpose(final_X)
    X_trans_dot_X = final_X_transpose.dot(final_X)
    X_trans_dot_X_inverse = linalg.inv(X_trans_dot_X)
    X_trans_dot_X_inverse_dot_X_transpose = X_trans_dot_X_inverse.dot(final_X_transpose)
    final_W_linear = X_trans_dot_X_inverse_dot_X_transpose.dot(y)
    #print ('final_X.shape   ',final_X.shape )
    #print ('final_X_transpose.shape ',final_X_transpose.shape)
    #print ('inv(X_transpose.dot(X)) ',X_trans_dot_X_inverse)
    print ('W = ',final_W_linear)
    # print ('Final W shape', final_W.shape)

    ## Check Train data Error
    y_predicted_train = predict_all (final_X,final_W_linear)
    error_train = chk_error(Y_trn,y_predicted_train)
    # print ('Train Data Error = ',error_train)
    print ('Sum Train Error = ',sum(error_train)/Y_trn.size)

    ## Test data calculation
    new_X_tst = reshape_X (n,X_tst)
    # print ('Final W shape', final_W.shape)
    # print ('Final Test X shape',new_X_tst.shape)
    y_predicted = predict_all (new_X_tst,final_W_linear)
    # print ('Predicted Y shape ', y_predicted.shape)
    # print ('y_predicted = ', y_predicted)
    # print ('y_tst = ', Y_tst)
    # Calculating test error
    #error_test = zeros(shape=y_predicted.shape)
    error_test = chk_error(Y_tst,y_predicted)
    # print ("Test Data Error = '",error_test)
    print ('Sum Test Error = ',sum(error_test)/Y_tst.size)


print ('Ridge Regression')
## Calculating Ridge
lam = []
for l in range(1,11):
    lam.append(l/100)
# print ("lambda ->",lam)
# print ("ridge calc")
# print ('X_trn shape     ',X_trn.shape)
N_Str = input("Enter the value for N")
N = int(N_Str)
k_dim = [2,5,10,N]

for i in n_dim:
    for j in k_dim:
        best_lambda, final_W, mean_square_error_trn, mean_square_error_tst = ridge_calc(X_trn,Y_trn, j, lam,i,X_tst,Y_tst)
        #final_W = calc_W(final_X,Y_trn,best_lambda)
        #mean_square_error_trn = mean_square_error(final_X, final_W, Y_trn)
        #mean_square_error_tst = mean_square_error(new_X_tst, final_W, Y_tst)
        print("For ",i," dimension, and k_fold value as ",j," the best lambda is:",best_lambda, " w is ",final_W,\
            " y trn error is ",mean_square_error_trn," y tst error is ", mean_square_error_tst)
print ("And with this, we are done...")

'''
#Some gradient descent settings
# iterations = 1500
iterations = 2
alpha = 0.01
# print "X is",X
# x, mean_r, std_r = feature_normalize(new_X)
#
#print ("x",new_X)
#print ('it',it)
# print "mean_r",mean_r
# print "std_r",std_r

#compute and display initial cost
print ("computed cost",compute_cost(it, y, theta))

theta, J_history = gradient_descent(it, y, theta, alpha, iterations)

print ("theta",theta)
print ("J_history",J_history)
#Predict values for population sizes of 35,000 and 70,000
#predict1 = array([1, 3.5]).dot(theta).flatten()
#print 'For population = 35,000, we predict a profit of %f' % (predict1 * 10000)
#predict2 = array([1, 7.0]).dot(theta).flatten()
#print 'For population = 70,000, we predict a profit of %f' % (predict2 * 10000)

#Plot the results
result = it.dot(theta).flatten()
#plot(data[:, 0], result)
plot(X_trn, result)
scatter(X_trn,Y_trn, marker='o', c='b')
show()


#Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100)
theta1_vals = linspace(-1, 4, 100)


#initialize J_vals to a matrix of 0's
J_vals = zeros(shape=(theta0_vals.size, theta1_vals.size))

#Fill out J_vals
for t1, element in enumerate(theta0_vals):
    for t2, element2 in enumerate(theta1_vals):
        thetaT = zeros(shape=(2, 1))
        thetaT[0][0] = element
        thetaT[1][0] = element2
        J_vals[t1, t2] = compute_cost(it, y, thetaT)

#Contour plot
J_vals = J_vals.T
#Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('theta_0')
ylabel('theta_1')
scatter(theta[0][0], theta[1][0])
show()
'''