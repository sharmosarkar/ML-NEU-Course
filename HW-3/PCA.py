'''
PCA
Implementing the PCA algorithm. The input to the algorithm must be the
input data matrix X ∈ R^(D×N) , where D is the ambient dimension and N is the number of points,
as well as the dimension of the low-dimensional representation, d. The output of the algorithm
must be the base of the low-dimensional subspace U ∈ R^(D×d), the mean of the subspace µ ∈ R^D
and the low-dimensional representations Y ∈ R^(d×N) . Applying the PCA algorithm for d = 2 to the
provided data and plotting the 2-dimensional data points in R^2
'''

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

DATA_LOCATION = 'Z:/ML/HW-3/data/data.mat'


class PCA (object):
	def __init__(self,input_data): 
		self.data = input_data


	## returns the PCA components
	def run_PCA (self,reduced_dimension_space):
		d = reduced_dimension_space
		## calculate mean and deviation of high-dimension data
		mean_data = np.mean(self.data, axis=0)
		deviation_data = self.data-mean_data
		## Singular Value Decomposition.
		## Factors the matrix deviation_data as u * np.diag(s) * v, where u and v are unitary and s is a 1-d array of a‘s singular values
		## deviation_data  : matrix of shape (M, N)
		## u : { (..., M, M), (..., M, K) } array, Unitary matrices. 
		## s : (..., K) array
		## v : { (..., N, N), (..., K, N) } array, Unitary matrices. 
		# Note:  deviation_data = U S V.H , the v returned by this function is V.H and u = U.
		# 	   If U is a unitary matrix, it means that it satisfies U.H = inv(U).
		# 	   The rows of v are the eigenvectors of deviation_data.H deviation_data
		# 	   The columns of u are the eigenvectors of deviation_data   deviation_data.H
		# 	   For row i in v and column i in u, the corresponding eigenvalue is s[i]**2.	
		## using the internal library  np.linalg.svd() 	for singular value decomposition according to Piazza @140
		u, s, v = np.linalg.svd(self.data)		
		required_components = v[:d]
		reduced_dimension_datapoints = np.dot(deviation_data, required_components.T)

		return reduced_dimension_datapoints



def plot_graph(title, datapoints):
		plt.scatter(datapoints[:,0] , datapoints[:,1], c='m')
		plt.title(title)
		plt.show()



def get_input_data():
    data = sio.loadmat(DATA_LOCATION)
    return np.array(data['X_Question1']).T

def main():
	datapoints = get_input_data()
	# print (datapoints.shape)
	# print (datapoints[1].shape)
	# plot input datapoints
	plot_graph ('Input High-Dimensional(D=40) DataPoints', datapoints)
	## get PCs for d = 2
	PCA_obj = PCA(datapoints)
	d = 2
	reduced_dimension_datapoints = PCA_obj.run_PCA(d)
	print (reduced_dimension_datapoints)
	plot_graph ('Output Low-Dimensional(d=2) DataPoints', reduced_dimension_datapoints)

	## verifying my results using sklearn
	# from sklearn import decomposition
	# pca = decomposition.PCA(n_components=2)
	# pca.fit(datapoints)
	# datapoints = pca.transform(datapoints)
	# plot_graph ('Output Low-Dimensional(d=2) DataPoints', datapoints)


if __name__ == '__main__':
	main()