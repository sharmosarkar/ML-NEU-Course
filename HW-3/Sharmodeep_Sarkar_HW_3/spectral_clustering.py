import scipy.io as sio
from numpy import shape, zeros, sqrt
import numpy as np
import math
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import k_means 					## k-means implemented in question Qs2 of this HW

DATA_LOCATION = 'Z:/ML/HW-3/data/data.mat'

class Spectral_Clustering(object):

	def __init__(self, datapoints, k): 
		self.k = k
		self.datapoints = datapoints
		self.datapoint_count = datapoints.shape[0]


	## computing affinity matrix depending on the euclidean distances of the datapoints from each other
	def compute_affinity_matrix(self,sigma):
		(n, d) = shape(self.datapoints)
		affinity_matrix = np.zeros((n,n))## initialize the affinity_matrix with 0
		sigma_squared = sigma*sigma ## NOTE: acc to question this is sigma only,
									##		 but gaussian uses sigma^2, results of clustering don't change
									##       hence I would be using sigma^2 in the gaussian denominator
		for i in range(n):
			for j in range(n):
				if i != j:  
					## compute distance from a point to all other points, but not to itself
					euclidean_dist = sqrt(np.sum((self.datapoints[i] - self.datapoints[j]) ** 2))
					euclidean_dist = (-1)*euclidean_dist
					affinity_weight = euclidean_dist/ sigma_squared
					affinity_matrix[i][j] = math.exp(affinity_weight)
		return affinity_matrix




	def normalize_matrix(self,v):
		(n,d) = shape(v)
		normalized_matrix = np.zeros((n,d))
		v_squared = v**2
		for i in range(n):
			row_sum = sqrt(np.sum(v_squared[i]))
			for j in range (d):
				if row_sum != 0:
					normalized_matrix[i][j] = v[i][j]/row_sum
		return normalized_matrix



	def compute_diagonal_matrix(self,A,n):
		diagonal_mat = np.zeros((n,n))
		for i in range(n):
			row_sum = np.sum(A[i])
			diagonal_mat[i][i] = row_sum
		return diagonal_mat




	def run_spectral_clustering(self,sigma):
		(n,d) = shape(self.datapoints)
		A = self.compute_affinity_matrix(sigma)		## computing the affinity matrix
		D = self.compute_diagonal_matrix(A,n) 	## computing the diagonal matrix
		L = D-A									## computing the Graph Laplacian from 
		## get the eigenvalues and eigenvectors of L
		## returns :: w : (..., M) ndarray, the eigenvalues in ascending order, 
		##									each repeated according to its multiplicity.
		## 			  v : {(..., M, M) ndarray, (..., M, M) matrix}, 
		##							The column v[:, i] is the normalized eigenvector corresponding to the eigenvalue w[i]
		w, v = eigh(L, eigvals=(0,self.k-1))
		mormalized_v = self.normalize_matrix(v)					## normalize the eigenvalue matrix
		'''
		K_means_obj = k_means.K_means(mormalized_v)			## initializing the k_means class
		optimul_cluster_indice, distance, optimul_centroid = K_means_obj.run_k_means()		## get cluster indices and centroids
		return optimul_centroid, optimul_cluster_indice
		'''
		from sklearn.cluster import k_means 					## for testing
		newK = KMeans(n_clusters=self.k).fit(mormalized_v)
		return newK.cluster_centers_, newK.labels_
		

## plot input and output graphs
def plot_graph(title, datapoints, cluster_indice=None, centroid=None):
	if cluster_indice != None:
		## Plot results (clustered dataset)
		plt.scatter(datapoints[:, 0], datapoints[:, 1], c=cluster_indice)
		plt.scatter(centroid[:, 0], centroid[:, 1], c='r')
		plt.title (title)
		plt.show()
	else:
		## Plot input dataset (without clusters)
		plt.scatter(datapoints[:,0] , datapoints[:,1], c='m')
		plt.title(title)
		plt.show()


## read input 
def get_input_data():
    data = sio.loadmat(DATA_LOCATION)
    return np.array(data['X_Question2_3']).T


def main():
	datapoints = get_input_data()
	plot_graph ('Unclustered Data Points', datapoints)
	k = 4
	sigma_list =  [0.001, 0.01, 0.1, 1]
	spectral_obj = Spectral_Clustering(datapoints,k)
	for sigma in sigma_list:
		## get cluster indices and centroids
		optimul_centroid, optimul_cluster_indice = spectral_obj.run_spectral_clustering(sigma)	
		plot_graph('Clustered data for sigma = '+str(sigma), datapoints, optimul_cluster_indice,optimul_centroid)


if __name__ == '__main__':
	main()
