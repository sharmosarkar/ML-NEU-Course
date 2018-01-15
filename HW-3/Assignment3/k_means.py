'''
K-means
'''
import scipy.io
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt

DATA_LOCATION = 'Z:/ML/HW-3/data/data.mat'


class K_means(object):

	def __init__(self, datapoints): 
		self.k = 4
		self.tolerance = 1e-5
		self.r = 150
		self.datapoints = datapoints
		self.datapoint_count = datapoints.shape[0]



	## returns cluster indices to which each data point belongs to
	##			and the euclidean distance of each data point to the cluster to which it belongs to
	def compute_cluster_membership(self,centroids):
		cluster_indices = np.zeros(self.datapoint_count, dtype=int)
		euclidean_distances = np.zeros(self.datapoint_count)
		for i in range(self.datapoint_count):
			datapoint_distance = np.sum((self.datapoints[i] - centroids) ** 2, 1)
			cluster_indices[i] = np.argmin(datapoint_distance)
			euclidean_distances[i] = datapoint_distance[cluster_indices[i]]

		return np.sqrt(euclidean_distances), cluster_indices


	def compute_new_centroid (self,clusters,cluster_cnt):
		computed_new_centroids = []
		for cluster_number in range(cluster_cnt):
			datapoint_indices = np.where(clusters == cluster_number)[0]
			current_centroids = np.take(self.datapoints, datapoint_indices, 0)
			computed_new_centroids.append(current_centroids.mean(0))

		return np.array(computed_new_centroids) 



	def k_means_algo(self,centroids):
		cluster_cnt = centroids.shape[0]
		dist = []	## holds the average distances 
		difference = self.tolerance+1
		while difference > self.tolerance:
			## check membership of each datapoint to each of the formed clusters till this point
			euclidean_distance_from_centroid, clusters = self.compute_cluster_membership(centroids)
			# get the average distances
			dist.append(np.mean(euclidean_distance_from_centroid, axis=-1))
			## new centroid required, as the difference of distances is greater than tolerance			
			if (difference > self.tolerance):
				centroids = self.compute_new_centroid (clusters,cluster_cnt)
			## calculate the difference in distances
			if len(dist) > 1:
				difference = dist[-2] - dist[-1]

		return dist[-1], centroids



	def run_k_means(self):
		least_distance = np.inf
		for i in range(self.r):
			rand_datapoint_indices = randint(0, self.datapoint_count, self.k)
			rand_centroids = np.array(np.take(self.datapoints, rand_datapoint_indices, 0))
			distance , centroid = self.k_means_algo(rand_centroids)
			## we want the clusters (ie. cluster centroids) that have least distances 
			if (distance<least_distance):
				least_distance = distance
				optimul_centroid = centroid

		distance, optimul_cluster_indice = self.compute_cluster_membership(optimul_centroid)
		return optimul_cluster_indice, distance, optimul_centroid


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





def get_input_data():
    data = scipy.io.loadmat(DATA_LOCATION)
    return data['X_Question2_3'].T

def main ():
    datapoints = get_input_data()
    plot_graph ('Unclustered Data Points', datapoints)
    K_means_obj = K_means(datapoints)
    optimul_cluster_indice, distance, optimul_centroid = K_means_obj.run_k_means()
    plot_graph('Clustered Data Points',datapoints, optimul_cluster_indice,optimul_centroid)


if __name__ == '__main__':
	main()