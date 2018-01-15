import scipy.io as sio
from numpy.random import randint
from numpy import shape, zeros, sqrt, argmin, array, take, mean
import numpy as np
import matplotlib.pyplot as plt

def readMatlabFile(filename):
    '''
    :param filename: The name of the file whose content needs to be returned
    :return:
    '''
    mat_contents = {}
    try:
        mat_contents = sio.loadmat(filename)
    except Exception as e:
        print(('Exception : {0}').format(e.message))
    return mat_contents['X_Question2_3']

def ComputeClusterDistance(data, centroids):
    '''
    :param data: Input data set
    :param centroids: Centroid Data Set
    :return: cluster_indices which is array of cluster indices to which each data point belongs to,
             distances which is euclidean distance of each data point with the cluster to which it belongs to
    '''
    (n, d) = shape(data)
    cluster_indices = zeros(n, dtype=int)
    distances = zeros(n)
    for i in range(n):
        dist = np.sum((data[i] - centroids) ** 2, 1)
        cluster_indices[i] = argmin(dist)
        distances[i] = dist[cluster_indices[i]]

    return cluster_indices, sqrt(distances)

def GetNewCentroid(X, cluster_indices, number_of_clusters):

    newCentroid = []

    for cluster_value in range(number_of_clusters):

        # Get all ith cluster
        data_indices = np.where(cluster_indices == cluster_value)[0]

        centroid_datapoints = take(X, data_indices, 0)

        newCentroid.append(centroid_datapoints.mean(0))

    return np.array(newCentroid)

def KMeans_Data(X,centroid_points,thresh):
    centroids = array(centroid_points, copy=True)

    avg_dist = []

    diff = thresh + 1.

    number_of_clusters = centroids.shape[0]

    while diff > thresh:

        # compute membership and distances between X and code_book
        cluster_of_datapoints, dist_to_centroid = ComputeClusterDistance(X, centroids)

        # compute average distance
        avg_dist.append(mean(dist_to_centroid, axis=-1))

        if (diff > thresh):
            centroids = GetNewCentroid(X, cluster_of_datapoints, number_of_clusters)

        if len(avg_dist) > 1:
            diff = avg_dist[-2] - avg_dist[-1]

    return centroids, avg_dist[-1]

def plotData(X,C,closest):
    plt.scatter(X[:, 0], X[:, 1], c=closest)
    plt.scatter(C[:, 0], C[:, 1], c='r')
    plt.show()

if __name__ == "__main__":

    trainX = readMatlabFile("./input_files/data.mat")

    trainX = np.array(trainX)

    # No. of clusters
    k = 4
    best_dist = np.inf
    r = 100

    observations = trainX.T

    total_observation = observations.shape[0]

    threshold = 1e-5

    for i in range(r):

        random_indices = randint(0, total_observation, k)

        random_centroid_datapoints = take(observations, random_indices, 0)

        centroid, dist = KMeans_Data(observations, random_centroid_datapoints, threshold)

        if dist < best_dist:
            best_centroid = centroid
            best_dist = dist

    bestClusterIndice, dist = ComputeClusterDistance(observations, best_centroid)

    plotData(observations,best_centroid,bestClusterIndice)